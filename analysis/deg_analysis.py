# scanpy_viewer/analysis/deg_analysis.py

import pandas as pd
import numpy as np
import scipy.sparse
import logging
from collections import OrderedDict
import anndata as ad # Import AnnData

# Conditional import for pyDESeq2 - crucial for this module
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    PYDESEQ2_INSTALLED = True
except ImportError:
    PYDESEQ2_INSTALLED = False
    # Define dummy classes if not installed to avoid NameErrors later,
    # although the main check should prevent their use.
    class DeseqDataSet: pass
    class DeseqStats: pass

from utils import AnalysisError, FactorNotFoundError, create_group_string # Use relative imports

logger = logging.getLogger(__name__)


def prepare_metadata_and_run_deseq(adata_agg, comparison_factors, replicate_factor,
                                   group1_levels, group2_levels, min_count_sum_filter=10,
                                   min_nonzero_samples=0):
    """
    Prepares metadata and runs pyDESeq2 on aggregated data.
    Operates on the provided aggregated AnnData object.

    Args:
        adata_agg (ad.AnnData): Aggregated AnnData object (output of aggregate_adata, sum aggregation).
        comparison_factors (list[str]): Factors defining the main comparison groups.
        replicate_factor (str | None): Factor defining replicates (or None).
        group1_levels (OrderedDict): Factor levels defining the numerator group.
        group2_levels (OrderedDict): Factor levels defining the denominator group.
        min_count_sum_filter (int): Minimum total count for a gene across samples.
        min_nonzero_samples (int): Minimum number of non-zero samples for a gene.

    Returns:
        pd.DataFrame: DESeq2 results DataFrame.

    Raises:
        AnalysisError: If metadata prep or DESeq2 fails.
        FactorNotFoundError: If factors are missing after aggregation.
        ValueError: If groups are invalid or data is unsuitable.
        ImportError: If pyDESeq2 is not installed (checked at the start).
        TypeError: If input is not an AnnData object.
        MemoryError: If operations run out of memory.
    """
    if not PYDESEQ2_INSTALLED:
        # This function should not be called if check fails in app.py, but double-check
        raise ImportError("pydeseq2 is not installed. Please install it to run DEG analysis.")

    # --- Input Validation ---
    if adata_agg is None:
         raise ValueError("Input AnnData object for DESeq2 is None.")
    if not isinstance(adata_agg, ad.AnnData):
        raise TypeError("Input for DESeq2 must be a valid AnnData object.")
    if not hasattr(adata_agg, 'X') or adata_agg.X is None:
        raise AnalysisError("Aggregated data for DESeq2 is missing .X matrix (expected counts).")
    if not hasattr(adata_agg, 'obs') or adata_agg.obs.empty:
        raise AnalysisError("Aggregated data for DESeq2 is missing .obs metadata.")

    logger.info(f"Starting DESeq2 preparation. Comparison Factors: {comparison_factors}, Replicate Factor: {replicate_factor}")
    logger.debug(f"Group 1 levels: {group1_levels}")
    logger.debug(f"Group 2 levels: {group2_levels}")

    # Important: Work on copies of metadata and potentially data to avoid side effects
    metadata = adata_agg.obs.copy()

    # --- Validate Factors Exist in Aggregated Metadata ---
    all_factors = comparison_factors + ([replicate_factor] if replicate_factor else [])
    missing_factors = [f for f in all_factors if f not in metadata.columns]
    if missing_factors:
        raise FactorNotFoundError(f"Factors {missing_factors} not found in aggregated metadata columns: {metadata.columns.tolist()}. Check if they were included during aggregation.")

    # --- Metadata Processing ---
    try:
        # 1. Create Combined Comparison Group Column
        logger.info(f"Creating combined comparison group column ('comparison_group') from factors: {comparison_factors}")

        # Ensure comparison factors are suitable for string joining (convert if necessary)
        for factor in comparison_factors:
            if factor not in metadata.columns: # Should be caught above, but double check
                 raise FactorNotFoundError(f"Comparison factor '{factor}' missing from metadata columns during processing.")
            if not pd.api.types.is_string_dtype(metadata[factor]) and not pd.api.types.is_categorical_dtype(metadata[factor]):
                logger.warning(f"Comparison factor '{factor}' in aggregated metadata is not string/categorical ({metadata[factor].dtype}). Converting to string.")
                metadata[factor] = metadata[factor].astype(str)
            elif pd.api.types.is_categorical_dtype(metadata[factor]):
                # Use the string representation of categories for joining
                if metadata[factor].cat.categories.dtype != 'object' and metadata[factor].cat.categories.dtype != 'string': # Check if categories themselves are non-string
                    logger.warning(f"Comparison factor '{factor}' categories have non-string dtype ({metadata[factor].cat.categories.dtype}). Converting categories to string.")
                    try:
                         metadata[factor] = metadata[factor].cat.rename_categories(metadata[factor].cat.categories.astype(str))
                    except Exception as cat_e:
                         raise AnalysisError(f"Failed to convert categories of factor '{factor}' to string: {cat_e}") from cat_e
                # Convert the series itself to string objects for reliable apply/join
                metadata[factor] = metadata[factor].astype(str)


        # Apply the helper function to create the combined group string
        try:
            metadata['comparison_group'] = metadata.apply(lambda row: create_group_string(row, comparison_factors), axis=1)
        except (KeyError, ValueError) as apply_e:
             # Catch errors from create_group_string if they occur row-wise
             logger.error(f"Error applying create_group_string: {apply_e}", exc_info=True)
             raise AnalysisError(f"Failed to create combined group strings: {apply_e}") from apply_e

        # Check for errors during creation (e.g., unexpected None results)
        if metadata['comparison_group'].isnull().any():
            # This might indicate issues in create_group_string or the input data
            raise AnalysisError("Failed to create some combined group identifiers (resulted in None/NaN). Check factor columns for unexpected values.")

        metadata['comparison_group'] = metadata['comparison_group'].astype('category')
        actual_groups = metadata['comparison_group'].cat.categories.tolist()
        logger.info(f"Created 'comparison_group' column with levels: {actual_groups}")

        # 2. Define Group Strings for Contrast
        # Ensure levels from input dicts are strings for consistent joining
        try:
            group1_str = '_'.join(str(group1_levels[factor]) for factor in comparison_factors)
            group2_str = '_'.join(str(group2_levels[factor]) for factor in comparison_factors)
        except KeyError as e:
            raise ValueError(f"Factor '{e}' used in group definition not found in comparison_factors list: {comparison_factors}") from e

        logger.info(f"Contrast definition: Numerator='{group1_str}', Denominator='{group2_str}'")

        # Validate group strings exist in the generated comparison groups
        if group1_str not in actual_groups:
            raise ValueError(f"Numerator group '{group1_str}' (derived from group 1 levels) not found in the combined 'comparison_group' column. Existing combined groups: {actual_groups}")
        if group2_str not in actual_groups:
            raise ValueError(f"Denominator group '{group2_str}' (derived from group 2 levels) not found in the combined 'comparison_group' column. Existing combined groups: {actual_groups}")
        if group1_str == group2_str:
            raise ValueError("Numerator (Group 1) and Denominator (Group 2) definitions result in the same combined group string. Comparison is invalid.")

        # 3. Construct Design Formula & Ensure Factor Types for DESeq2
        design_factors_list = []

        # Process replicate factor if provided
        if replicate_factor:
            # Ensure replicate factor is treated as categorical by DESeq2
            if not pd.api.types.is_categorical_dtype(metadata[replicate_factor]):
                logger.warning(f"Converting replicate factor '{replicate_factor}' (dtype: {metadata[replicate_factor].dtype}) to categorical for DESeq2 design.")
                try:
                    metadata[replicate_factor] = pd.Categorical(metadata[replicate_factor])
                except Exception as e:
                    raise AnalysisError(f"Could not convert replicate factor '{replicate_factor}' to categorical: {e}") from e

            # Check for sufficient replication within comparison groups - crucial for model fitting
            try:
                 # Count unique replicates within each *combined* comparison group
                 rep_check = metadata.groupby('comparison_group', observed=False)[replicate_factor].nunique()
                 groups_with_insufficient_reps = rep_check[rep_check < 2].index.tolist()
                 if groups_with_insufficient_reps:
                     # Log a strong warning, as DESeq2 might fail or produce unreliable results
                     logger.error(f"Insufficient replication (< 2 unique replicates based on factor '{replicate_factor}') found in combined comparison groups: {groups_with_insufficient_reps}. DESeq2 model may be unstable or fail.")
                     # Optionally raise an error immediately:
                     # raise ValueError(f"Insufficient replication (< 2 unique replicates for factor '{replicate_factor}') in groups: {groups_with_insufficient_reps}. Cannot run reliable DEG analysis.")
            except Exception as rep_check_e:
                 logger.error(f"Could not perform replication check: {rep_check_e}", exc_info=True)
                 # Optionally raise an error or proceed with caution

            design_factors_list.append(replicate_factor)


        # Add the primary comparison factor (already categorical)
        design_factors_list.append('comparison_group')

        # Build the formula string
        design_formula = f"~ {' + '.join(design_factors_list)}"
        logger.info(f"Using DESeq2 design formula: {design_formula}")

    except (KeyError, ValueError, FactorNotFoundError, AnalysisError) as e:
        # Catch specific errors from metadata processing
        logger.error(f"Error during metadata preparation for DESeq2: {e}", exc_info=True)
        # Re-raise as AnalysisError for consistent handling upstream
        raise AnalysisError(f"Metadata preparation failed: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during metadata prep
        logger.error(f"Unexpected error during metadata preparation: {e}", exc_info=True)
        raise AnalysisError(f"Unexpected metadata preparation error: {e}") from e


    # --- Prepare Counts Matrix ---
    try:
        logger.info("Preparing counts matrix for pyDESeq2...")
        # Use the .X from the aggregated object
        counts_matrix = adata_agg.X
        is_sparse = scipy.sparse.issparse(counts_matrix)

        # 1. Check for non-negative values
        min_val = counts_matrix.min()
        if min_val < 0:
            logger.error(f"Counts matrix contains negative values (min={min_val}). DESeq2 requires non-negative counts.")
            raise ValueError("Input data for DESeq2 contains negative values.")

        # 2. Check for non-integers and convert if necessary
        is_integer = False
        if is_sparse:
            # Check only the stored data points in sparse matrix for efficiency
            # Check if data array exists and has elements before checking mod
            if hasattr(counts_matrix, 'data') and counts_matrix.data.size > 0:
                is_integer = np.all(np.equal(np.mod(counts_matrix.data, 1), 0))
            else: # Empty sparse matrix is considered integer
                is_integer = True
        else: # Dense array
             if counts_matrix.size > 0:
                 is_integer = np.all(np.equal(np.mod(counts_matrix, 1), 0))
             else: # Empty dense matrix
                  is_integer = True


        if not is_integer:
            logger.warning("Counts matrix contains non-integer values. Truncating to integers for DESeq2.")
            counts_matrix = counts_matrix.astype(int)
        elif not np.issubdtype(counts_matrix.dtype, np.integer):
            # If it contained only integers but had float dtype, convert explicitly
            logger.info(f"Counts matrix has float dtype ({counts_matrix.dtype}) but contains only integers. Converting to integer type.")
            counts_matrix = counts_matrix.astype(int)


        # 3. Convert to DataFrame (pyDESeq2 requires DataFrame)
        if is_sparse:
            # Densify sparse matrix - required by current pyDESeq2 versions. Warn about memory.
            logger.warning("Densifying sparse counts matrix for pyDESeq2 input. This may consume significant memory for large datasets.")
            counts_df = pd.DataFrame(counts_matrix.toarray(), index=adata_agg.obs_names, columns=adata_agg.var_names)
        else:
            # If already dense, just create DataFrame
            counts_df = pd.DataFrame(counts_matrix, index=adata_agg.obs_names, columns=adata_agg.var_names)

        # 4. Ensure DataFrame dtype is integer after potential densification/conversion
        if not all(counts_df.dtypes.apply(pd.api.types.is_integer_dtype)):
            logger.warning("Counts DataFrame columns are not all integer type after conversion. Forcing conversion.")
            try:
                # Optimize conversion if possible (e.g., using downcasting)
                # Use nullable integer if NaNs might be present, though counts shouldn't have NaNs
                counts_df = counts_df.astype(np.int32) # Use a reasonably sized integer type
            except Exception as e:
                raise ValueError(f"Could not convert counts DataFrame columns to integer type: {e}") from e

        # 5. Filter genes by minimum total count across all pseudobulk samples
        if min_count_sum_filter > 0:
            logger.info(f"Filtering genes with total count < {min_count_sum_filter} across all {counts_df.shape[0]} samples.")
            try:
                genes_to_keep = counts_df.sum(axis=0) >= min_count_sum_filter
                n_before = counts_df.shape[1]
                counts_df = counts_df.loc[:, genes_to_keep] # Filter columns (genes)
                n_after = counts_df.shape[1]
                logger.info(f"Filtered {n_before - n_after} genes. {n_after} genes remaining.")
                if n_after == 0:
                    raise AnalysisError(f"No genes remaining after minimum count filtering (threshold={min_count_sum_filter}). Cannot run DESeq2.")
            except Exception as filter_e:
                logger.error(f"Error during gene count filtering: {filter_e}", exc_info=True)
                raise AnalysisError(f"Failed to filter genes by count sum: {filter_e}") from filter_e

        # 6. Filter genes by minimum number of non-zero samples
        if min_nonzero_samples > 0:
            logger.info(f"Filtering genes with non-zero counts in fewer than {min_nonzero_samples} samples.")
            try:
                # Count non-zero values for each gene across samples
                nonzero_counts = (counts_df > 0).sum(axis=0)
                genes_to_keep_nonzero = nonzero_counts >= min_nonzero_samples
                n_before = counts_df.shape[1]
                counts_df = counts_df.loc[:, genes_to_keep_nonzero]  # Filter columns (genes)
                n_after = counts_df.shape[1]
                logger.info(f"Filtered {n_before - n_after} genes by non-zero count. {n_after} genes remaining.")
                if n_after == 0:
                    raise AnalysisError(f"No genes remaining after non-zero sample filtering (threshold={min_nonzero_samples}). Cannot run DESeq2.")
            except Exception as filter_e:
                logger.error(f"Error during gene non-zero filtering: {filter_e}", exc_info=True)
                raise AnalysisError(f"Failed to filter genes by non-zero samples: {filter_e}") from filter_e


        if counts_df.empty:
            raise AnalysisError("Counts DataFrame is empty after processing and filtering. Cannot run DESeq2.")
        if counts_df.shape[1] < 1: # Should be caught by n_after check, but ensure
             raise AnalysisError("No variables remaining in counts matrix for DESeq2.")


    except (ValueError, AnalysisError) as e:
        logger.error(f"Error preparing counts matrix: {e}", exc_info=True)
        raise # Re-raise specific error
    except MemoryError as me:
        logger.error(f"MemoryError during counts matrix preparation (potentially densification): {me}", exc_info=True)
        raise AnalysisError("Not enough memory to prepare counts matrix for DESeq2.") from me
    except Exception as e:
        logger.error(f"Unexpected error preparing counts matrix: {e}", exc_info=True)
        raise AnalysisError(f"Counts matrix preparation failed unexpectedly: {e}") from e


    # --- Run pyDESeq2 ---
    try:
        logger.info(f"Initializing DeseqDataSet with {counts_df.shape[0]} samples and {counts_df.shape[1]} genes.")

        # Ensure metadata index matches counts index perfectly before creating DDS object
        if not metadata.index.equals(counts_df.index):
            logger.warning("Metadata index does not match counts index before DESeqDataSet init. Reindexing metadata.")
            try:
                # Align metadata to the potentially filtered counts_df index
                metadata = metadata.reindex(counts_df.index)
                if metadata.isnull().any().any(): # Check if reindexing introduced NaNs in crucial columns
                    nan_cols = metadata.columns[metadata.isnull().any()].tolist()
                    raise ValueError(f"Reindexing metadata to match counts introduced NaN values in required columns: {nan_cols}. Check sample consistency between counts and metadata.")
            except Exception as e:
                raise AnalysisError(f"Failed to align metadata index with counts index: {e}") from e

        # Check final dimensions before running
        if metadata.shape[0] < 2 or counts_df.shape[0] < 2:
            raise AnalysisError(f"Cannot run DESeq2 with fewer than 2 samples (found {metadata.shape[0]}).")
        # Check if design matrix will be full rank (basic check: enough variation?)
        for factor in design_factors_list:
            if metadata[factor].nunique() < 2:
                 logger.warning(f"Design factor '{factor}' has only {metadata[factor].nunique()} level(s). Model may not be full rank.")


        # Create DESeqDataSet object
        # design_factors must include all terms in the formula
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design_factors=design_factors_list, # List all factors used in the formula
            refit_cooks=True, # Default and generally recommended for outlier handling
            quiet=False # Set to False initially for more debugging info from pyDESeq2
        )
        logger.info(f"Running DESeq2 analysis (dds.deseq2()) using formula: {design_formula}...")
        # The design formula is implicitly used by deseq2() based on the factors in metadata
        dds.deseq2()
        logger.info("DESeq2 dispersion estimation and model fitting complete.")

        # Define the contrast using the combined group name
        contrast_list = ["comparison_group", group1_str, group2_str]
        logger.info(f"Running statistical test (DeseqStats) for contrast: {contrast_list}")
        stat_res = DeseqStats(
            dds,
            contrast=contrast_list,
            quiet=False # Get more output during stats calculation
        )

        logger.info("Summarizing results (stat_res.summary())...")
        stat_res.summary() # Runs the Wald test and computes p-values, LFC etc.
        results_df = stat_res.results_df # Get the results DataFrame

        if results_df is None or results_df.empty:
            logger.warning("DESeq2 analysis ran but produced an empty results DataFrame.")
            # Return an empty DataFrame or raise error? Return empty for now.
            results_df = pd.DataFrame() # Ensure it's an empty DF, not None
        else:
            logger.info(f"DESeq2 finished. Results table shape: {results_df.shape}")
            # Optional: Add gene symbols if available in the original aggregated var annotations
            if 'gene_symbols' in adata_agg.var.columns:
                logger.debug("Merging gene symbols into results.")
                try:
                    # Ensure index names match if necessary (usually they are gene IDs)
                    if results_df.index.name != adata_agg.var.index.name:
                        logger.debug(f"Aligning index names for merge ('{results_df.index.name}' vs '{adata_agg.var.index.name}')")
                        results_df.index.name = adata_agg.var.index.name
                    # Merge gene symbols based on index (gene ID)
                    results_df = results_df.merge(adata_agg.var[['gene_symbols']], left_index=True, right_index=True, how='left')
                except Exception as e:
                    logger.warning(f"Could not merge gene symbols into results: {e}")

            # Sort by adjusted p-value for convenience
            results_df = results_df.sort_values(by="padj")

        # Clean up DESeq2 objects? Optional, might help release memory sooner.
        del dds
        del stat_res

        return results_df

    except ImportError as ie:
        # Should be caught earlier, but handle here too
        logger.error(f"ImportError during DESeq2 run: {ie}. Is pyDESeq2 correctly installed?", exc_info=True)
        raise AnalysisError("pyDESeq2 not found or import failed. Please ensure it is installed.") from ie
    except ValueError as ve: # Catch potential DESeq2 internal ValueErrors (e.g., singular model, contrast errors)
        logger.error(f"ValueError during DESeq2 run: {ve}", exc_info=True)
        raise AnalysisError(f"DESeq2 failed: {ve}. This might be due to issues with the model (e.g., singular design matrix, insufficient data/replicates), contrast definition, or data values.") from ve
    except MemoryError as me:
        logger.error(f"MemoryError during DESeq2 execution: {me}", exc_info=True)
        raise AnalysisError("Not enough memory to run DESeq2 analysis.") from me
    except Exception as e:
        # Catch any other unexpected errors from pyDESeq2
        logger.error(f"Error during pyDESeq2 execution: {e}", exc_info=True)
        raise AnalysisError(f"pyDESeq2 execution failed unexpectedly: {e}") from e
