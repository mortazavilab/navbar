# navbar/aggregation.py

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse
import logging
import hashlib
import streamlit as st
from utils import AggregationError, FactorNotFoundError, copy_index_to_obs

logger = logging.getLogger(__name__)

def check_aggregation_source(adata_obj, selected_layer_key, required_func='sum'):
    """
    Determines the AnnData object and layer argument for aggregation.

    Args:
        adata_obj (ad.AnnData): The AnnData object to check.
        selected_layer_key (str): User's layer selection ('adata.X', 'adata.raw.X', layer name, 'Auto-Select').
        required_func (str): The aggregation function planned ('sum', 'mean', etc.) - used for 'Auto-Select' hint.

    Returns:
        tuple: (adata_to_agg, agg_layer_arg, source_log_msg)
               adata_to_agg: AnnData object (or adata.raw) to use for data access.
               agg_layer_arg: Layer argument for sc.get.aggregate (None or str).
               source_log_msg: A descriptive string about the chosen source.

    Raises:
        AggregationError: If a suitable source cannot be found or the selection is invalid.
    """
    adata_to_agg = None
    agg_layer_arg = None
    source_base_log_msg = ""

    if selected_layer_key == "adata.raw.X":
        if adata_obj.raw is not None:
            adata_to_agg = adata_obj.raw
            # Need to check if .X exists in raw
            if not hasattr(adata_to_agg, 'X') or adata_to_agg.X is None:
                 raise AggregationError("Selected 'adata.raw.X', but adata.raw.X is missing or None.")
            agg_layer_arg = None
            source_base_log_msg = "Using adata.raw.X"
        else:
            raise AggregationError("Selected 'adata.raw.X', but adata.raw does not exist.")

    elif selected_layer_key == "adata.X":
        if hasattr(adata_obj, 'X') and adata_obj.X is not None:
            adata_to_agg = adata_obj
            agg_layer_arg = None
            source_base_log_msg = "Using adata.X"
        else:
            # Distinguish between .X attribute missing and .X being None
            if not hasattr(adata_obj, 'X'):
                raise AggregationError("Selected 'adata.X', but the .X attribute is missing from the AnnData object.")
            else:
                raise AggregationError("Selected 'adata.X', but adata.X is None.")

    elif selected_layer_key == "Auto-Select":
        found_source = False
        # Prioritize 'counts' layer for 'sum' aggregation if available and shape matches
        if required_func == 'sum' and 'counts' in getattr(adata_obj, 'layers', {}):
            if adata_obj.layers['counts'].shape == adata_obj.shape:
                adata_to_agg = adata_obj
                agg_layer_arg = 'counts'
                source_base_log_msg = "Auto-Select: Using layer 'counts'"
                found_source = True
            else:
                 logger.warning("Auto-Select found 'counts' layer, but shape mismatch. Skipping.")

        # Fallback to .X if not found or unsuitable
        if not found_source and hasattr(adata_obj, 'X') and adata_obj.X is not None:
            adata_to_agg = adata_obj
            agg_layer_arg = None
            source_base_log_msg = "Auto-Select: Using adata.X"
            found_source = True

        # Fallback to .raw.X if still not found
        if not found_source and adata_obj.raw is not None:
             if hasattr(adata_obj.raw, 'X') and adata_obj.raw.X is not None:
                adata_to_agg = adata_obj.raw
                agg_layer_arg = None
                source_base_log_msg = "Auto-Select: Using adata.raw.X"
                found_source = True
             else:
                  logger.warning("Auto-Select found adata.raw, but adata.raw.X is missing/None. Skipping.")


        if not found_source:
            raise AggregationError("Auto-Select failed: Could not find suitable data in 'counts' layer, adata.X, or adata.raw.X.")

    else: # User selected a specific layer name
        if selected_layer_key in getattr(adata_obj, 'layers', {}):
            # Check shape consistency
            if adata_obj.layers[selected_layer_key].shape != adata_obj.shape:
                raise AggregationError(f"Selected layer '{selected_layer_key}' shape {adata_obj.layers[selected_layer_key].shape} "
                                       f"does not match AnnData shape {adata_obj.shape}.")
            adata_to_agg = adata_obj
            agg_layer_arg = selected_layer_key
            source_base_log_msg = f"Using layer '{selected_layer_key}'"
        else:
            raise AggregationError(f"Selected layer '{selected_layer_key}' not found in adata.layers.")

    # Final check on the selected source object and layer argument
    if adata_to_agg is None:
         # This case should ideally be caught by the logic above
         raise AggregationError("Internal Error: Failed to assign a source AnnData object (adata_to_agg is None).")

    # Ensure the chosen data source actually exists
    if agg_layer_arg:
         if agg_layer_arg not in getattr(adata_to_agg, 'layers', {}):
              raise AggregationError(f"Internal Error: Selected layer '{agg_layer_arg}' not found in the final source object '{source_base_log_msg}'.")
         if adata_to_agg.layers[agg_layer_arg] is None:
              raise AggregationError(f"Internal Error: Selected layer '{agg_layer_arg}' is None in the final source object.")
    else: # Relying on .X
        if not hasattr(adata_to_agg, 'X') or adata_to_agg.X is None:
            raise AggregationError(f"Internal Error: The final source object '{source_base_log_msg}' is missing its .X attribute or .X is None.")


    source_log_msg = f"{source_base_log_msg} (for aggregation func: {required_func})."
    logger.info(f"Aggregation source determined: {source_log_msg}")
    return adata_to_agg, agg_layer_arg, source_log_msg


def aggregate_adata(adata_ref, grouping_vars, selected_layer_key, agg_func='sum'):
    """
    Performs pseudobulking aggregation on an AnnData object.

    Args:
        adata_ref (ad.AnnData): The AnnData object containing the obs columns for grouping
                                 (e.g., could be adata_vis or adata_full depending on context).
        grouping_vars (list[str]): List of column names in adata_ref.obs to group by.
        selected_layer_key (str): Layer selection ('adata.X', 'adata.raw.X', layer name, 'Auto-Select').
        agg_func (str): Aggregation function ('sum', 'mean', etc.).

    Returns:
        ad.AnnData: The aggregated AnnData object (a copy) with index copied to obs.

    Raises:
        AggregationError: If aggregation fails.
        FactorNotFoundError: If grouping variables are missing in adata_ref.obs.
        TypeError: If adata_ref is not an AnnData object.
        MemoryError: If aggregation runs out of memory.
    """
    logger.info(f"Starting aggregation by {grouping_vars} using func='{agg_func}', layer_key='{selected_layer_key}'.")
    grouping_vars = list(grouping_vars)
    # --- Input Validation ---
    if not isinstance(adata_ref, ad.AnnData):
        raise TypeError("Input 'adata_ref' must be an AnnData object.")
    if not grouping_vars:
        raise AggregationError("No grouping variables provided for aggregation.")

    missing_vars = [var for var in grouping_vars if var not in adata_ref.obs]
    if missing_vars:
        raise FactorNotFoundError(f"Grouping variables {missing_vars} not found in reference adata.obs columns: {adata_ref.obs.columns.tolist()}")

    # Warn about NaNs in grouping columns
    nan_info = {}
    for var in grouping_vars:
        try:
            nan_count = adata_ref.obs[var].isnull().sum()
            if nan_count > 0:
                nan_info[var] = nan_count
        except KeyError:
             # Should be caught above, but belt-and-suspenders
             raise FactorNotFoundError(f"Grouping variable '{var}' disappeared unexpectedly.")
    if nan_info:
         nan_details = ", ".join([f"'{k}': {v} NaNs" for k,v in nan_info.items()])
         logger.warning(f"Grouping variable(s) contain NaN values ({nan_details}). Corresponding cells ({sum(nan_info.values())} total rows potentially affected) will be dropped by scanpy.get.aggregate.")


    # --- Determine Data Source ---
    try:
        # Pass the reference AnnData to check for layers etc.
        adata_source, agg_layer_arg, source_log_msg = check_aggregation_source(
            adata_ref, selected_layer_key, required_func=agg_func
        )
        # adata_source is the object from which data (.X or .layers) will be read
    except AggregationError as e:
        logger.error(f"Failed to determine aggregation source: {e}")
        raise # Re-raise the error

    # --- Prepare Data for Aggregation ---
    # Create a temporary AnnData with only necessary parts to avoid unexpected behavior
    # Use the *reference* adata's obs for grouping, but the *source* adata's data (X/layer)

    # 1. Get the observation data for grouping from the reference object
    try:
        logger.info(f"fetching {grouping_vars} from reference adata.obs for aggregation.")
        logger.info(f"adata_ref.obs columns: {adata_ref.obs.columns.tolist()}")
        temp_obs_for_agg = adata_ref.obs[grouping_vars].copy()
    except KeyError:
        # This should be caught by the initial check, but double-check
        raise FactorNotFoundError("Grouping variables vanished from reference adata.obs.")

    # 2. Convert grouping columns to string/categorical for reliable groupby
    for col in grouping_vars:
        if pd.api.types.is_numeric_dtype(temp_obs_for_agg[col]):
            # Decide how to handle numeric grouping vars - convert to string or categorical?
            # Converting to string is generally safer for direct groupby keys.
            logger.warning(f"Grouping variable '{col}' is numeric. Converting to string for aggregation.")
            temp_obs_for_agg[col] = temp_obs_for_agg[col].astype(str)
        elif not pd.api.types.is_categorical_dtype(temp_obs_for_agg[col]):
            # Convert object/other types to categorical
            logger.debug(f"Converting grouping var '{col}' to categorical for aggregation.")
            try:
                temp_obs_for_agg[col] = pd.Categorical(temp_obs_for_agg[col])
            except TypeError as e:
                 raise AggregationError(f"Could not convert grouping column '{col}' to categorical (may contain mixed types): {e}") from e


    # 3. Ensure index alignment between temp_obs (from adata_ref) and adata_source data
    # This is crucial if adata_source is adata.raw, which might have a different index
    if not adata_ref.obs_names.equals(adata_source.obs_names):
        logger.warning(f"Aligning obs for aggregation as reference index ({adata_ref.n_obs} cells) and data source index ({adata_source.n_obs} cells) differ (likely using raw or a subset).")
        try:
            # Align the temporary obs DataFrame to the index of the data source
            common_index = adata_ref.obs_names.intersection(adata_source.obs_names)
            if len(common_index) == 0:
                 raise AggregationError("No common observation names between reference AnnData and aggregation data source.")
            if len(common_index) < adata_ref.n_obs:
                 logger.warning(f"Reference AnnData has {adata_ref.n_obs - len(common_index)} cells not present in the data source.")

            # Ensure temp_obs_for_agg only contains common indices before proceeding
            temp_obs_for_agg = temp_obs_for_agg.loc[common_index]

            # Subset the data source to match the common index ONLY if it's not the same object as reference
            # Avoid modifying the original raw object if adata_source points to it
            if adata_source is not adata_ref:
                 adata_source_subset = adata_source[common_index, :].copy() # Make copy to avoid modifying original raw
                 logger.info(f"Aligned data source to {len(common_index)} common cells by subsetting.")
            else:
                 adata_source_subset = adata_source[common_index, :] # Subset without copy if it's the same object (already a subset?)
                 logger.info(f"Aligned data source to {len(common_index)} common cells by subsetting (reference object).")

        except KeyError as e:
            raise AggregationError(f"Could not align observation indices between reference AnnData and aggregation source. Error: {e}") from e
        except Exception as e:
            raise AggregationError(f"Unexpected error during index alignment for aggregation: {e}") from e
    else:
         # Indices match, use the full source object
         adata_source_subset = adata_source


    # 4. Create the minimal AnnData object for sc.get.aggregate
    # It needs .obs for 'by' and the data (.X or .layers) from the (potentially subsetted) source
    # Ensure var index also matches if subsetting occurred
    temp_adata_for_agg = ad.AnnData(obs=temp_obs_for_agg, var=adata_source_subset.var)

    # Assign data source carefully to the temporary object
    if agg_layer_arg:
        if agg_layer_arg not in getattr(adata_source_subset, 'layers', {}):
            raise AggregationError(f"Internal Error: Layer '{agg_layer_arg}' vanished from source data after potential subsetting.")
        temp_adata_for_agg.layers[agg_layer_arg] = adata_source_subset.layers[agg_layer_arg]
        logger.debug(f"Using layer '{agg_layer_arg}' from source object for aggregation.")
    else:
        if not hasattr(adata_source_subset, 'X') or adata_source_subset.X is None:
            raise AggregationError("Internal Error: .X is missing from source data after potential subsetting.")
        temp_adata_for_agg.X = adata_source_subset.X
        logger.debug("Using .X from source object for aggregation.")


    # --- Perform Aggregation ---
    adata_agg = None
    logger.info(f"Calling sc.get.aggregate on data with shape {temp_adata_for_agg.shape} by={grouping_vars}, func='{agg_func}', layer='{agg_layer_arg}'")
    try:
        # Ensure 'observed=True' for categorical groupby to avoid issues with unused categories
        # Scanpy's aggregate might handle this, but being explicit via pandas before could be safer if needed.
        # However, we converted to categorical earlier.
        adata_agg = sc.get.aggregate(temp_adata_for_agg, by=grouping_vars, func=agg_func, layer=agg_layer_arg)
        logger.info(f"Aggregation successful. Result shape: {adata_agg.shape}")

        # --- Post-Aggregation Processing ---
        if adata_agg is None:
             raise AggregationError("Aggregation returned None.")

        # Ensure .X exists, promoting layer if necessary
        if not hasattr(adata_agg, 'X') or adata_agg.X is None:
            if agg_func in getattr(adata_agg, 'layers', {}):
                 logger.debug(f"Aggregation result missing .X, promoting layer '{agg_func}' to .X.")
                 adata_agg.X = adata_agg.layers[agg_func].copy()
            else:
                 raise AggregationError(f"Aggregation result missing .X matrix and expected layer '{agg_func}'.")


        if not isinstance(adata_agg.X, (np.ndarray, scipy.sparse.spmatrix)):
            raise AggregationError(f"Aggregated data (.X) is not a valid matrix type: {type(adata_agg.X)}")


        # Re-attach full original var info if lost/subsetted during aggregation
        # Use adata_ref.var as the reference for full var info
        if not adata_agg.var.index.equals(adata_ref.var.index):
            logger.warning("Aggregated var index differs from reference. Attempting to restore full var info.")
            try:
                # Ensure we only take var info for genes present in the aggregated result
                common_vars = adata_agg.var_names.intersection(adata_ref.var_names)
                if len(common_vars) < adata_agg.n_vars:
                    logger.warning(f"Some variables ({adata_agg.n_vars - len(common_vars)}) in aggregated data were not found in reference var index.")

                if len(common_vars) > 0:
                     # Subset aggregated data first if necessary to match common vars
                     if len(common_vars) < adata_agg.n_vars:
                          adata_agg = adata_agg[:, common_vars].copy()
                     # Assign var info from reference, ensuring index matches
                     adata_agg.var = adata_ref.var.loc[common_vars].copy()
                     logger.info(f"Restored var annotation from reference data for {len(common_vars)} variables.")
                else:
                     # This case means aggregation produced genes not in original .var - should not happen
                     logger.error("No common variables found between aggregated data and reference var index. Cannot restore var info.")
                     # Raise error? Or continue with minimal var info?
            except Exception as var_e:
                 logger.error(f"Could not re-attach full var info after aggregation: {var_e}", exc_info=True)


        # Copy index to obs for metadata linkage (crucial)
        try:
            # This function now returns a copy
            adata_agg = copy_index_to_obs(adata_agg)
            logger.debug("Copied aggregation index/multi-index to .obs columns.")
        except ValueError as e:
            logger.error(f"Failed to copy index to obs after aggregation: {e}")
            raise AggregationError(f"Failed post-aggregation index processing: {e}") from e

        # --- Aggregation Result Summary ---
        summary_lines = []

        summary_lines.append("\n--- Aggregation Summary ---")
        summary_lines.append(f"Grouping variables used: {grouping_vars}")
        summary_lines.append(f"Number of groups (rows) in output AnnData: {adata_agg.n_obs}")
        summary_lines.append(f"Number of features (columns): {adata_agg.n_vars}")

        for var in grouping_vars:
            if var in adata_agg.obs.columns:
                levels = adata_agg.obs[var].unique()
                summary_lines.append(f"- '{var}': {len(levels)} unique levels: {levels[:10]}{' ...' if len(levels)>10 else ''}")
                value_counts = adata_agg.obs[var].value_counts()
                summary_lines.append(f"  Top level counts: {value_counts.head(5).to_dict()}")
            else:
                summary_lines.append(f"- [WARNING] Grouping variable '{var}' not in output .obs")

        # Check for missing/NaN values in output .obs
        n_nans = adata_agg.obs[grouping_vars].isnull().sum().sum()
        if n_nans > 0:
            summary_lines.append(f"[WARNING] There are {n_nans} missing (NaN) values in grouping columns after aggregation.")

        # Check for duplicated obs_names (should not happen)
        if adata_agg.obs.index.duplicated().any():
            summary_lines.append("[WARNING] There are duplicated group indices in output .obs.index!")

        # Show the first 3 groups as a quick preview
        summary_lines.append("First 3 aggregated groups (index and grouping vars):")
        summary_lines.append(f"\n{adata_agg.obs[grouping_vars].head(3)}\n")

        summary_msg = "\n".join(summary_lines)
        logger.info(summary_msg)

        # Optional: if running in Streamlit, show in app as well
        try:
            st.markdown("#### Aggregation Summary (Debug)")
            for l in summary_lines:
                st.write(l)
        except Exception:
            pass  # If not in Streamlit, just skip

        logger.info("Aggregation and index processing complete.")
        return adata_agg # Return the processed aggregated AnnData object
        
    except ValueError as ve:
        # Check for common pandas/numpy aggregation errors
        err_str = str(ve).lower()
        if "cannot calculate the" in err_str or "no numeric data to aggregate" in err_str or "cannot perform" in err_str or "buffer source array is read-only" in err_str or "check that the value variance" in err_str:
            logger.error(f"Aggregation failed likely due to empty groups, non-numeric source data, data type/variance issues: {ve}", exc_info=True)
            # Log group sizes from original data to help diagnose
            try:
                group_counts_orig = adata_ref.obs.groupby(grouping_vars, observed=True).size()
                logger.info(f"Cell counts per group combination in original data:\n{group_counts_orig.to_string()}")
                if (group_counts_orig == 0).any():
                    logger.warning("Found original group combinations with zero cells.")
            except Exception as e_count:
                logger.warning(f"Could not calculate original group sizes for debugging: {e_count}")
            raise AggregationError(f"Aggregation failed: Check logs. Potential causes: empty groups, non-numeric data, low variance. Error: {ve}") from ve
        else:
            logger.error(f"Aggregation failed with ValueError: {ve}", exc_info=True)
            raise AggregationError(f"Aggregation failed: {ve}") from ve
    except MemoryError as me:
         logger.error(f"MemoryError during aggregation: {me}", exc_info=True)
         raise AggregationError("Not enough memory to perform aggregation.") from me
    except Exception as e:
        logger.error(f"Aggregation failed with unexpected error: {e}", exc_info=True)
        raise AggregationError(f"Aggregation failed: {e}") from e

@st.cache_data(show_spinner=False)
def cached_aggregate_adata(
    _adata_ref: ad.AnnData,
    _adata_ref_hash: str,
    grouping_vars_tuple: tuple,
    selected_layer_key: str,
    agg_func: str = "sum"
):
    """
    Cached wrapper for aggregate_adata to avoid redundant computations.

    Parameters:
        _adata_ref_hash (str): Hash of the reference AnnData object to ensure cache invalidation when data changes.
        grouping_vars_tuple (tuple): Tuple of grouping variables for aggregation.
        selected_layer_key (str): Key for the data layer to aggregate (e.g., 'X', 'counts').
        agg_func (str): Aggregation function to use ('sum' or 'mean').

    Returns:
        AnnData: Aggregated AnnData object.
    """
    # Create a unique cache key based on input parameters
    cache_key = hashlib.sha256(
        f"{_adata_ref_hash}-{grouping_vars_tuple}-{selected_layer_key}-{agg_func}".encode()
    ).hexdigest()

    # Perform aggregation using the provided function
    aggregated_adata = aggregate_adata(
        adata_ref=_adata_ref,  # Replace with the actual AnnData object
        grouping_vars=grouping_vars_tuple,
        selected_layer_key=selected_layer_key,
        agg_func=agg_func
    )

    return aggregated_adata

