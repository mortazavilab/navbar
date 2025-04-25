# scanpy_viewer/analysis/pca_analysis.py

import scanpy as sc
import anndata as ad
import numpy as np
import scipy.sparse
import logging
from utils import AnalysisError 

logger = logging.getLogger(__name__)

def preprocess_and_run_pca(adata_agg, n_hvgs, n_comps=50):
    """
    Performs preprocessing (log, HVG, scale) and PCA on aggregated data.
    Operates on a *copy* of the input AnnData to avoid modifying cached objects.

    Args:
        adata_agg (ad.AnnData): Aggregated AnnData object (output of aggregate_adata).
                                Should ideally contain sum-aggregated data (e.g., counts or normalized).
        n_hvgs (int): Number of highly variable genes to select.
        n_comps (int): Max number of principal components to compute.

    Returns:
        ad.AnnData: AnnData object (a copy) with PCA results in .obsm['X_pca'] and .uns['pca'].

    Raises:
        AnalysisError: If preprocessing or PCA fails.
        ValueError: If input is invalid or data is unsuitable.
        TypeError: If input is not an AnnData object.
    """
    if adata_agg is None:
        raise ValueError("Input AnnData object for PCA is None.")
    if not isinstance(adata_agg, ad.AnnData):
        raise TypeError("Input for PCA preprocessing must be a valid AnnData object.")
    if not hasattr(adata_agg, 'X') or adata_agg.X is None:
        raise AnalysisError("Aggregated data for PCA is missing .X matrix.")

    logger.info(f"Starting PCA preprocessing and calculation on aggregated data ({adata_agg.shape}). HVGs={n_hvgs}, Max Comps={n_comps}")
    # Important: Work on a copy to avoid modifying the potentially cached aggregated object
    adata_pca = adata_agg.copy()

    try:
        # --- Basic Checks ---
        if adata_pca.n_obs < 2:
            raise ValueError(f"Only {adata_pca.n_obs} samples after aggregation. Cannot run PCA (need at least 2).")
        if adata_pca.n_vars < 1:
            raise ValueError("Zero variables found in aggregated data before PCA.")

        # --- Ensure Float Data ---
        # Check if data is already float, complex, or boolean (can be cast)
        needs_conversion = True
        if hasattr(adata_pca.X, 'dtype'): # Check if X has dtype attribute
             dtype = adata_pca.X.dtype
             if np.issubdtype(dtype, np.floating) or \
                np.issubdtype(dtype, complex) or \
                np.issubdtype(dtype, np.bool_):
                 needs_conversion = False
        else: # If X has no dtype (e.g., None or unusual type), attempt conversion
             logger.warning("adata_pca.X lacks dtype attribute. Attempting conversion to float32.")

        if needs_conversion:
            logger.warning(f"PCA input data type ({getattr(adata_pca.X, 'dtype', 'unknown')}) requires conversion to float. Converting to float32.")
            try:
                # Use .astype() which works for both sparse and dense
                if adata_pca.X is None: raise ValueError("Cannot convert None .X matrix to float.")
                adata_pca.X = adata_pca.X.astype(np.float32)
            except Exception as conv_e:
                raise AnalysisError(f"Could not convert aggregated data to float for PCA: {conv_e}") from conv_e

        # --- Preprocessing ---
        logger.info("Preprocessing pseudobulk data (log1p)...")
        # Check for negative values before log1p
        try:
             min_val = adata_pca.X.min()
             if min_val < 0:
                 logger.warning(f"Data contains negative values (min={min_val}) before log1p. This might lead to NaNs.")
                 # Optionally raise an error or apply a shift if appropriate for the data type
                 # Example shift: adata_pca.X = adata_pca.X - min_val + 0.001 # Use with caution! Only if data context allows.
        except Exception as min_check_e:
             logger.warning(f"Could not check for negative values before log1p: {min_check_e}")


        sc.pp.log1p(adata_pca)
        # Check for NaNs/Infs after log1p
        try:
            x_data_for_check = adata_pca.X.data if scipy.sparse.issparse(adata_pca.X) else adata_pca.X
            if x_data_for_check.size > 0 and not np.all(np.isfinite(x_data_for_check)):
                n_nonfinite = np.sum(~np.isfinite(x_data_for_check))
                logger.error(f"Data contains {n_nonfinite} non-finite values (NaN/inf) after log1p. Cannot proceed with PCA.")
                raise AnalysisError("Non-finite values detected after log1p transformation. Check for negative input values or other issues.")
        except AttributeError: # Handle case where .data attribute might be missing (e.g. dense matrix)
             if not np.all(np.isfinite(adata_pca.X)):
                n_nonfinite = np.sum(~np.isfinite(adata_pca.X))
                logger.error(f"Data contains {n_nonfinite} non-finite values (NaN/inf) after log1p. Cannot proceed with PCA.")
                raise AnalysisError("Non-finite values detected after log1p transformation. Check for negative input values or other issues.")


        # HVG selection
        n_hvgs_actual = min(n_hvgs, adata_pca.n_vars) # Cannot select more HVGs than exist
        if n_hvgs_actual < 1:
            logger.warning("Requested number of HVGs is less than 1. Skipping HVG selection.")
        elif adata_pca.n_vars < 2:
            logger.info("Skipping HVG selection (fewer than 2 genes).")
        else:
            # Ensure n_top_genes is not greater than n_vars-1 for some methods if needed
            n_top_genes_safe = min(n_hvgs_actual, adata_pca.n_vars -1 if adata_pca.n_vars > 1 else 1)
            if n_top_genes_safe < 1:
                logger.warning(f"Cannot select HVGs as adjusted n_top_genes ({n_top_genes_safe}) < 1. Skipping.")
            else:
                logger.info(f"Selecting top {n_top_genes_safe} highly variable genes (flavor='seurat_v3')...")
                try:
                    sc.pp.highly_variable_genes(adata_pca, n_top_genes=n_top_genes_safe, flavor='seurat_v3')

                    # Check if 'highly_variable' column was actually added
                    if 'highly_variable' not in adata_pca.var.columns:
                         logger.error("HVG selection failed: 'highly_variable' column not found in adata.var after running sc.pp.highly_variable_genes.")
                         raise AnalysisError("HVG selection step did not produce the expected 'highly_variable' column.")

                    if not adata_pca.var['highly_variable'].any():
                        logger.warning("No highly variable genes were identified using seurat_v3 flavor. Proceeding with all genes.")
                    else:
                        n_found = adata_pca.var['highly_variable'].sum()
                        logger.info(f"Found {n_found} HVGs.")
                        # Subset the AnnData object to keep only HVGs
                        adata_pca = adata_pca[:, adata_pca.var['highly_variable']].copy() # Ensure copy after subsetting
                        if adata_pca.n_vars == 0:
                            raise AnalysisError("Zero genes remaining after HVG filtering.")
                        logger.info(f"Data shape after HVG filter: {adata_pca.shape}")
                except Exception as hvg_e:
                    logger.error(f"Error during HVG selection: {hvg_e}", exc_info=True)
                    raise AnalysisError(f"Failed to select highly variable genes: {hvg_e}") from hvg_e


        # Scaling
        logger.info("Scaling data (max_value=10)...")
        sc.pp.scale(adata_pca, max_value=10)
        # Check for NaNs/Infs after scaling
        try:
            x_data_for_check = adata_pca.X.data if scipy.sparse.issparse(adata_pca.X) else adata_pca.X
            if x_data_for_check.size > 0 and not np.all(np.isfinite(x_data_for_check)):
                 n_nonfinite = np.sum(~np.isfinite(x_data_for_check))
                 logger.error(f"Data contains {n_nonfinite} non-finite values (NaN/inf) after scaling. Cannot proceed with PCA.")
                 raise AnalysisError("Non-finite values detected after scaling. Check for constant genes or issues in previous steps.")
        except AttributeError:
             if not np.all(np.isfinite(adata_pca.X)):
                n_nonfinite = np.sum(~np.isfinite(adata_pca.X))
                logger.error(f"Data contains {n_nonfinite} non-finite values (NaN/inf) after scaling. Cannot proceed with PCA.")
                raise AnalysisError("Non-finite values detected after scaling. Check for constant genes or issues in previous steps.")


        # --- Run PCA ---
        # Determine number of components dynamically and safely
        # Max components is min(n_obs, n_vars) - 1
        if adata_pca.n_obs <= 1 or adata_pca.n_vars <= 1:
            raise ValueError(f"Cannot run PCA with current data shape after preprocessing: {adata_pca.shape}. Need >1 observation and >1 variable.")

        max_possible_comps = min(adata_pca.n_obs -1 , adata_pca.n_vars -1)
        # Ensure n_comps doesn't exceed max possible, and is at least 1
        n_comps_actual = max(1, min(n_comps, max_possible_comps))

        if n_comps_actual < 1:
            # This case should be caught by the shape check above, but double-check
            raise ValueError(f"Cannot compute principal components. Need at least 1, but max possible components is {max_possible_comps} (based on shape {adata_pca.shape} after preprocessing).")

        logger.info(f"Running PCA with n_comps={n_comps_actual} using 'arpack' solver...")
        try:
            # Use arpack for stability, especially when n_comps is much smaller than data dimensions
            sc.tl.pca(adata_pca, n_comps=n_comps_actual, svd_solver='arpack', random_state=0) # Add random_state
        except Exception as pca_e:
            logger.error(f"Error during sc.tl.pca execution: {pca_e}", exc_info=True)
            # Try to provide more specific error messages if possible
            if "Input contains NaN" in str(pca_e):
                raise AnalysisError("PCA calculation failed due to NaN values in the data after preprocessing. Check logs.") from pca_e
            raise AnalysisError(f"PCA calculation failed: {pca_e}") from pca_e

        # Check if PCA results were stored correctly
        if 'X_pca' not in adata_pca.obsm or 'pca' not in adata_pca.uns:
             logger.error("PCA calculation ran but results are missing in .obsm['X_pca'] or .uns['pca'].")
             raise AnalysisError("PCA calculation failed to store results in the AnnData object.")

        logger.info("Pseudobulk PCA calculation complete.")
        return adata_pca # Return the processed copy

    except (ValueError, TypeError, AnalysisError) as e:
        # Catch specific errors raised within the function
        logger.error(f"Error during PCA pipeline: {e}", exc_info=False) # Log less detail for known errors
        raise # Re-raise the specific error
    except MemoryError as me:
         logger.error(f"MemoryError during PCA pipeline: {me}", exc_info=True)
         raise AnalysisError("Not enough memory for PCA preprocessing or calculation.") from me
    except Exception as pipeline_e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during PCA pipeline: {pipeline_e}", exc_info=True)
        raise AnalysisError(f"Unexpected error in PCA pipeline: {pipeline_e}") from pipeline_e
