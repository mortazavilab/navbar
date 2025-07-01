# navbar/data_loader.py

import scanpy as sc
import anndata as ad
import io
import logging
import os
import pandas as pd
import numpy as np
import scipy.sparse # For placeholder check
from utils import DataLoaderError, AnalysisError

logger = logging.getLogger(__name__)

def load_h5ad(file_input):
    """
    Loads AnnData object from a file path (str) or uploaded file bytes/BytesIO.
    Performs basic validation and uniqueness checks.

    Args:
        file_input (str | bytes | io.BytesIO): Path to the file or file-like object.

    Returns:
        ad.AnnData: The loaded AnnData object.

    Raises:
        DataLoaderError: If loading fails or input is invalid.
        FileNotFoundError: If the file path does not exist.
        MemoryError: If the file is too large for available RAM.
    """
    adata = None
    source_description = ""
    try:
        if isinstance(file_input, str):
            source_description = f"path: {file_input}"
            if not os.path.exists(file_input):
                raise FileNotFoundError(f"File not found at path: {file_input}")
            logger.info(f"Loading H5AD from {source_description}")
            # Consider using 'backed' mode for large files if memory is a concern,
            # but this adds complexity for downstream processing. Default to in-memory.
            adata = sc.read_h5ad(file_input)
        elif isinstance(file_input, (bytes, io.BytesIO)):
            source_description = "uploaded file bytes"
            logger.info(f"Loading H5AD from {source_description}")
            # Ensure it's BytesIO for reading
            if isinstance(file_input, bytes):
                file_bytes_io = io.BytesIO(file_input)
            else: # Already BytesIO or similar file-like object
                file_bytes_io = file_input
            adata = sc.read_h5ad(file_bytes_io)
        else:
            raise DataLoaderError(f"Invalid input type for load_h5ad: {type(file_input)}. Expected str, bytes, or BytesIO.")

        if adata is None:
            raise DataLoaderError("AnnData object is None after reading. Loading may have failed silently.")

        logger.info(f"Successfully loaded AnnData object ({adata.n_obs} obs x {adata.n_vars} vars) from {source_description}.")

        # Basic Post-Load Checks & Preparation
        try:
            adata.var_names_make_unique()
            logger.debug("Made var_names unique.")
        except Exception as e:
            logger.warning(f"Could not make var_names unique: {e}")
            # Decide if this should be a hard error or just a warning

        # Ensure obs_names are unique strings (often needed for indexing)
        if not adata.obs_names.is_unique:
             logger.warning("Observation names (adata.obs_names) are not unique. Making them unique.")
             adata.obs_names_make_unique()

        if not pd.api.types.is_string_dtype(adata.obs_names.dtype):
             logger.warning(f"Observation names (adata.obs_names) have non-string dtype ({adata.obs_names.dtype}). Converting to string.")
             adata.obs_names = adata.obs_names.astype(str)


        # Optional: Check for essential components like .X
        # Allow .X to be None initially, checks should happen where .X is needed.
        if not hasattr(adata, 'X'):
            logger.warning("Loaded AnnData object has no .X attribute.")
            # Add a placeholder if strictly needed by some downstream function?
            # This is risky, better to check where X is used.
            # adata.X = scipy.sparse.csr_matrix((adata.n_obs, adata.n_vars), dtype=np.float32)

        # Optional: Check if raw exists and has same number of vars
        if adata.raw is not None:
            try:
                 # Ensure raw has var_names before comparing length
                 if hasattr(adata.raw, 'var_names') and len(adata.raw.var_names) != adata.n_vars:
                    logger.warning(f"adata.raw exists but has a different number of variables ({adata.raw.n_vars}) than adata ({adata.n_vars}). This might cause issues if using raw data.")
            except Exception as raw_check_e:
                 logger.warning(f"Could not perform check on adata.raw structure: {raw_check_e}")


        return adata

    except FileNotFoundError as e:
        logger.error(f"H5AD Loading Error: {e}")
        raise # Re-raise specific error for clearer handling upstream
    except MemoryError as e:
         logger.error(f"MemoryError loading H5AD from {source_description}. File may be too large for available RAM.")
         raise DataLoaderError(f"Not enough memory to load the H5AD file ({source_description}).") from e
    except Exception as e:
        logger.error(f"H5AD Loading Error from {source_description}: {e}", exc_info=True)
        # Provide a more informative general error message
        raise DataLoaderError(f"Error loading H5AD file from {source_description}: {e}. Please ensure it's a valid H5AD file and has read permissions.") from e


def subsample_adata(adata, max_cells):
    """
    Subsamples AnnData object if it exceeds max_cells using scanpy.pp.subsample.
    Returns a *copy* of the subsampled data or the original data if no subsampling occurs.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")
    if not isinstance(max_cells, int) or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer.")

    if adata.n_obs > max_cells:
        logger.info(f"Subsampling AnnData from {adata.n_obs} to {max_cells} cells for visualization/analysis.")
        try:
            # Use copy=True to ensure the original object isn't modified
            adata_subsampled = sc.pp.subsample(adata, n_obs=max_cells, copy=True, random_state=0) # Add random_state for reproducibility
            logger.info(f"Subsampling complete. New shape: {adata_subsampled.shape}")
            return adata_subsampled
        except Exception as e:
             logger.error(f"Error during subsampling: {e}", exc_info=True)
             # Re-raise as AnalysisError for consistent handling
             raise AnalysisError(f"Failed to subsample data: {e}") from e
    else:
        logger.info("No subsampling needed (n_obs <= max_cells). Using original data for visualization.")
        # Return the original object reference if no subsampling happens
        # The caller should handle whether a copy is needed in this case
        return adata