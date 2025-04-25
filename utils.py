# scanpy_viewer/utils.py

import pandas as pd
import numpy as np
import scipy.sparse
import logging
import streamlit as st
import io
import os
import re
import datetime
import anndata as ad
import matplotlib.pyplot as plt
from config import USE_DATE_IN_FILENAMES # Import config

# --- Custom Exceptions ---
class ScanpyViewerError(Exception):
    """Base exception for the application."""
    pass

class DataLoaderError(ScanpyViewerError):
    """Exception related to loading data."""
    pass

class AggregationError(ScanpyViewerError):
    """Exception related to data aggregation."""
    pass

class AnalysisError(ScanpyViewerError):
    """Exception related to analysis steps (PCA, DEG, Markers)."""
    pass

class GSEAError(AnalysisError):
    """Exception specific to GSEA analysis steps."""
    pass
    
class FactorNotFoundError(AnalysisError):
    """Exception when a required factor/column is missing."""
    pass

class PlottingError(ScanpyViewerError):
    """Exception related to generating plots."""
    pass


# --- Helper Functions ---

def setup_logging(level="INFO"):
    """Configures basic logging using Streamlit's logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    # Get the root logger potentially configured by Streamlit
    logger = logging.getLogger()
    # Set level for the root logger
    logger.setLevel(log_level)

    # Check if handlers already exist (to avoid duplicate logs in some environments)
    if not logger.handlers:
        # Add a handler if none exist (e.g., basic Streamlit setup might not add one)
        handler = logging.StreamHandler() # Default to stderr
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logging.info(f"Logging level set to {level.upper()}")


def copy_index_to_obs(adata_agg):
    """
    Helper to copy index (single or multi) to obs columns, ensuring categorical.
    Returns the modified AnnData object. Operates on a copy.
    Raises ValueError if issues occur during conversion.
    """
    logger = logging.getLogger(__name__)
    if not isinstance(adata_agg, ad.AnnData):
         raise TypeError("Input must be an AnnData object.")
    adata_copy = adata_agg.copy() # Work on a copy

    if isinstance(adata_copy.obs.index, pd.MultiIndex):
        for i, name in enumerate(adata_copy.obs.index.names):
            level_values = adata_copy.obs.index.get_level_values(i)
            # Use a predictable name if the original index level name is None or empty
            col_name = name if name and isinstance(name, str) else f'_index_level_{i}'

            # Add or update the column in .obs
            try:
                # Convert level values to categorical
                categorical_values = pd.Categorical(level_values)
                # Check if column exists and needs update, or add if new
                if col_name not in adata_copy.obs.columns or not adata_copy.obs[col_name].equals(categorical_values):
                    adata_copy.obs[col_name] = categorical_values
                # If it exists but is not categorical, force conversion
                elif not pd.api.types.is_categorical_dtype(adata_copy.obs[col_name]):
                    adata_copy.obs[col_name] = categorical_values
            except Exception as e:
                logger.warning(f"Could not assign or convert multi-index level '{col_name}' to categorical obs column: {e}. Skipping this level.")
                # Optionally raise error: raise ValueError(f"Failed to set obs column '{col_name}' from index: {e}") from e

    else: # Single index
        name = adata_copy.obs.index.name
        idx_values = adata_copy.obs.index
        col_name_to_set = name if name and isinstance(name, str) else '_index' # Use '_index' if index is unnamed

        try:
            categorical_values = pd.Categorical(idx_values)
            if col_name_to_set not in adata_copy.obs.columns or not adata_copy.obs[col_name_to_set].equals(categorical_values):
                adata_copy.obs[col_name_to_set] = categorical_values
            elif not pd.api.types.is_categorical_dtype(adata_copy.obs[col_name_to_set]):
                adata_copy.obs[col_name_to_set] = categorical_values
        except Exception as e:
            logger.warning(f"Could not assign or convert single-index '{col_name_to_set}' to categorical obs column: {e}. Skipping.")
            # Optionally raise error

    return adata_copy


def create_group_string(row, factors):
    """
    Creates a combined string identifier from specified factors in a DataFrame row.
    Handles potential non-string data types in the row.
    Returns the combined string or raises KeyError/ValueError on issues.
    """
    try:
        # Ensure factors are accessed correctly, converting each element to string
        return '_'.join(str(row[factor]) for factor in factors)
    except KeyError as e:
        logging.error(f"KeyError creating group string: Factor '{e}' not found in row with columns {row.index.tolist()}.")
        raise # Re-raise the KeyError for calling function to handle
    except Exception as e:
        logging.error(f"Unexpected error creating group string for factors {factors}: {e}")
        raise ValueError(f"Failed to create group string: {e}") from e


def get_figure_bytes(fig, format="png", dpi=300):
    """Exports a matplotlib figure to bytes and closes the figure."""
    if fig is None:
        raise ValueError("Input figure is None.")
    try:
        buf = io.BytesIO()
        # Use bbox_inches='tight' to prevent labels cutoff
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        # Crucially, close the figure to release memory
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        # Ensure figure is closed even if save failed
        try:
            plt.close(fig)
        except Exception:
            pass # Ignore errors during cleanup
        raise PlottingError(f"Failed to export figure to {format}: {e}") from e


def generate_download_button(data_bytes, filename, label="Download Data", mime='application/octet-stream', key=None, help_text=None):
    """Generates a Streamlit download button with basic validation."""
    if not isinstance(data_bytes, bytes):
        st.error("Invalid data type for download button (internal error: must be bytes).")
        logging.error(f"generate_download_button called with non-bytes type: {type(data_bytes)}")
        return

    try:
        st.download_button(
            label=label,
            data=data_bytes,
            file_name=filename,
            mime=mime,
            key=key,
            help=help_text
        )
    except Exception as e:
         st.error(f"Failed to create download button for '{filename}': {e}")
         logging.error(f"Download button creation failed: {e}", exc_info=True)


def get_adata_hash(adata):
    """
    Generates a simple hashable proxy for an AnnData object for caching purposes.
    Focuses on metadata and shape, as hashing .X can be very slow.
    """
    if adata is None:
        return None
    try:
        # Combine shape, obs/var columns, layers, obsm, uns keys, and raw info as a proxy
        obs_cols = tuple(sorted(adata.obs.columns)) if hasattr(adata, 'obs') else ()
        var_cols = tuple(sorted(adata.var.columns)) if hasattr(adata, 'var') else ()
        layer_keys = tuple(sorted(adata.layers.keys())) if hasattr(adata, 'layers') else ()
        obsm_keys = tuple(sorted(adata.obsm.keys())) if hasattr(adata, 'obsm') else ()
        uns_keys = tuple(sorted(adata.uns.keys())) if hasattr(adata, 'uns') else () # Basic check on uns

        # Add raw info if exists
        raw_info = None
        if adata.raw:
            try:
                 # Check if raw var_names exist before accessing
                 raw_var_names = tuple(sorted(adata.raw.var_names[:20])) if hasattr(adata.raw, 'var_names') else ()
                 raw_info = (adata.raw.n_obs, adata.raw.n_vars, raw_var_names)
            except Exception as raw_e:
                 logging.warning(f"Could not get raw info for hash: {raw_e}")
                 raw_info = None

        # Consider hashing a sample of .X? Risky/slow. Let's skip .X hashing.
        # Maybe include basic stats of X if needed? e.g., adata.X.min(), adata.X.max(), adata.X.sum() if not too slow
        # x_stats = None
        # try:
        #      if hasattr(adata, 'X') and adata.X is not None:
        #           # Be careful with sparse matrices and sum()
        #           x_sum = adata.X.sum() if adata.X.size > 0 else 0
        #           x_stats = (adata.X.min(), adata.X.max(), x_sum)
        # except Exception: # Catch errors during stat calculation
        #      x_stats = None

        return (adata.shape, obs_cols, var_cols, layer_keys, obsm_keys, uns_keys, raw_info) # Add x_stats here if calculated

    except Exception as e:
        logging.warning(f"Could not generate detailed hash for AnnData object: {e}. Caching might be less precise.")
        # Fallback to a very simple hash if detailed hashing fails
        return (adata.shape,)

def sanitize_filename(filename_base, extension=""):
    """Removes potentially problematic characters and adds date if configured."""
    # Remove or replace characters not typically allowed in filenames
    # Allow alphanumeric, underscore, hyphen, period
    sanitized = re.sub(r'[^\w.-]+', '_', filename_base)
    # Remove leading/trailing underscores/periods
    sanitized = sanitized.strip('._')

    # Add date prefix if configured
    if USE_DATE_IN_FILENAMES:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        sanitized = f"{date_str}_{sanitized}"

    # Limit length gracefully (e.g., 100 chars for base)
    max_len_base = 100
    if len(sanitized) > max_len_base:
        sanitized = sanitized[:max_len_base].rstrip('._') # Trim and clean again

    # Ensure it's not empty
    sanitized = sanitized if sanitized else "download"

    # Add extension
    if extension:
        sanitized = f"{sanitized}.{extension.lstrip('.')}"

    return sanitized
