# scanpy_viewer/analysis/marker_analysis.py

import scanpy as sc
import anndata as ad
import pandas as pd
import logging
# Use relative import within package if 'utils' is in the parent directory
try:
    from ..utils import AnalysisError, FactorNotFoundError
except ImportError:
    from utils import AnalysisError, FactorNotFoundError


logger = logging.getLogger(__name__)

# --- Define standard columns we want to extract ---
MARKER_DF_COLS = ['group', 'names', 'scores', 'logfoldchanges', 'pvals_adj', 'pvals']
# --- MODIFIED: Added actual column names found from sc.get ---
MARKER_DF_OPTIONAL_COLS = ['pts', 'pts_rest', 'pct_nz_group', 'pct_nz_reference']

# --- get_markers_df_from_uns function remains the same ---
# It already checks MARKER_DF_OPTIONAL_COLS, so it will now pick up pct_nz_* if present
def get_markers_df_from_uns(adata_uns, expected_groupby=None):
    """
    Safely extracts the full marker DataFrame from adata.uns['rank_genes_groups'].
    Includes standard columns plus optional fraction columns if available.
    """
    warning_flag = False; groupby_key = None; markers_df = None
    if not isinstance(adata_uns, dict): logger.warning("Input adata_uns not dict."); return None, None, warning_flag
    if 'rank_genes_groups' not in adata_uns: logger.info("rank_genes_groups not in adata.uns"); return None, None, warning_flag
    rgg_results = adata_uns['rank_genes_groups']
    if not isinstance(rgg_results, dict): logger.warning(".uns['rank_genes_groups'] not dict."); return None, None, warning_flag
    params = rgg_results.get('params', {}); groupby_key = params.get('groupby') if isinstance(params, dict) else None; method = params.get('method', 'unknown') if isinstance(params, dict) else 'unknown'
    logger.info(f"Found precomputed markers (groupby='{groupby_key}', method='{method}').")
    if expected_groupby is not None and groupby_key != expected_groupby: logger.warning(f"Precomputed groupby='{groupby_key}', selected='{expected_groupby}'."); warning_flag = True
    try:
        temp_ad = ad.AnnData(shape=(0,0), uns={'rank_genes_groups': rgg_results})
        markers_df_raw = sc.get.rank_genes_groups_df(temp_ad, group=None, pval_cutoff=None, log2fc_min=None)
        if markers_df_raw.empty: logger.warning("sc.get returned empty DF for precomputed."); return None, groupby_key, warning_flag
        cols_to_select = []
        for col in MARKER_DF_COLS:
             if col in markers_df_raw.columns: cols_to_select.append(col)
             else: logger.warning(f"Expected column '{col}' missing from precomputed markers.")
        # Check all possible optional column names
        for col in MARKER_DF_OPTIONAL_COLS:
             if col in markers_df_raw.columns: cols_to_select.append(col)
        if 'names' not in cols_to_select or 'group' not in cols_to_select: raise AnalysisError("Essential columns missing from precomputed markers.")
        markers_df = markers_df_raw[cols_to_select].copy(); markers_df = markers_df.sort_values(by=['group', 'scores'], ascending=[True, False])
        logger.info(f"Successfully extracted precomputed markers DF for key '{groupby_key}'. Columns: {markers_df.columns.tolist()}")
        return markers_df, groupby_key, warning_flag
    except ImportError as ie: raise AnalysisError(f"Failed extract (sc.get missing?): {ie}") from ie
    except Exception as e: logger.error(f"Error parsing precomputed markers: {e}", exc_info=True); return None, groupby_key, warning_flag

# --- calculate_rank_genes_df (Debug logging removed) ---
def calculate_rank_genes_df(adata_obj, groupby_key, method='wilcoxon', use_raw=False):
    """
    Calculates marker genes using scanpy.tl.rank_genes_groups and returns the full DataFrame.
    Includes standard columns plus optional fraction columns if available.
    """
    # [ Initial checks and data prep code remains the same... ]
    if not isinstance(adata_obj, ad.AnnData): raise TypeError("Input must be AnnData.")
    if groupby_key not in adata_obj.obs: raise FactorNotFoundError(f"Grouping key '{groupby_key}' not found.")
    data_description = ""; temp_adata = None
    try: # Data Prep
        if use_raw:
            if adata_obj.raw is None: raise AnalysisError("adata.raw is None.")
            adata_target = adata_obj.raw; data_description = "adata.raw.X"
            common_index = adata_obj.obs_names.intersection(adata_target.obs_names)
            if len(common_index) == 0: raise AnalysisError("No common obs between adata and adata.raw.")
            temp_adata = ad.AnnData(X=adata_target[common_index].X, var=adata_target.var)
            temp_adata.obs = adata_obj.obs.loc[common_index, [groupby_key]].copy()
        else:
            temp_adata = adata_obj.copy(); data_description = "adata.X"
    except Exception as e: raise AnalysisError(f"Error preparing data (use_raw={use_raw}): {e}") from e
    if temp_adata is None: raise AnalysisError("Failed to create temp AnnData.")
    logger.info(f"Calculating markers: '{method}', group='{groupby_key}', data='{data_description}'...")
    if groupby_key not in temp_adata.obs: raise FactorNotFoundError(f"'{groupby_key}' missing from temp AnnData.")
    if not pd.api.types.is_categorical_dtype(temp_adata.obs[groupby_key]):
        logger.warning(f"'{groupby_key}' not categorical. Converting.");
        try: temp_adata.obs[groupby_key] = temp_adata.obs[groupby_key].astype('category')
        except Exception as conv_e: raise AnalysisError(f"Cannot convert '{groupby_key}' to category: {conv_e}") from conv_e
    if temp_adata.n_obs < 2 or temp_adata.n_vars < 1: raise ValueError(f"Insufficient data (shape: {temp_adata.shape}).")
    group_counts = temp_adata.obs[groupby_key].value_counts()
    if len(group_counts) < 2: raise ValueError(f"Need >= 2 groups. Found {len(group_counts)} for '{groupby_key}'.")
    if (group_counts < 2).any(): logger.warning(f"Groups have < 2 obs: {group_counts[group_counts < 2].index.tolist()}. Method '{method}' might fail/warn.")

    # Run Scanpy Calculation
    rank_key = "rank_genes_groups_calculated"
    try:
        sc.tl.rank_genes_groups(temp_adata, groupby=groupby_key, method=method, use_raw=False, key_added=rank_key, pts=True, n_genes=temp_adata.n_vars)
        logger.info("Scanpy rank_genes_groups calculation complete.")
        # Optional: Log keys stored just to be sure, but removed detailed type/shape logs
        # if rank_key in temp_adata.uns: logger.debug(f".uns keys after calc: {list(temp_adata.uns[rank_key].keys())}")

    except Exception as e: raise AnalysisError(f"Error calculating rank_genes: {e}") from e

    # Extract Full Results DataFrame using sc.get
    try:
        logger.info(f"Attempting extraction with sc.get.rank_genes_groups_df (key='{rank_key}')...")
        markers_df_raw = sc.get.rank_genes_groups_df(temp_adata, key=rank_key, group=None)
        logger.debug(f"Columns returned by sc.get: {markers_df_raw.columns.tolist()}") # Keep this debug line

        if markers_df_raw.empty: raise AnalysisError("sc.get returned empty DataFrame.")

        # Select and reorder desired columns based on updated list
        cols_to_select = []
        found_optional = []
        for col in MARKER_DF_COLS:
             if col in markers_df_raw.columns: cols_to_select.append(col)
             else: logger.warning(f"Expected column '{col}' missing from sc.get DataFrame.")
        # Check all possible optional column names
        for col in MARKER_DF_OPTIONAL_COLS:
             if col in markers_df_raw.columns:
                  cols_to_select.append(col)
                  found_optional.append(col)

        logger.info(f"Found optional columns in sc.get DataFrame: {found_optional}")

        if 'names' not in cols_to_select or 'group' not in cols_to_select:
             raise AnalysisError("Essential 'names' or 'group' column missing from sc.get DataFrame.")

        markers_df = markers_df_raw[cols_to_select].copy()
        markers_df = markers_df.sort_values(by=['group', 'scores'], ascending=[True, False])

        logger.info(f"Successfully extracted markers DataFrame. Final columns: {markers_df.columns.tolist()}")
        del temp_adata
        return markers_df

    except ImportError as ie: raise AnalysisError(f"Failed extract (sc.get missing?): {ie}") from ie
    except KeyError as e: raise AnalysisError(f"Failed extract (missing columns?): {e}") from e
    except Exception as e: raise AnalysisError(f"Failed to extract results via sc.get after calculation: {e}") from e