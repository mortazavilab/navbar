# analysis/gsea_analysis.py
import pandas as pd
import logging
import streamlit as st # For caching

# Try importing gseapy, handle if not installed
try:
    import gseapy
    GSEAPY_INSTALLED = True
    # Get available gene sets (this might make an API call, do it once)
    try:
        AVAILABLE_GENE_SETS = gseapy.get_library_name()
    except Exception as e:
        logging.warning(f"Could not fetch gseapy gene set libraries: {e}")
        AVAILABLE_GENE_SETS = ["Error fetching libraries"] # Provide fallback
except ImportError:
    GSEAPY_INSTALLED = False
    AVAILABLE_GENE_SETS = ["gseapy not installed"]
    logging.warning("gseapy library not found. Pathway analysis will be disabled.")

from config import CACHE_MAX_ENTRIES 
from utils import AnalysisError 

logger = logging.getLogger(__name__)

# --- Constants ---
# Define standard columns expected from calculate_rank_genes_df
# Adjust these if your function returns different column names
RANK_GENE_COL = 'names'
RANK_GROUP_COL = 'group'
RANK_SCORE_COL = 'scores' # Or 'logfoldchanges', depending on what you want to rank by

def inspect_gsea_results(pre_res):
    """Helper function to inspect GSEA results structure for debugging."""
    logger.debug("GSEA Results inspection:")
    logger.debug(f"Type of pre_res: {type(pre_res)}")
    
    if hasattr(pre_res, 'res2d'):
        logger.debug(f"res2d shape: {pre_res.res2d.shape}")
        logger.debug(f"res2d columns: {pre_res.res2d.columns.tolist()}")
        logger.debug(f"res2d head:\n{pre_res.res2d.head()}")
    
    if hasattr(pre_res, 'results'):
        logger.debug(f"results type: {type(pre_res.results)}")
        logger.debug(f"results length: {len(pre_res.results)}")

@st.cache_data(max_entries=CACHE_MAX_ENTRIES, show_spinner=False) # Cache GSEA results
def run_gsea_prerank(
    _marker_results_df_ref: pd.DataFrame, # Pass DataFrame for cache dependency
    marker_df_hash: str, # Pass hash of the df for cache invalidation
    selected_group: str,
    gene_sets: str,
    ranking_metric: str = RANK_SCORE_COL,
    min_size: int = 15,
    max_size: int = 500,
    permutation_num: int = 100, # Keep low for speed in interactive app
    seed: int = 42,
    threads: int = 1
) -> pd.DataFrame:
    """
    Runs GSEA Prerank using gseapy on a specific group from marker gene results.

    Args:
        _marker_results_df_ref: DataFrame containing the marker results
                                (must contain group, gene name, and score columns).
                                Passed by reference for caching dependency.
        marker_df_hash: A hash string representing the state of the marker df.
        selected_group: The specific group/cluster to analyze from the marker results.
        gene_sets: Name of the gene set library (e.g., 'KEGG_2019_Human').
        ranking_metric: Column name in _marker_results_df_ref to use for ranking.
        min_size: Minimum size of gene sets to consider.
        max_size: Maximum size of gene sets to consider.
        permutation_num: Number of permutations for calculating significance.
        seed: Random seed for reproducibility.
        threads: Number of threads to use.

    Returns:
        DataFrame: The GSEA Prerank results from gseapy.

    Raises:
        AnalysisError: If gseapy is not installed, data is missing, or gsea fails.
        ValueError: If input parameters are invalid.
    """
    if not GSEAPY_INSTALLED:
        raise AnalysisError("gseapy library is not installed. Cannot run Pathway Analysis.")

    if _marker_results_df_ref is None or _marker_results_df_ref.empty:
        raise AnalysisError("Marker results are not available or empty.")

    required_cols = [RANK_GROUP_COL, RANK_GENE_COL, ranking_metric]
    if not all(col in _marker_results_df_ref.columns for col in required_cols):
        raise AnalysisError(f"Marker results DataFrame missing required columns: {required_cols}. Found: {_marker_results_df_ref.columns.tolist()}")

    if selected_group not in _marker_results_df_ref[RANK_GROUP_COL].unique():
        raise ValueError(f"Selected group '{selected_group}' not found in marker results groups.")

    logger.info(f"Starting GSEA Prerank for group '{selected_group}' using gene sets '{gene_sets}' and ranking metric '{ranking_metric}'.")

    # Filter for the selected group and prepare the ranked list
    group_df = _marker_results_df_ref[_marker_results_df_ref[RANK_GROUP_COL] == selected_group].copy()

    # Ensure no NaN values in the ranking metric column for this group
    if group_df[ranking_metric].isnull().any():
        logger.warning(f"NaN values found in ranking metric '{ranking_metric}' for group '{selected_group}'. Dropping these genes.")
        group_df = group_df.dropna(subset=[ranking_metric])

    if group_df.empty:
         raise AnalysisError(f"No valid genes found for group '{selected_group}' after filtering.")
    group_df[RANK_GENE_COL] = group_df[RANK_GENE_COL].astype(str)  # Ensure gene names are strings
    group_df[RANK_GENE_COL] = group_df[RANK_GENE_COL].str.upper()  # Convert gene names to uppercase for consistency
    # Prepare the rank list: DataFrame with gene names and scores
    # gseapy.prerank expects a Series/DataFrame indexed by gene name/ID with scores as values
    # Or a two-column DataFrame [gene_name, score]
    rank_list = group_df[[RANK_GENE_COL, ranking_metric]].set_index(RANK_GENE_COL)
    rank_list = rank_list.squeeze() # Convert to Series if possible
    rank_list = rank_list.sort_values(ascending=False) # Important: GSEA ranks from high to low

    if rank_list.empty:
        raise AnalysisError(f"Rank list for group '{selected_group}' is empty after processing.")

    logger.debug(f"Prepared rank list for GSEA Prerank (Top 5): \n{rank_list.head()}")

    try:
        # Run Prerank
        print(f"rank list: \n{rank_list}")
        pre_res = gseapy.prerank(
            rnk=rank_list,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutation_num, # Reduced permutations
            outdir=None, # Don't write files
            seed=seed,
            verbose=True, # Logs progress to console/log file
            threads=threads
        )
        if logger.isEnabledFor(logging.DEBUG):
            inspect_gsea_results(pre_res)

        if pre_res is None:
            logger.warning(f"GSEA Prerank returned no results for group '{selected_group}', gene set '{gene_sets}'.")
            # Return an empty DataFrame with expected columns if possible
            # Check the specific version of gseapy for exact column names if needed
            return pd.DataFrame(columns=['Term', 'NES', 'p_val', 'fdr', 'genes']) # Example columns

        logger.info(f"GSEA Prerank completed successfully for group '{selected_group}'. Found {len(pre_res.results)} terms.")
        # Check what columns are actually available
        results_df = pre_res.res2d
        logger.debug(f"Available columns in GSEA results: {results_df.columns.tolist()}")

        # Try different possible column names for FDR
        fdr_columns = ['fdr', 'FDR q-val', 'FDR', 'qval', 'q_val', 'padj']
        sort_column = None

        for col in fdr_columns:
            if col in results_df.columns:
                sort_column = col
                break

        # If no FDR column found, try p-value columns
        if sort_column is None:
            pval_columns = ['pval', 'p_val', 'NOM p-val', 'p-val', 'pvalue']
            for col in pval_columns:
                if col in results_df.columns:
                    sort_column = col
                    break

        # If still no sorting column found, use the first numeric column
        if sort_column is None:
            numeric_cols = results_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                sort_column = numeric_cols[0]
                logger.warning(f"No standard p-value or FDR column found. Sorting by '{sort_column}'")
            else:
                logger.warning("No numeric columns found for sorting. Returning unsorted results.")
                return results_df

        # Sort by the identified column
        return results_df.sort_values(sort_column, ascending=True)

    except Exception as e:
        logger.error(f"GSEA Prerank failed for group '{selected_group}': {e}", exc_info=True)
        # Check for specific gseapy errors if necessary
        if "Possible reason: check gene names" in str(e):
            raise AnalysisError(f"GSEA Prerank failed. Check if gene names in marker results ('{RANK_GENE_COL}' column) match the identifiers expected by the selected gene set '{gene_sets}'. Error: {e}")
        raise AnalysisError(f"An error occurred during GSEA Prerank: {e}")

