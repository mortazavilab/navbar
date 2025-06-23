# navbar/app.py

import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
import os # Need os for path operations
import argparse
from collections import OrderedDict
import anndata as ad # Import AnnData

from config import (
    MAX_CELLS, N_HVG_PSEUDOBULK, DEFAULT_N_MARKERS, CACHE_MAX_ENTRIES, AGGREGATION_LAYER_OPTIONS,
    MARKER_METHODS, DEFAULT_PCA_COMPONENTS, MIN_DESEQ_COUNT_SUM, PLOT_FILE_FORMATS, LOGGING_LEVEL,
    DEFAULT_PLOT_DPI, SAVE_PLOT_DPI, DEFAULT_EMBEDDING
)
from utils import (
    setup_logging, get_adata_hash, sanitize_filename, 
    ScanpyViewerError, DataLoaderError, AggregationError, AnalysisError, FactorNotFoundError, PlottingError
)
from data_loader import load_h5ad, subsample_adata
from aggregation import aggregate_adata, cached_aggregate_adata

# Import analysis functions from their specific modules
from analysis.marker_analysis import calculate_rank_genes_df
from analysis.pca_analysis import preprocess_and_run_pca
from analysis.deg_analysis import prepare_metadata_and_run_deseq, PYDESEQ2_INSTALLED
from analysis.gsea_analysis import run_gsea_prerank, GSEAPY_INSTALLED, AVAILABLE_GENE_SETS, RANK_GROUP_COL, RANK_SCORE_COL

# Import tab rendering functions using absolute paths
from tabs import summary_tab, embedding_tab, gene_expression_tab, marker_genes_tab, pseudobulk_pca_tab, pseudobulk_deg_tab, gsea_tab


# --- Helper Function for Loading & Processing ---
def _load_and_process_adata(file_source, source_name):
    """
    Loads h5ad, subsamples, and updates session state.
    Returns True on success, False on failure.
    Manages spinner and logging within the function.
    Sets st.session_state.load_error_message on failure.
    """
    success = False
    st.session_state.load_error_message = None # Reset error specific to this attempt
    try:
        # Clear previous data state and cache for any new load attempt
        # (Keep sidebar selections etc.)
        st.session_state.adata_full = None
        st.session_state.adata_vis = None
        st.session_state.current_h5ad_source = None
        st.session_state.calculated_markers_result = None
        st.session_state.calculated_markers_error = None
        st.session_state.calculated_markers_params = None
        st.session_state.pca_adata_result = None
        st.session_state.pca_error = None
        st.session_state.pca_grouping_vars_used = []
        st.session_state.deg_results_df = None
        st.session_state.deg_error = None
        st.session_state.deg_params_display = None
        st.session_state.gsea_results_df = None
        st.session_state.gsea_error = None
        st.session_state.gsea_params_display = None
        st.cache_data.clear() # Clear all cached functions

        logger.info(f"Attempting load & process for: {source_name}")
        with st.spinner(f"Loading {source_name}..."):
            adata = load_h5ad(file_source)
            st.session_state.adata_full = adata
            st.session_state.current_h5ad_source = source_name
            logger.info(f"Successfully loaded {source_name} ({adata.n_obs}x{adata.n_vars}).")

        with st.spinner(f"Processing {source_name}..."):
            # Subsample immediately after loading
            st.session_state.adata_vis = subsample_adata(adata, MAX_CELLS)
            logger.info(f"Processing complete. Visualization data shape: {st.session_state.adata_vis.shape}")

        success = True

    except (DataLoaderError, FileNotFoundError, AnalysisError) as e:
        st.session_state.load_error_message = f"Loading/Processing Error: {e}"
        logger.error(f"Failed to load or subsample AnnData ({source_name}): {e}", exc_info=True)
        st.session_state.adata_full = None
        st.session_state.adata_vis = None
    except MemoryError as e:
        st.session_state.load_error_message = f"Memory Error: Not enough RAM to load or process '{source_name}'. Try a smaller file or increase memory."
        logger.error(f"MemoryError during loading/subsampling ({source_name}): {e}", exc_info=True)
        st.session_state.adata_full = None
        st.session_state.adata_vis = None
    except Exception as e:
        st.session_state.load_error_message = f"An unexpected error occurred during load/process: {e}"
        logger.error(f"Unexpected error during load/subsample ({source_name}): {e}", exc_info=True)
        st.session_state.adata_full = None
        st.session_state.adata_vis = None
    finally:
        # Mark that an initial load attempt happened *after* this function runs
        # This ensures it's set even if loading fails
        st.session_state.initial_load_attempted = True

    return success

# --- Initial Setup ---
st.set_page_config(layout="wide", page_title="Navbar")
setup_logging(LOGGING_LEVEL)
logger = logging.getLogger(__name__)
sc.set_figure_params(dpi=DEFAULT_PLOT_DPI, dpi_save=SAVE_PLOT_DPI)

# --- Initialize Session State ---
# Default state dictionary (including new flag)
default_state = {
    'adata_full': None,
    'adata_vis': None,
    'load_error_message': None,
    'current_h5ad_source': None,
    'show_advanced_markers': False,
    'show_advanced_pca': False,
    'show_advanced_deg': False,
    'calculated_markers_result_df': None, # <-- Changed name
    'calculated_markers_error': None,
    'calculated_markers_params': None,
    'pca_adata_result': None,
    'pca_error': None,
    'pca_grouping_vars_used': [],
    'deg_results_df': None,
    'deg_error': None,
    'deg_params_display': None,
    'gsea_results_df': None,
    'gsea_error': None,
    'gsea_params_display': None,
    'show_advanced_gsea': False, # For advanced GSEA options if needed
    'sidebar_selection_embedding': None,
    'sidebar_selection_color': None,
    'initial_load_attempted': False
}
# Initialize state keys if they don't exist
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "active_tab" not in st.session_state:
    # Check if tab is in query params
    tab_from_query = st.query_params.get("tab")
    try:
        if tab_from_query in tab_keys:
            st.session_state.active_tab = tab_from_query
        else:
            st.session_state.active_tab = "📄 Summary Info"  # Default tab
    except:
        st.session_state.active_tab = "📄 Summary Info"

st.sidebar.title("🔎 Navbar h5ad Explorer")
# --- Argument Parsing & Initial Load (Keep existing code) ---
# ... (rest of the argument parsing and initial load logic remains the same) ...
parsed_args = None
try:
    parser = argparse.ArgumentParser(description="Navbar h5ad Explorer - Argument Parser")
    parser.add_argument("--h5ad", type=str, help="Path to h5ad file to load automatically via command line.")
    # Use parse_known_args to avoid conflicts with Streamlit's own args
    parsed_args, _ = parser.parse_known_args()
    args_h5ad_path_cmd = parsed_args.h5ad if parsed_args else None
    if args_h5ad_path_cmd:
         logger.info(f"Found --h5ad command-line argument: {args_h5ad_path_cmd}")
except Exception as e:
    logger.warning(f"Argparse failed, possibly due to Streamlit context: {e}")
    args_h5ad_path_cmd = None

args_h5ad_path_query = None
try:
    args_h5ad_path_query = st.query_params.get("h5ad")
    if args_h5ad_path_query:
         logger.info(f"Found 'h5ad' query parameter: {args_h5ad_path_query}")
except Exception as e:
    logger.debug(f"Could not check query_params (may not be in Streamlit server context yet): {e}")
    args_h5ad_path_query = None

args_h5ad_path = args_h5ad_path_cmd or args_h5ad_path_query
initial_file_to_load = None
if args_h5ad_path:
    initial_file_to_load = os.path.expanduser(args_h5ad_path.strip())
    logger.info(f"Using initial file path (cleaned and expanded): {initial_file_to_load}")
else:
     logger.info("No initial file path provided via --h5ad argument or h5ad query parameter.")

load_button = None
# --- Attempt initial load ONLY if path provided AND not already attempted ---
if initial_file_to_load and not st.session_state.initial_load_attempted:
    logger.info(f"Attempting initial load from command line/query param: {initial_file_to_load}")
    # Perform path checks before calling load function
    path_exists = os.path.exists(initial_file_to_load)
    is_file = os.path.isfile(initial_file_to_load)
    can_read = os.access(initial_file_to_load, os.R_OK) if path_exists and is_file else False

    if path_exists and is_file and can_read:
        source_name = os.path.basename(initial_file_to_load)
        # Call the helper function to load and process
        load_success = _load_and_process_adata(initial_file_to_load, source_name)
        if load_success:
            logger.info(f"Initial load successful for {source_name}")
            # Display success message immediately (will also show in sidebar later)
            st.success(f"Automatically loaded: {source_name}")
        # Error message is set within the helper and will be displayed later
    elif not path_exists:
        st.session_state.load_error_message = f"Error: Initial file not found: {initial_file_to_load}"
        logger.warning(f"Initial file not found: {initial_file_to_load}")
        st.session_state.initial_load_attempted = True # Mark attempt
    elif not is_file:
        st.session_state.load_error_message = f"Error: Initial path is not a file: {initial_file_to_load}"
        logger.warning(f"Initial path is not a file: {initial_file_to_load}")
        st.session_state.initial_load_attempted = True # Mark attempt
    elif not can_read:
        st.session_state.load_error_message = f"Error: Initial file found but no read permissions: {initial_file_to_load}"
        logger.warning(f"Read permission denied for initial path: {initial_file_to_load}")
        st.session_state.initial_load_attempted = True # Mark attempt


# --- Sidebar: File Upload and Data Selection ---
st.sidebar.title("🔎 Navbar h5ad Explorer")

# Check if a file was provided via command line or query parameters
file_provided_externally = initial_file_to_load is not None

# Only show the data loading section if no file was provided externally
if not file_provided_externally:
    st.sidebar.header("1. Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload H5AD File", type=["h5ad"])
    
    # Use the cleaned initial path as default *only if* no data has been successfully loaded yet via args
    default_path_value = initial_file_to_load if initial_file_to_load and st.session_state.adata_vis is None else ""
    file_path_input = st.sidebar.text_input("Or enter file path:", value=default_path_value)
    
    load_button = st.sidebar.button("Load AnnData", key="load_data_button")
else:
    # Initialize variables to None when not showing the UI elements
    uploaded_file = None
    file_path_input = ""
    load_button = False
    
    # Optional: Show a message indicating the file was loaded from command line
    st.sidebar.text(f"Using file from command line: {os.path.basename(initial_file_to_load)}")

if load_button:
    # Logic is now simplified, mostly calls the helper function
    file_source = None
    source_name = "unknown"
    st.session_state.load_error_message = None # Reset error message specifically for button click

    if uploaded_file is not None:
        file_source = uploaded_file.getvalue()
        source_name = uploaded_file.name
        logger.info(f"Load button clicked: Attempting to load from uploaded file: {source_name}")
        _load_and_process_adata(file_source, source_name) # Call helper
    elif file_path_input:
        cleaned_path_input = file_path_input.strip()
        if not cleaned_path_input:
            st.session_state.load_error_message = "File path input cannot be empty or only whitespace."
        else:
            expanded_path = os.path.expanduser(cleaned_path_input)
            logger.info(f"Load button clicked: User provided path '{file_path_input}', cleaned to '{cleaned_path_input}', expanded to '{expanded_path}'")
            path_exists = os.path.exists(expanded_path)
            is_file = os.path.isfile(expanded_path)
            can_read = os.access(expanded_path, os.R_OK) if path_exists and is_file else False

            if path_exists and is_file and can_read:
                file_source = expanded_path
                source_name = os.path.basename(expanded_path)
                logger.info(f"Path check successful. Attempting load via button.")
                _load_and_process_adata(file_source, source_name) # Call helper
            # Handle errors for button press load attempt
            elif not path_exists:
                st.session_state.load_error_message = f"Error: File not found: {expanded_path}"
                logger.warning(f"File check failed for path: {expanded_path}")
            elif not is_file:
                st.session_state.load_error_message = f"Error: Path is not a file: {expanded_path}"
                logger.warning(f"Path is not a file: {expanded_path}")
            elif not can_read:
                st.session_state.load_error_message = f"Error: File found but no read permissions: {expanded_path}"
                logger.warning(f"Read permission denied for path: {expanded_path}")
    else:
        st.session_state.load_error_message = "Please upload an H5AD file or provide a valid file path."

    # Update sidebar success/info messages after button click attempt result
    # Check if loading was successful (adata_vis is now populated and no new error)
    if st.session_state.adata_vis is not None and not st.session_state.load_error_message:
        if st.session_state.adata_full is not None: # Ensure full is loaded too
            st.sidebar.success(f"Loaded: {st.session_state.current_h5ad_source}\n({st.session_state.adata_full.n_obs} obs x {st.session_state.adata_full.n_vars} vars)")
            if st.session_state.adata_vis.n_obs < st.session_state.adata_full.n_obs:
                st.sidebar.info(f"Using {st.session_state.adata_vis.n_obs} cells for visualization (subsampled from {st.session_state.adata_full.n_obs}).")
            else:
                st.sidebar.info(f"Using all {st.session_state.adata_vis.n_obs} cells for visualization.")
        else: # Should not happen if helper ran correctly, but handle case
             st.sidebar.warning("Data loaded for visualization, but full data reference missing.")

# Display loading errors if any (from initial load or button press)
# Ensure error is displayed *after* potential success messages from initial load are cleared
if st.session_state.load_error_message:
    st.sidebar.error(st.session_state.load_error_message)


# --- Main Application Area ---
# Check if visualization data is valid AnnData before proceeding
if isinstance(st.session_state.get('adata_vis'), ad.AnnData):

    adata_vis = st.session_state.adata_vis
    # Ensure full data is also available or handle gracefully
    adata_full = st.session_state.get('adata_full')
    if not isinstance(adata_full, ad.AnnData):
        logger.warning("Full AnnData object is not available or invalid.")
        # Ensure adata_full is at least an empty AnnData for marker check resilience
        adata_full = ad.AnnData() # Create empty AnnData as fallback

    adata_vis_hash = get_adata_hash(adata_vis) # Get hash for caching keys

    st.header(f"Exploring: `{st.session_state.current_h5ad_source}`")
    st.write(f"Using data: **{adata_vis.n_obs} cells** &times; **{adata_vis.n_vars} genes**")
    if adata_full is not None and hasattr(adata_full, 'n_obs') and adata_vis.n_obs < adata_full.n_obs:
        st.caption(f"(Subsampled from {adata_full.n_obs} cells for visualization and analysis where noted)")


    # --- Dynamically get options from adata_vis ---
    # Use try-except for robustness if obs/obsm might be missing or empty
    obs_cat_cols, obsm_keys, layer_keys = [], [], []
    valid_obs_cat_cols = [""] # Default if extraction fails
    valid_obsm_keys = [""]
    dynamic_layer_options_base = AGGREGATION_LAYER_OPTIONS.copy() # Start with base options

    try:
        if hasattr(adata_vis, 'obs') and not adata_vis.obs.empty:
            obs_cols = adata_vis.obs.columns.tolist()
            # Filter out potential all-NA columns which can cause issues in selectbox/multiselect
            obs_cat_cols = [
                col for col in obs_cols
                if (pd.api.types.is_categorical_dtype(adata_vis.obs[col]) or pd.api.types.is_object_dtype(adata_vis.obs[col]))
                   and adata_vis.obs[col].notna().any()
            ]
            valid_obs_cat_cols = obs_cat_cols if obs_cat_cols else [""] # Ensure at least one empty option
        else:
            logger.warning("adata_vis.obs is missing or empty.")
    except Exception as e:
        logger.error(f"Error accessing adata_vis.obs columns: {e}")
        st.warning("Could not read observation columns (`adata.obs`).", icon="⚠️")

    try:
        if hasattr(adata_vis, 'obsm') and adata_vis.obsm:
            obsm_keys = list(adata_vis.obsm.keys())
            valid_obsm_keys = obsm_keys if obsm_keys else [""]
        else:
            logger.warning("adata_vis.obsm is missing or empty.")
            valid_obsm_keys = [""]
    except Exception as e:
        logger.error(f"Error accessing adata_vis.obsm keys: {e}")
        st.warning("Could not read embedding keys (`adata.obsm`).", icon="⚠️")
        valid_obsm_keys = [""]

    try:
        if hasattr(adata_vis, 'layers') and adata_vis.layers:
            layer_keys = list(adata_vis.layers.keys())
    except Exception as e:
        logger.error(f"Error accessing adata_vis.layers keys: {e}")
        st.warning("Could not read layer keys (`adata.layers`).", icon="⚠️")

    # Combine dynamic layer options
    dynamic_layer_options = dynamic_layer_options_base + layer_keys


    # --- Sidebar Visualization Options ---
    if load_button or initial_file_to_load: # Check if data was loaded
        st.sidebar.header("Visualization Options")
    else:
        st.sidebar.header("2. Visualization Options") # Keep number if no auto-load
    # Embedding selection
    default_embedding_index = 0
    if DEFAULT_EMBEDDING in valid_obsm_keys:
        try: default_embedding_index = valid_obsm_keys.index(DEFAULT_EMBEDDING)
        except ValueError: pass
    elif 'X_umap' in valid_obsm_keys: # Check specific fallback
        try: default_embedding_index = valid_obsm_keys.index('X_umap')
        except ValueError: pass
    elif 'X_tsne' in valid_obsm_keys: # Check specific fallback
        try: default_embedding_index = valid_obsm_keys.index('X_tsne')
        except ValueError: pass
    elif 'X_pca' in valid_obsm_keys: # Fallback to PCA if others missing
        try: default_embedding_index = valid_obsm_keys.index('X_pca')
        except ValueError: pass

    st.session_state.sidebar_selection_embedding = st.sidebar.selectbox(
            "Embedding:",
            options=valid_obsm_keys,
            index=default_embedding_index,
            key="embedding_select_sidebar",
            help="Select the precomputed embedding coordinates from `adata.obsm`."
            )

    # Prioritized Default for Color Selection
    default_color_index = 0 # Default to the first option
    if valid_obs_cat_cols and valid_obs_cat_cols != [""]: # Check if list has valid columns
        preferred_defaults = ['leiden', 'louvain'] # Order of preference
        found_preferred = False
        for preferred_key in preferred_defaults:
            if preferred_key in valid_obs_cat_cols:
                try:
                    default_color_index = valid_obs_cat_cols.index(preferred_key)
                    logger.info(f"Found preferred default color key: '{preferred_key}' at index {default_color_index}")
                    found_preferred = True
                    break # Stop after finding the first preferred key
                except ValueError:
                    pass # Should not happen if check passed, but good practice
        if not found_preferred and valid_obs_cat_cols: # Ensure list isn't empty before accessing index 0
            logger.info(f"Preferred default keys ({preferred_defaults}) not found. Defaulting to first categorical column: '{valid_obs_cat_cols[0]}'")
            default_color_index = 0 # Explicitly set back to 0 if loop didn't find anything

    # Color selection (Categorical only for now)
    st.session_state.sidebar_selection_color = st.sidebar.selectbox(
            "Color Cells By:",
            options=valid_obs_cat_cols,
            index=default_color_index, # Use calculated default index
            key="color_select_sidebar",
            help="Select categorical observation data from `adata.obs` to color cells."
            )


    # --- Define Tabs ---
    tab_keys = [
        "📄 Summary Info", 
        "📊 Embedding Plot",
        "📈 Gene Expression",
        "🧬 Marker Genes",
        "📊 Pathway Analysis",
        "🔬 Pseudobulk PCA",
        "🧪 Pseudobulk DEG"
    ]

    # Create the tabs with st.tabs
    tab_objects = st.tabs(tab_keys)
    # Update query parameter to match active tab
    if st.session_state.active_tab in tab_keys:
        st.query_params["tab"] = st.session_state.active_tab

    # Get the currently selected tab index based on session state
    selected_tab_index = tab_keys.index(st.session_state.active_tab) if st.session_state.active_tab in tab_keys else 0
    # Unpack tabs corresponding to keys
    tab_summary, tab_embedding, tab_gene_expr, tab_markers, tab_gsea, tab_pca, tab_deg = tab_objects

    # --- Analysis Execution Forms (Placed within relevant tab context) ---

    # -- Render Tabs using Imported Functions ---
    #--- Pathway Analysis Tab ---
    with tab_summary: 
        summary_tab.render_summary_tab(adata_vis)

    with tab_embedding: # Embedding Plot
        embedding_tab.render_embedding_tab(
            adata_vis,
            st.session_state.sidebar_selection_embedding,
            st.session_state.sidebar_selection_color
        )

    with tab_gene_expr: # Gene Expression
        gene_expression_tab.render_gene_expression_tab(
            adata_vis,
            st.session_state.sidebar_selection_embedding
        )

    with tab_markers: # Marker Genes Display (Form is also here)
        marker_genes_tab.render_marker_genes_tab(
            adata_vis=adata_vis,
            adata_full=adata_full, # Pass full adata for checking precomputed markers
            valid_obs_cat_cols=valid_obs_cat_cols,
            selected_color_var=st.session_state.sidebar_selection_color # Pass sidebar selection for context
        )
    
    with tab_gsea:
        gsea_tab.render_gsea_tab()

    with tab_pca: # Pseudobulk PCA Display (Form is also here)
        pseudobulk_pca_tab.render_pseudobulk_pca_tab(
            adata_vis=adata_vis,
            valid_obs_cat_cols=valid_obs_cat_cols,
            dynamic_layer_options=dynamic_layer_options
        )

    with tab_deg: # Pseudobulk DEG Display (Form is also here)
        pseudobulk_deg_tab.render_pseudobulk_deg_tab(
            adata_vis=adata_vis,
            valid_obs_cat_cols=valid_obs_cat_cols,
            dynamic_layer_options=dynamic_layer_options
        )


# --- Footer or Initial Prompt ---
elif not load_button and st.session_state.adata_full is None and not st.session_state.load_error_message:
    st.info("⬅️ Upload an H5AD file or provide a valid file path (e.g., using `--h5ad` argument) to begin.")

# Add a final check for the case where adata_vis failed to initialize properly
elif st.session_state.load_error_message and st.session_state.adata_vis is None:
    st.error(f"Could not initialize data. Error: {st.session_state.load_error_message}")