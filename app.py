# scanpy_viewer/app.py

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
from tabs import summary_tab, embedding_tab, gene_expression_tab, marker_genes_tab, pseudobulk_pca_tab, pseudobulk_deg_tab


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

    # Marker Gene Calculation Form (remains in its tab)
    with tab_markers:
        st.markdown("---") # Separator from precomputed results display
        st.markdown("#### Calculate Markers (on current data)")
        st.caption(f"Runs `sc.tl.rank_genes_groups` on the current data ({adata_vis.n_obs} cells). Results will appear below the form.")
        with st.form(key="marker_calc_form"):
            # Group by selector (mirrors display selector but needed for form submission)
            marker_groupby_calc = st.selectbox(
                "Group By:",
                options=valid_obs_cat_cols,
                # Use the selection from the *display* dropdown as the default
                index=valid_obs_cat_cols.index(st.session_state.get('marker_group_select_display', st.session_state.sidebar_selection_color)) if st.session_state.get('marker_group_select_display', st.session_state.sidebar_selection_color) in valid_obs_cat_cols else 0,
                key="marker_group_calc_form"
            )
            col_mark1, col_mark2 = st.columns([2,1])
            marker_method = col_mark1.selectbox("Method:", options=MARKER_METHODS, key="marker_method_select_form")
            n_markers_calc = col_mark2.number_input("Top N:", min_value=1, max_value=50, value=DEFAULT_N_MARKERS, key="marker_n_calc_form")

            # Option to use raw data if available
            use_raw_markers = False
            raw_available = (adata_vis.raw is not None)
            if raw_available:
                use_raw_markers = st.checkbox("Use adata.raw", value=False, key="marker_use_raw_form", help="Calculate markers using data from `adata.raw.X` instead of `adata.X`")

            submitted_markers = st.form_submit_button("Calculate Markers Now")

            if submitted_markers:
                st.session_state.active_tab = "🧬 Marker Genes"
                st.session_state.calculated_markers_result = None # Clear previous results
                st.session_state.calculated_markers_error = None
                if not marker_groupby_calc:
                    st.warning("Please select a 'Group By' variable.")
                    st.session_state.calculated_markers_error = "No grouping variable selected."
                else:
                    with st.spinner(f"Calculating markers ({marker_method}, N={n_markers_calc}, GroupBy={marker_groupby_calc}, Use Raw={use_raw_markers})..."):
                        try:
                            # Call analysis function from the specific module
                            top_markers_calculated = calculate_rank_genes_df(
                                adata_vis, # Pass the main object, function handles raw access
                                groupby_key=marker_groupby_calc,
                                method=marker_method,
                                #n_genes=n_markers_calc,
                                use_raw=use_raw_markers
                            )
                            st.session_state.calculated_markers_result_df = top_markers_calculated # Store in state for display
                            st.session_state.calculated_markers_params = { # Store params for context
                                'groupby': marker_groupby_calc,
                                'method': marker_method,
                                'n_genes': n_markers_calc,
                                'use_raw': use_raw_markers
                            }
                            #st.session_state.calculated_markers_error = None # Clear any previous error
                            st.success("Marker calculation complete. Results displayed below.") # Add success message
                        except (AnalysisError, FactorNotFoundError, ValueError, TypeError) as e:
                            st.session_state.calculated_markers_error = f"Marker Calculation Error: {e}"
                            logger.error(f"Marker calculation failed: {e}", exc_info=True)
                        except Exception as e:
                            st.session_state.calculated_markers_error = f"An unexpected error occurred during marker calculation: {e}"
                            logger.error(f"Unexpected marker calc error: {e}", exc_info=True)

    # --- NEW: Pathway Analysis Tab ---
    with tab_gsea:
        st.header("Pathway Enrichment Analysis (GSEA Prerank)")

        if not GSEAPY_INSTALLED:
            st.warning("`gseapy` library not installed. Pathway analysis is unavailable. Please install it (`pip install gseapy`) and restart.", icon="⚠️")
        elif 'calculated_markers_result_df' not in st.session_state or st.session_state.calculated_markers_result_df is None:
            st.info("📊 Please calculate marker genes in the 'Marker Genes' tab first to enable Pathway Analysis.")
        else:
            st.markdown("Perform Gene Set Enrichment Analysis (GSEA) using the ranked gene lists from the 'Marker Genes' tab.")

            # Display info about the source marker calculation
            marker_params = st.session_state.get('calculated_markers_params', {})
            if marker_params:
                 st.caption(f"Using marker gene results calculated with: GroupBy='{marker_params.get('groupby', 'N/A')}', Method='{marker_params.get('method', 'N/A')}', Use Raw='{marker_params.get('use_raw', 'N/A')}'")
            else:
                 st.caption("Using previously calculated marker gene results.")

            # GSEA Calculation Form
            with st.form("gsea_form"):
                 # Select group from marker results
                 marker_groups = [""] # Add empty option
                 try:
                     # Get unique groups from the marker results dataframe
                     if st.session_state.calculated_markers_result_df is not None and RANK_GROUP_COL in st.session_state.calculated_markers_result_df.columns:
                          marker_groups.extend(sorted(st.session_state.calculated_markers_result_df[RANK_GROUP_COL].astype(str).unique()))
                     else:
                         st.warning("Could not extract groups from marker results.")
                 except Exception as e:
                     st.error(f"Error reading groups from marker results: {e}")
                     logger.error(f"Error reading marker groups: {e}")

                 gsea_selected_group = st.selectbox(
                     "Select Group/Cluster to Analyze:",
                     options=marker_groups,
                     key="gsea_group_select_form",
                     help="Choose the specific group from the marker gene results for GSEA."
                 )

                 # Select gene set library
                 gsea_gene_sets = st.selectbox(
                     "Select Gene Set Library:",
                     options=AVAILABLE_GENE_SETS,
                     key="gsea_library_select_form",
                     help="Choose the gene set database (e.g., KEGG, GO) for enrichment analysis. Requires internet connection on first use for some libraries."
                 )

                 # Advanced Options (Optional)
                 st.checkbox("Show Advanced GSEA Options", key="show_advanced_gsea")
                 gsea_min_size = 15
                 gsea_max_size = 500
                 gsea_permutations = 100 # Keep low for web app speed
                 gsea_ranking_metric = marker_params.get('ranking_metric', RANK_SCORE_COL) # Use default from markers calc

                 if st.session_state.show_advanced_gsea:
                     with st.expander("Advanced GSEA Options"):
                         # Allow overriding the ranking metric if marker df has multiple (e.g., logfc)
                         # This requires knowing the columns produced by calculate_rank_genes_df
                         possible_metrics = [col for col in st.session_state.calculated_markers_result_df.columns if col in ['scores', 'logfoldchanges', 'pvals_adj']] # Example potential metrics
                         if not possible_metrics: possible_metrics = [RANK_SCORE_COL] # Fallback
                         gsea_ranking_metric = st.selectbox("Ranking Metric:", options=possible_metrics, index=possible_metrics.index(gsea_ranking_metric) if gsea_ranking_metric in possible_metrics else 0, key="gsea_metric_select")

                         gsea_min_size = st.number_input("Min Gene Set Size:", min_value=1, value=15, key="gsea_min_size")
                         gsea_max_size = st.number_input("Max Gene Set Size:", min_value=10, value=500, key="gsea_max_size")
                         gsea_permutations = st.number_input("Number of Permutations:", min_value=10, max_value=1000, value=100, step=10, key="gsea_perms", help="Higher numbers increase runtime but improve p-value accuracy.")


                 # Submit Button
                 run_gsea_button = st.form_submit_button("Run Pathway Analysis")

                 if run_gsea_button:
                     st.session_state.active_tab = "📊 Pathway Analysis"
                     st.session_state.gsea_results_df = None # Clear previous results
                     st.session_state.gsea_error = None

                     if not gsea_selected_group:
                         st.warning("Please select a Group/Cluster to analyze.")
                         st.session_state.gsea_error = "No group selected."
                     elif not gsea_gene_sets or "not installed" in gsea_gene_sets or "Error fetching" in gsea_gene_sets:
                         st.warning("Please select a valid Gene Set Library.")
                         st.session_state.gsea_error = "Invalid or unavailable gene set library selected."
                     else:
                         with st.spinner(f"Running GSEA Prerank for group '{gsea_selected_group}' on '{gsea_gene_sets}'..."):
                             try:
                                 # Get hash of marker df for caching
                                 marker_df = st.session_state.calculated_markers_result_df
                                 marker_df_hash_val = get_adata_hash(marker_df) if marker_df is not None else "no_markers"

                                 # Call analysis function
                                 gsea_results = run_gsea_prerank(
                                     _marker_results_df_ref=marker_df,
                                     marker_df_hash=marker_df_hash_val,
                                     selected_group=gsea_selected_group,
                                     gene_sets=gsea_gene_sets,
                                     ranking_metric=gsea_ranking_metric,
                                     min_size=gsea_min_size,
                                     max_size=gsea_max_size,
                                     permutation_num=gsea_permutations
                                     # Pass other advanced params if needed
                                 )
                                 st.session_state.gsea_results_df = gsea_results
                                 st.session_state.gsea_params_display = { # Store params for context
                                     'group': gsea_selected_group,
                                     'gene_sets': gsea_gene_sets,
                                     'ranking_metric': gsea_ranking_metric,
                                     'min_size': gsea_min_size,
                                     'max_size': gsea_max_size,
                                     'permutations': gsea_permutations
                                 }
                                 st.success(f"GSEA Prerank complete for group '{gsea_selected_group}'. Results below.")
                                 logger.info(f"GSEA analysis successful for group {gsea_selected_group}")

                             except (AnalysisError, ValueError, TypeError) as e:
                                 st.session_state.gsea_error = f"Pathway Analysis Error: {e}"
                                 logger.error(f"GSEA analysis failed: {e}", exc_info=True)
                             except Exception as e:
                                 st.session_state.gsea_error = f"An unexpected error occurred during Pathway Analysis: {e}"
                                 logger.error(f"Unexpected GSEA error: {e}", exc_info=True)
                             finally:
                                st.rerun() # Rerun to display results/errors immediately


            # --- Display GSEA Results ---
            st.markdown("---")
            st.subheader("GSEA Results")

            if st.session_state.get('gsea_error'):
                st.error(st.session_state.gsea_error)

            if st.session_state.get('gsea_results_df') is not None:
                gsea_df = st.session_state.gsea_results_df
                gsea_params = st.session_state.gsea_params_display

                st.write(f"Displaying results for Group: **{gsea_params.get('group', 'N/A')}**, Gene Set: **{gsea_params.get('gene_sets', 'N/A')}**")

                if gsea_df.empty:
                     st.info("No significant pathways found with the current settings.")
                else:
                    st.dataframe(
                        gsea_df,
                        use_container_width=True,
                        # Optional: configure column formats
                         column_config={
                            "NES": st.column_config.NumberColumn("NES", format="%.3f"),
                            "p_val": st.column_config.NumberColumn("p-value", format="%.2E"),
                            "fdr": st.column_config.NumberColumn("FDR q-value", format="%.2E"),
                            "genes": st.column_config.ListColumn("Core Enrichment Genes", width="large")
                        }
                    )
                    # Add download button for GSEA results
                    csv = gsea_df.to_csv(index=False).encode('utf-8')
                    fname = sanitize_filename(f"gsea_prerank_{gsea_params.get('group', 'group')}_{gsea_params.get('gene_sets', 'geneset')}.csv")
                    st.download_button(
                        label="Download GSEA Results as CSV",
                        data=csv,
                        file_name=fname,
                        mime='text/csv',
                    )
            else:
                st.write("Run Pathway Analysis using the form above to see results.")
    # Pseudobulk PCA Calculation Form (remains in its tab)
    with tab_pca:
        st.subheader("Pseudobulk Principal Component Analysis (PCA)")
        st.markdown("Aggregate data by selected groups, then perform PCA on the pseudobulk samples.")
        
        if "pca_form_submitted" in st.session_state and st.session_state.pca_form_submitted:
            st.session_state.active_tab = "🔬 Pseudobulk PCA"
            st.session_state.pca_form_submitted = False  # Reset the flag

        with st.form("pca_form"):
            # Grouping Vars (allow multiple)
            pca_grouping_vars = st.multiselect(
                "Group By for Aggregation:",
                options=valid_obs_cat_cols,
                key="pca_group_select_form",
                help="Select one or more categorical variables to define the pseudobulk samples."
            )

            # Data Source Selection
            pca_layer_key = st.selectbox(
                "Data Source for Aggregation:",
                options=dynamic_layer_options, # Use combined list
                index=0, # Default to 'Auto-Select'
                key="pca_layer_select_form",
                help="Select the data matrix/layer to aggregate (typically 'sum' or 'mean') for PCA."
            )

            # Aggregation Function (optional, could default to sum or mean)
            # pca_agg_func = st.selectbox("Aggregation Function:", options=['sum', 'mean'], index=0, key="pca_agg_func_form")
            pca_agg_func = 'sum' # Defaulting to sum for count-based PCA

            # Advanced Options Expander (inside the form)
            st.checkbox("Show Advanced PCA Options", key="show_advanced_pca") # Use state key
            n_hvgs_pca = N_HVG_PSEUDOBULK # Get defaults
            n_comps_pca = DEFAULT_PCA_COMPONENTS
            if st.session_state.show_advanced_pca:
                with st.expander("Advanced Options"):
                    # Ensure max_value for HVGs doesn't exceed available vars
                    max_hvgs = adata_vis.n_vars if adata_vis else 10000 # Safe default if adata_vis not loaded
                    n_hvgs_pca = st.number_input("Number of HVGs:", min_value=10, max_value=max_hvgs, value=min(N_HVG_PSEUDOBULK, max_hvgs), step=100, key="pca_hvg_n_form")
                    n_comps_pca = st.number_input("Max PCA Components:", min_value=1, max_value=100, value=DEFAULT_PCA_COMPONENTS, key="pca_comps_n_form")

            # Submit Button
            run_pca_button = st.form_submit_button("Run Pseudobulk PCA")

            if run_pca_button:
                # Set active tab and update query parameter immediately
                st.session_state.active_tab = "🔬 Pseudobulk PCA"
                st.query_params["tab"] = "🔬 Pseudobulk PCA"

                st.session_state.pca_adata_result = None # Clear previous results
                st.session_state.pca_error = None
                st.session_state.pca_grouping_vars_used = [] # Clear previous grouping vars
                if not pca_grouping_vars:
                    st.warning("Please select at least one grouping variable for PCA aggregation.")
                    st.session_state.pca_error = "No grouping variables selected."
                else:
                    try:
                        # Step 1: Aggregation (Cached)
                        logger.info("Requesting Aggregation for PCA...")
                        grouping_tuple = tuple(sorted(pca_grouping_vars)) # Use sorted tuple for cache key
                        adata_agg_pca = cached_aggregate_adata(
                            _adata_ref =adata_vis, # Use the visualization data
                            _adata_ref_hash=get_adata_hash(adata_vis), # Pass hash for cache invalidation
                            grouping_vars_tuple=grouping_tuple,
                            selected_layer_key=pca_layer_key,
                            agg_func=pca_agg_func # Use selected agg func
                        )
                        logger.info(f"Aggregation for PCA complete. Shape: {adata_agg_pca.shape}")

                        # Step 2: PCA (using the function from pca_analysis.py)
                        with st.spinner(f"Running PCA (HVGs={n_hvgs_pca}, Comps={n_comps_pca})..."):
                            # Pass the result of aggregation to the PCA function
                            adata_pca_result = preprocess_and_run_pca(
                                adata_agg_pca, # Pass the actual aggregated data
                                n_hvgs=n_hvgs_pca,
                                n_comps=n_comps_pca
                            )
                        st.session_state.pca_adata_result = adata_pca_result # Store result in state
                        # Store grouping vars used for this successful run for context in display tab
                        st.session_state.pca_grouping_vars_used = pca_grouping_vars
                        logger.info("PCA calculation complete.")
                        st.success("PCA analysis complete. Results displayed below.") # Add success message

                    except (AggregationError, AnalysisError, FactorNotFoundError, ValueError, TypeError) as e:
                        st.session_state.pca_error = f"Pseudobulk PCA Error: {e}"
                        logger.error(f"Pseudobulk PCA failed: {e}", exc_info=True)
                    except Exception as e:
                        st.session_state.pca_error = f"An unexpected error occurred during Pseudobulk PCA: {e}"
                        logger.error(f"Unexpected Pseudobulk PCA error: {e}", exc_info=True)


    # Pseudobulk DEG Calculation Form (remains in its tab)
    with tab_deg:
        st.subheader("Pseudobulk Differential Expression (pyDESeq2)")
        if not PYDESEQ2_INSTALLED:
            st.warning("`pydeseq2` library not installed. DEG analysis is unavailable. Please install it (`pip install pydeseq2`) and restart.", icon="⚠️")
        else:
            st.markdown("Aggregate counts by selected factors, then run DESeq2 to find DE genes between two complex groups.")
            with st.form("deg_form"):
                st.markdown("**1. Define Factors & Data Source**")
                deg_comparison_factors = st.multiselect(
                    "Comparison Factors:", options=valid_obs_cat_cols, key="deg_comparison_select_form",
                    help="Select variables defining the comparison groups (e.g., [treatment, cell_type]). Order matters for group definition."
                )
                deg_replicate_factor = st.selectbox(
                    "Replicate Factor (Optional):", options=[None] + valid_obs_cat_cols, key="deg_replicate_select_form",
                    help="Select variable identifying replicates (e.g., patient_id)."
                )
                # Automatically suggest 'counts' layer if it exists
                default_deg_layer_idx = 0
                if 'counts' in dynamic_layer_options: default_deg_layer_idx = dynamic_layer_options.index('counts')
                elif 'Auto-Select' in dynamic_layer_options: default_deg_layer_idx = dynamic_layer_options.index('Auto-Select')

                deg_layer_key = st.selectbox(
                    "Data Source for Aggregation (Counts):", options=dynamic_layer_options, index=default_deg_layer_idx, key="deg_layer_select_form",
                    help="Select matrix/layer with raw counts for DESeq2 (aggregation by sum)."
                )

                st.markdown("**2. Define Comparison Groups**")
                group1_levels = OrderedDict()
                group2_levels = OrderedDict()
                deg_form_valid = bool(deg_comparison_factors) # Initial validation

                if deg_comparison_factors:
                    col_g1, col_g2 = st.columns(2)
                    # Dynamically create selectors based on chosen factors
                    with col_g1:
                        st.markdown("**Group 1 (Numerator)**")
                        for factor in deg_comparison_factors:
                            unique_levels = []
                            try:
                                # Ensure factor exists before accessing
                                if factor not in adata_vis.obs.columns:
                                    raise ValueError(f"Factor '{factor}' not found in columns.")
                                unique_levels = adata_vis.obs[factor].astype('category').cat.categories.tolist()
                                if not unique_levels: raise ValueError("No levels found.")
                            except Exception as e:
                                st.warning(f"Could not get levels for factor '{factor}': {e}", icon="⚠️")
                                deg_form_valid = False # Invalidate form if levels can't be retrieved
                            if unique_levels:
                                group1_levels[factor] = st.selectbox(f"Level for '{factor}' (G1):", options=unique_levels, key=f"deg_g1_{factor}")

                    with col_g2:
                        st.markdown("**Group 2 (Denominator / Base)**")
                        for factor in deg_comparison_factors:
                            unique_levels = []
                            try:
                                if factor not in adata_vis.obs.columns:
                                    raise ValueError(f"Factor '{factor}' not found in columns.")
                                unique_levels = adata_vis.obs[factor].astype('category').cat.categories.tolist()
                                if not unique_levels: raise ValueError("No levels found.")
                            except Exception as e:
                                st.warning(f"Could not get levels for factor '{factor}': {e}", icon="⚠️")
                                deg_form_valid = False
                            if unique_levels:
                                # Default second group to second level if available, else first
                                default_idx_g2 = 1 if len(unique_levels) > 1 else 0
                                group2_levels[factor] = st.selectbox(f"Level for '{factor}' (G2):", options=unique_levels, key=f"deg_g2_{factor}", index=default_idx_g2)

                    # Check if groups are identical only if form is still valid
                    is_identical = False
                    if deg_form_valid and len(group1_levels) == len(deg_comparison_factors) and len(group2_levels) == len(deg_comparison_factors):
                        is_identical = (group1_levels == group2_levels)
                        if is_identical:
                            st.warning("Group 1 and Group 2 definitions are identical.", icon="⚠️")
                            deg_form_valid = False
                else:
                    st.info("Select Comparison Factors above to define groups.")
                    deg_form_valid = False


                st.markdown("**3. Run Analysis**")
                st.checkbox("Show Advanced DEG Options", key="show_advanced_deg") # State key
                min_sum_filter = MIN_DESEQ_COUNT_SUM # Default
                if st.session_state.show_advanced_deg:
                    with st.expander("Advanced Options"):
                        min_sum_filter = st.number_input("Min Gene Count Sum Filter:", min_value=0, value=MIN_DESEQ_COUNT_SUM, step=5, key="deg_min_count_form", help="Filter genes with total count across pseudobulk samples < this value.")

                # Enable button only if factors selected, levels retrieved, and groups different
                run_deg_button = st.form_submit_button("Run Pseudobulk DEG", disabled=not deg_form_valid)

                if run_deg_button:
                    st.session_state.active_tab = "🧪 Pseudobulk DEG"
                    st.session_state.deg_results_df = None # Clear previous results
                    st.session_state.deg_error = None
                    st.session_state.deg_params_display = None # Clear previous params

                    try:
                        # Step 1: Aggregation (Sum - Cached)
                        deg_aggregation_keys = sorted(list(set(deg_comparison_factors + ([deg_replicate_factor] if deg_replicate_factor else []))))
                        logger.info(f"Requesting Aggregation for DEG by keys: {deg_aggregation_keys}")
                        agg_keys_tuple = tuple(deg_aggregation_keys)
                        adata_agg_deg = cached_aggregate_adata(
                            _adata_ref=adata_vis,
                            _adata_ref_hash=adata_vis_hash, # Pass hash for cache invalidation
                            grouping_vars_tuple=agg_keys_tuple,
                            selected_layer_key=deg_layer_key,
                            agg_func='sum' # MUST be sum for DESeq2 counts
                        )
                        logger.info(f"Aggregation for DEG complete. Shape: {adata_agg_deg.shape}")


                        # Step 2: Run DESeq2 (using function from deg_analysis.py)
                        with st.spinner("Running pyDESeq2..."):
                            deg_results_df = prepare_metadata_and_run_deseq(
                                adata_agg_deg, # Pass the actual aggregated data
                                comparison_factors=deg_comparison_factors, # Pass as list
                                replicate_factor=deg_replicate_factor,
                                group1_levels=group1_levels, # Pass OrderedDict
                                group2_levels=group2_levels, # Pass OrderedDict
                                min_count_sum_filter=min_sum_filter,
                                min_nonzero_samples=min_nonzero_samples
                            )
                        st.session_state.deg_results_df = deg_results_df # Store result
                        # Store params used for this successful run for display context
                        st.session_state.deg_params_display = {'group1': group1_levels, 'group2': group2_levels}
                        logger.info(f"pyDESeq2 analysis complete. Found {len(deg_results_df) if deg_results_df is not None else 0} genes.")
                        st.success("DEG analysis complete. Results displayed below.") # Add success message

                    except (AggregationError, AnalysisError, FactorNotFoundError, ValueError, ImportError, TypeError) as e:
                        st.session_state.deg_error = f"Pseudobulk DEG Error: {e}"
                        logger.error(f"Pseudobulk DEG failed: {e}", exc_info=True)
                    except Exception as e:
                        st.session_state.deg_error = f"An unexpected error occurred during Pseudobulk DEG: {e}"
                        logger.error(f"Unexpected Pseudobulk DEG error: {e}", exc_info=True)


    # --- Render Tabs using Imported Functions ---
    with tab_summary: # New Tab Rendering
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

    with tab_pca: # Pseudobulk PCA Display (Form is also here)
        pseudobulk_pca_tab.render_pseudobulk_pca_tab(
            pca_result=st.session_state.get('pca_adata_result'),
            # Pass grouping vars used in the last run (if successful) for context
            pca_grouping_vars=st.session_state.get('pca_grouping_vars_used', [])
        )

    with tab_deg: # Pseudobulk DEG Display (Form is also here)
        if PYDESEQ2_INSTALLED:
            # Corrected State Retrieval for DEG Params
            stored_params = st.session_state.get('deg_params_display')
            default_params = {'group1': OrderedDict(), 'group2': OrderedDict()}
            deg_params = stored_params if stored_params is not None else default_params

            pseudobulk_deg_tab.render_pseudobulk_deg_tab(
                deg_results=st.session_state.get('deg_results_df'),
                group1_levels=deg_params['group1'], # Now deg_params is guaranteed to be a dict
                group2_levels=deg_params['group2']
            )
        # Else: the warning is displayed within the form area above


# --- Footer or Initial Prompt ---
elif not load_button and st.session_state.adata_full is None and not st.session_state.load_error_message:
    st.info("⬅️ Upload an H5AD file or provide a valid file path (e.g., using `--h5ad` argument) to begin.")

# Add a final check for the case where adata_vis failed to initialize properly
elif st.session_state.load_error_message and st.session_state.adata_vis is None:
    st.error(f"Could not initialize data. Error: {st.session_state.load_error_message}")