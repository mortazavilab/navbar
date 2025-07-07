# navbar/app.py

import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
import os
import argparse
import csv
from collections import OrderedDict
import anndata as ad
import requests
from io import BytesIO
import tempfile
import re

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

from analysis.marker_analysis import calculate_rank_genes_df
from analysis.pca_analysis import preprocess_and_run_pca
from analysis.deg_analysis import prepare_metadata_and_run_deseq, PYDESEQ2_INSTALLED
from analysis.gsea_analysis import run_gsea_prerank, GSEAPY_INSTALLED, AVAILABLE_GENE_SETS, RANK_GROUP_COL, RANK_SCORE_COL

from tabs import summary_tab, qc_tab, embedding_tab, gene_expression_tab, marker_genes_tab, pseudobulk_pca_tab, pseudobulk_deg_tab, gsea_tab


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

def is_url(path_or_url):
    return bool(re.match(r"^https?://", path_or_url, re.IGNORECASE))

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

# --- Tab keys (define early for session state) ---
tab_keys = [
    "📄 Summary Info",
    "📋 QC", 
    "🗺️ Embedding Plot",
    "📈 Gene Expression",
    "🔬 Marker Genes",
    "📊 Pathway Analysis",
    "🧬 Pseudobulk PCA",
    "🧪 Pseudobulk DEG"
]

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

st.sidebar.title("🔎 Navbar")
# --- Argument Parsing & Config Splash ---
parsed_args = None
try:
    parser = argparse.ArgumentParser(description="Navbar h5ad Explorer - Argument Parser")
    parser.add_argument("--h5ad", type=str, help="Path to h5ad file to load automatically via command line.")
    parser.add_argument("--config", type=str, help="CSV file with dataID,dataPath entries.")
    parsed_args, _ = parser.parse_known_args()
    config_path = parsed_args.config if parsed_args else None
    args_h5ad_path_cmd = parsed_args.h5ad if parsed_args else None
except Exception as e:
    logger.warning(f"Argparse failed, possibly due to Streamlit context: {e}")
    config_path = None
    args_h5ad_path_cmd = None

# --- If config file provided, parse it ---
config_entries = []
if config_path and os.path.exists(config_path):
    with open(config_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                config_entries.append({'dataID': row[0], 'dataPath': row[1]})

# --- Splash page for dataset selection ---
if config_entries and 'selected_dataID' not in st.session_state:
    st.title("🔎 Navbar")
    st.header("Select a dataset to load")
    data_ids = [entry['dataID'] for entry in config_entries]
    selected_idx = st.radio("Choose a dataset:", options=range(len(data_ids)), format_func=lambda i: data_ids[i], key="dataset_radiobox")
    if st.button("Load Selected Dataset"):
        st.session_state.selected_dataID = data_ids[selected_idx]
        st.session_state.selected_dataPath = config_entries[selected_idx]['dataPath']
        st.rerun()
    st.stop()

# --- Use selected dataset if chosen ---
if config_entries and 'selected_dataID' in st.session_state:
    initial_file_to_load = st.session_state.selected_dataPath
    initial_file_is_url = is_url(initial_file_to_load)
    st.sidebar.info(f"Selected dataset: {st.session_state.selected_dataID}")
    # Optionally, allow user to reset selection
    if st.sidebar.button("Choose a different dataset"):
        del st.session_state.selected_dataID
        del st.session_state.selected_dataPath
        st.session_state.initial_load_attempted = False
        st.rerun()
else:
    # --- Argument Parsing & Initial Load for --h5ad and query param ---
    args_h5ad_path_query = None
    try:
        args_h5ad_path_query = st.query_params.get("h5ad")
        if args_h5ad_path_query:
            logger.info(f"Found 'h5ad' query parameter: {args_h5ad_path_query}")
    except Exception as e:
        logger.debug(f"Could not check query_params: {e}")
        args_h5ad_path_query = None

    args_h5ad_path = args_h5ad_path_cmd or args_h5ad_path_query
    initial_file_to_load = None
    initial_file_is_url = False
    if args_h5ad_path:
        if is_url(args_h5ad_path.strip()):
            initial_file_to_load = args_h5ad_path.strip()
            initial_file_is_url = True
            logger.info(f"Using initial URL: {initial_file_to_load}")
        else:
            initial_file_to_load = os.path.expanduser(args_h5ad_path.strip())
            logger.info(f"Using initial file path: {initial_file_to_load}")
    else:
        logger.info("No initial file path provided via --h5ad argument or h5ad query parameter.")

    # --- Sidebar upload/URL/path logic ---
    file_provided_externally = initial_file_to_load is not None
    if not file_provided_externally:
        st.sidebar.header("1. Load Data")
        uploaded_file = st.sidebar.file_uploader("Upload H5AD File", type=["h5ad"])
        h5ad_url = st.sidebar.text_input("Or enter remote H5AD URL:", value="", key="h5ad_url_input")
        default_path_value = initial_file_to_load if initial_file_to_load and st.session_state.adata_vis is None else ""
        file_path_input = st.sidebar.text_input("Or enter file path:", value=default_path_value)
        load_button = st.sidebar.button("Load AnnData", key="load_data_button")
    else:
        uploaded_file = None
        file_path_input = ""
        h5ad_url = ""
        load_button = False
        st.sidebar.text(f"Using file from command line: {os.path.basename(initial_file_to_load)}")

    if load_button:
        file_source = None
        source_name = "unknown"
        st.session_state.load_error_message = None
        if uploaded_file is not None:
            file_source = uploaded_file.getvalue()
            source_name = uploaded_file.name
            logger.info(f"Load button clicked: Attempting to load from uploaded file: {source_name}")
            _load_and_process_adata(file_source, source_name)
        elif h5ad_url:
            try:
                logger.info(f"Load button clicked: Attempting to download h5ad from URL: {h5ad_url}")
                response = requests.get(h5ad_url, stream=True)
                response.raise_for_status()
                file_source = BytesIO(response.content)
                source_name = os.path.basename(h5ad_url)
                _load_and_process_adata(file_source, source_name)
            except Exception as e:
                st.session_state.load_error_message = f"Failed to download or load h5ad from URL: {e}"
                logger.error(f"Failed to download/load h5ad from URL: {e}", exc_info=True)
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
                    _load_and_process_adata(file_source, source_name)
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
            st.session_state.load_error_message = "Please upload an H5AD file, provide a valid file path, or enter a remote URL."
        if st.session_state.adata_vis is not None and not st.session_state.load_error_message:
            if st.session_state.adata_full is not None:
                st.sidebar.success(f"Loaded: {st.session_state.current_h5ad_source}\n({st.session_state.adata_full.n_obs} obs x {st.session_state.adata_full.n_vars} vars)")
                if st.session_state.adata_vis.n_obs < st.session_state.adata_full.n_obs:
                    st.sidebar.info(f"Using {st.session_state.adata_vis.n_obs} cells for visualization (subsampled from {st.session_state.adata_full.n_obs}).")
                else:
                    st.sidebar.info(f"Using all {st.session_state.adata_vis.n_obs} cells for visualization.")
            else:
                st.sidebar.warning("Data loaded for visualization, but full data reference missing.")

# --- Progress bar for loading (for config/cmd/query param) ---
if 'initial_file_to_load' in locals() and initial_file_to_load and not st.session_state.initial_load_attempted:
    logger.info(f"Attempting initial load from config/cmd: {initial_file_to_load}")
    with st.spinner("Loading data..."):
        progress = st.progress(0)
        try:
            if initial_file_is_url:
                response = requests.get(initial_file_to_load, stream=True)
                total = int(response.headers.get('content-length', 0))
                bytes_io = BytesIO()
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        bytes_io.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            progress.progress(min(downloaded / total, 1.0))
                file_source = BytesIO(bytes_io.getvalue())
                source_name = os.path.basename(initial_file_to_load)
            else:
                file_source = initial_file_to_load
                source_name = os.path.basename(initial_file_to_load)
            load_success = _load_and_process_adata(file_source, source_name)
            progress.progress(1.0)
            if load_success:
                logger.info(f"Initial load successful for {source_name}")
                st.success(f"Automatically loaded: {source_name}")
        except Exception as e:
            st.session_state.load_error_message = f"Failed to download or load h5ad: {e}"
            logger.error(f"Failed to download/load h5ad: {e}", exc_info=True)
            st.session_state.initial_load_attempted = True
        finally:
            progress.empty()

# --- Display loading errors if any ---
if st.session_state.load_error_message:
    st.sidebar.error(st.session_state.load_error_message)

# --- Main Application Area ---
if isinstance(st.session_state.get('adata_vis'), ad.AnnData):
    adata_vis = st.session_state.adata_vis
    adata_full = st.session_state.get('adata_full')
    if not isinstance(adata_full, ad.AnnData):
        logger.warning("Full AnnData object is not available or invalid.")
        adata_full = ad.AnnData()
    adata_vis_hash = get_adata_hash(adata_vis)
    st.header(f"Navigating: `{st.session_state.current_h5ad_source}`")
    st.write(f"Using data: **{adata_vis.n_obs} cells** &times; **{adata_vis.n_vars} genes**")
    if adata_full is not None and hasattr(adata_full, 'n_obs') and adata_vis.n_obs < adata_full.n_obs:
        st.caption(f"(Subsampled from {adata_full.n_obs} cells for visualization and analysis where noted)")

    # --- Dynamically get options from adata_vis ---
    obs_cat_cols, obsm_keys, layer_keys = [], [], []
    valid_obs_cat_cols = [""] 
    valid_obsm_keys = [""]
    dynamic_layer_options_base = AGGREGATION_LAYER_OPTIONS.copy()
    try:
        if hasattr(adata_vis, 'obs') and not adata_vis.obs.empty:
            obs_cols = adata_vis.obs.columns.tolist()
            obs_cat_cols = [
                col for col in obs_cols
                if (pd.api.types.is_categorical_dtype(adata_vis.obs[col]) or pd.api.types.is_object_dtype(adata_vis.obs[col]))
                   and adata_vis.obs[col].notna().any()
            ]
            valid_obs_cat_cols = obs_cat_cols if obs_cat_cols else [""]
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
    dynamic_layer_options = dynamic_layer_options_base + layer_keys

    # --- Sidebar Visualization Options ---
    if 'load_button' in locals() and load_button or initial_file_to_load:
        st.sidebar.header("Visualization Options")
    else:
        st.sidebar.header("2. Visualization Options")
    default_embedding_index = 0
    if DEFAULT_EMBEDDING in valid_obsm_keys:
        try: default_embedding_index = valid_obsm_keys.index(DEFAULT_EMBEDDING)
        except ValueError: pass
    elif 'X_umap' in valid_obsm_keys:
        try: default_embedding_index = valid_obsm_keys.index('X_umap')
        except ValueError: pass
    elif 'X_tsne' in valid_obsm_keys:
        try: default_embedding_index = valid_obsm_keys.index('X_tsne')
        except ValueError: pass
    elif 'X_pca' in valid_obsm_keys:
        try: default_embedding_index = valid_obsm_keys.index('X_pca')
        except ValueError: pass
    st.session_state.sidebar_selection_embedding = st.sidebar.selectbox(
        "Embedding:",
        options=valid_obsm_keys,
        index=default_embedding_index,
        key="embedding_select_sidebar",
        help="Select the precomputed embedding coordinates from `adata.obsm`."
    )
    default_color_index = 0
    if valid_obs_cat_cols and valid_obs_cat_cols != [""]:
        preferred_defaults = ['leiden', 'louvain']
        found_preferred = False
        for preferred_key in preferred_defaults:
            if preferred_key in valid_obs_cat_cols:
                try:
                    default_color_index = valid_obs_cat_cols.index(preferred_key)
                    logger.info(f"Found preferred default color key: '{preferred_key}' at index {default_color_index}")
                    found_preferred = True
                    break
                except ValueError:
                    pass
        if not found_preferred and valid_obs_cat_cols:
            logger.info(f"Preferred default keys ({preferred_defaults}) not found. Defaulting to first categorical column: '{valid_obs_cat_cols[0]}'")
            default_color_index = 0
    st.session_state.sidebar_selection_color = st.sidebar.selectbox(
        "Color Cells By:",
        options=valid_obs_cat_cols,
        index=default_color_index,
        key="color_select_sidebar",
        help="Select categorical observation data from `adata.obs` to color cells."
    )


    # --- Define Tabs ---
    tab_objects = st.tabs(tab_keys)
    if st.session_state.active_tab in tab_keys:
        st.query_params["tab"] = st.session_state.active_tab
    selected_tab_index = tab_keys.index(st.session_state.active_tab) if st.session_state.active_tab in tab_keys else 0
    tab_summary, tab_qc, tab_embedding, tab_gene_expr, tab_markers, tab_gsea, tab_pca, tab_deg = tab_objects

    with tab_summary: 
        summary_tab.render_summary_tab(adata_vis)
    with tab_qc:
        qc_tab.render_qc_tab(adata_vis)
    with tab_embedding:
        embedding_tab.render_embedding_tab(
            adata_vis,
            st.session_state.sidebar_selection_embedding,
            st.session_state.sidebar_selection_color
        )
    with tab_gene_expr:
        gene_expression_tab.render_gene_expression_tab(
            adata_vis,
            st.session_state.sidebar_selection_embedding
        )
    with tab_markers:
        marker_genes_tab.render_marker_genes_tab(
            adata_vis=adata_vis,
            adata_full=adata_full,
            valid_obs_cat_cols=valid_obs_cat_cols,
            selected_color_var=st.session_state.sidebar_selection_color
        )
    with tab_gsea:
        gsea_tab.render_gsea_tab()
    with tab_pca:
        pseudobulk_pca_tab.render_pseudobulk_pca_tab(
            adata_vis=adata_vis,
            valid_obs_cat_cols=valid_obs_cat_cols,
            dynamic_layer_options=dynamic_layer_options
        )
    with tab_deg:
        pseudobulk_deg_tab.render_pseudobulk_deg_tab(
            adata_vis=adata_vis,
            valid_obs_cat_cols=valid_obs_cat_cols,
            dynamic_layer_options=dynamic_layer_options
        )

elif not ('load_button' in locals() and load_button) and st.session_state.adata_full is None and not st.session_state.load_error_message:
    st.info("⬅️ Upload an H5AD file or provide a valid file path (e.g., using `--h5ad` argument) to begin.")

elif st.session_state.load_error_message and st.session_state.adata_vis is None:
    st.error(f"Could not initialize data. Error: {st.session_state.load_error_message}")

