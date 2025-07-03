import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import logging
import re # Import regular expressions for pattern matching
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI
from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
from tabs.heatmap_utils import create_plate_heatmap_fig, natural_sort_key

logger = logging.getLogger(__name__)

def render_qc_tab(adata_vis):
    # --- Cell Count Distribution (Knee Plot) ---
    st.subheader("Cell Count Distribution (Knee Plot)")
    data_source_msg = ""
    counts_per_cell = None
    fig_knee = None
    try:
        count_matrix = None
        if 'counts' in getattr(adata_vis, 'layers', {}) and adata_vis.layers['counts'].shape[0] == adata_vis.n_obs:
            layer_data = adata_vis.layers['counts']
            if hasattr(layer_data, 'dtype') and np.issubdtype(layer_data.dtype, np.number): 
                count_matrix = layer_data; data_source_msg = "`adata.layers['counts']`"
                logger.info("Using layer 'counts' for knee plot.")
            else: 
                logger.warning("Layer 'counts' non-numeric.")
        if count_matrix is None and hasattr(adata_vis, 'X') and adata_vis.X is not None and adata_vis.X.shape[0] == adata_vis.n_obs:
            if hasattr(adata_vis.X, 'dtype') and np.issubdtype(adata_vis.X.dtype, np.number):
                count_matrix = adata_vis.X
                data_source_msg = "`adata.X`"
                logger.info("Using adata.X for knee plot.")
                min_val = count_matrix.min() if count_matrix.size > 0 else 0; is_int = True
                try:
                    if scipy.sparse.issparse(count_matrix) and count_matrix.data.size > 0: 
                        is_int = np.all(np.equal(np.mod(count_matrix.data, 1), 0))
                    elif isinstance(count_matrix, np.ndarray) and count_matrix.size > 0: 
                        is_int = np.all(np.equal(np.mod(count_matrix[count_matrix != 0], 1), 0))
                except Exception as check_e: 
                    logger.warning(f"Int check failed on {data_source_msg}: {check_e}")
                if min_val < 0 or not is_int: 
                    st.caption(f"Note: {data_source_msg} may not be raw counts.")
                    logger.warning(f"Data in {data_source_msg} may not be raw counts.")
            else: 
                logger.warning("adata.X non-numeric.")
        if count_matrix is None: 
            st.info("No suitable count data found for knee plot.")
        else:
            with st.spinner("Calculating counts per cell..."):
                if not np.issubdtype(count_matrix.dtype, np.number): 
                    raise TypeError(f"Count matrix not numeric.")
                counts_per_cell = np.asarray(count_matrix.sum(axis=1)).flatten()
            if counts_per_cell is not None and len(counts_per_cell) > 0:
                counts_per_cell_sorted = np.sort(counts_per_cell[counts_per_cell > 0])[::-1]
                if len(counts_per_cell_sorted) == 0: 
                    st.warning(f"All cells have zero counts ({data_source_msg}).")
                else:
                    ranks = np.arange(1, len(counts_per_cell_sorted) + 1)
                    fig_knee, ax_knee = plt.subplots(figsize=(7, 5))
                    ax_knee.plot(ranks, counts_per_cell_sorted, marker='.', linestyle='-', markersize=2, color='blue')
                    ax_knee.set_xscale('log'); ax_knee.set_yscale('log')
                    ax_knee.set_xlabel("Cell Rank")
                    ax_knee.set_ylabel("Total Counts (log)")
                    ax_knee.set_title("Knee Plot")
                    ax_knee.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')
                    plt.tight_layout()
                    st.pyplot(fig_knee)
                    st.caption(f"Using {data_source_msg}. Zero-count cells excluded.")
                    # Download button
                    try:
                        fname_base = "knee_plot"; fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                        img_bytes = get_figure_bytes(fig_knee, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                        generate_download_button(img_bytes, filename=fname, label=f"Download Knee Plot ({DEFAULT_PLOT_FORMAT.upper()})", mime=f"image/{DEFAULT_PLOT_FORMAT}", key="download_knee_plot")
                    except PlottingError as pe: 
                        st.error(f"Knee plot download error: {pe}")
                    except Exception as dle: 
                        st.error(f"Unexpected knee plot download error: {dle}")
                        logger.error(f"Knee plot download error: {dle}", exc_info=True)
            else: 
                st.warning(f"Could not calculate per-cell counts ({data_source_msg}).")
    except Exception as e: 
        logger.error(f"Error in knee plot section: {e}", exc_info=True)
        st.error(f"Could not generate knee plot.")
    finally:
        if fig_knee: 
            plt.close(fig_knee)
    
    # --- Well Cell Count Heatmaps ---
    st.markdown("---")
    obs_df = adata_vis.obs if hasattr(adata_vis, 'obs') else pd.DataFrame()
    st.subheader("Well Cell Count Heatmaps")
    if 'plate' not in obs_df.columns: 
        st.info("No `plate` column found.")
    elif obs_df.empty: 
        st.info("`adata.obs` is empty.")
    else:
        well_col_pattern = r"^bc\d+_well$"
        possible_well_cols = [col for col in obs_df.columns if re.match(well_col_pattern, col)]
        if not possible_well_cols: 
            st.warning("`plate` column found, but no `bcX_well` columns found.")
        else:
            unique_plates = obs_df['plate'].dropna().unique()
            for plate_id in unique_plates:
                st.markdown(f"---\n### Plate: `{plate_id}`")
                plate_df = obs_df[obs_df['plate'] == plate_id]
                if plate_df.empty:
                    st.info(f"No data for plate `{plate_id}`.")
                    continue
                sorted_well_cols = sorted(possible_well_cols, key=natural_sort_key)
                nice_well_names = [f"barcode {col.split('bc')[1].split('_')[0]}" for col in sorted_well_cols]
                st.caption(f"Generating heatmaps for: {', '.join(nice_well_names)}")
                for well_col in sorted_well_cols:
                    barcode_number = well_col.split('bc')[1].split('_')[0]
                    nice_name = f"barcode {barcode_number}"
                    st.markdown(f"#### Heatmap for {nice_name}")
                    plate_heatmap_fig = None
                    try:
                        with st.spinner(f"Generating heatmap: `{nice_name}`..."):
                            well_counts = plate_df[well_col].astype(str).value_counts()
                            if well_counts.empty: 
                                st.warning(f"`{well_col}`: No data for plate `{plate_id}`.")
                                continue
                            heatmap_title = f"Cell Counts per Well ({nice_name}, Plate {plate_id})"
                            heatmap_label = 'Number of Cells'
                            plate_heatmap_fig = create_plate_heatmap_fig(heatmap_title, heatmap_label, well_counts.to_dict())
                            st.pyplot(plate_heatmap_fig)
                            # Download button
                            try:
                                fname_base = f"plate_{plate_id}_heatmap_barcode_{barcode_number}"
                                fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                                img_bytes = get_figure_bytes(plate_heatmap_fig, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                                generate_download_button(
                                    img_bytes, filename=fname,
                                    label=f"Download Heatmap ({DEFAULT_PLOT_FORMAT.upper()})",
                                    mime=f"image/{DEFAULT_PLOT_FORMAT}",
                                    key=f"download_plate_heatmap_{plate_id}_{well_col}"
                                )
                            except PlottingError as pe: 
                                st.error(f"Error preparing plate heatmap download: {pe}")
                            except Exception as dle: 
                                st.error(f"Unexpected error during plate heatmap download: {dle}")
                                logger.error(f"Plate heatmap download error: {dle}", exc_info=True)
                    except Exception as e: 
                        logger.error(f"Error generating plate heatmap for '{well_col}' in plate '{plate_id}': {e}", exc_info=True)
                        st.error(f"Could not generate heatmap for `{well_col}` in plate `{plate_id}`.")
                    finally:
                        if plate_heatmap_fig: 
                            plt.close(plate_heatmap_fig)

