# navbar/tabs/qc_tab.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import logging
import re # Import regular expressions for pattern matching
import seaborn as sns # Import seaborn for violin plots

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
    
    # --- Basic QC Metrics Violin Plots ---
    st.markdown("---")
    st.subheader("Basic QC Metrics") # Added chart and magnifying glass emojis

    obs_df = adata_vis.obs if hasattr(adata_vis, 'obs') else pd.DataFrame()

    if obs_df.empty:
        st.info("`adata.obs` is empty. Cannot generate QC metrics plots.")
        return

    # --- Robust QC metric selection ---
    raw_metric_keys = ["total_counts_raw", "n_genes_by_counts_raw", "pct_counts_mt_raw"]
    nonraw_metric_keys = ["total_counts", "n_genes_by_counts", "pct_counts_mt"]

    # Check if all raw columns are present
    if all(col in obs_df.columns for col in raw_metric_keys):
        default_qc_metrics = {
            "total_counts_raw": "Number of UMI detected",
            "n_genes_by_counts_raw": "Number of genes detected",
            "pct_counts_mt_raw": "% Mitochondrial reads"
        }
    else:
        default_qc_metrics = {
            "total_counts": "Number of UMI detected",
            "n_genes_by_counts": "Number of genes detected",
            "pct_counts_mt": "% Mitochondrial reads"
        }

    # Optionally add doublet_score if present
    if "doublet_score" in obs_df.columns:
        default_qc_metrics["doublet_score"] = "% Doublets"

    # Filter for metrics actually present in adata.obs
    available_metrics = {k: v for k, v in default_qc_metrics.items() if k in obs_df.columns}

    if not available_metrics:
        st.info("No default QC metrics (UMI, genes, mitochondrial reads, doublets) found in `adata.obs`.")


    # Get all available observation columns for grouping
    potential_grouping_cols = []
    for col in obs_df.columns:
        if obs_df[col].dtype in ['category', 'object', 'bool']:
            # Ensure there's at least one non-NA value to count unique values meaningfully
            if obs_df[col].notna().any():
                if obs_df[col].nunique() < 50: # Keep the limit for practical plotting
                    potential_grouping_cols.append(col)
    
    grouping_variables = ['None'] + sorted(potential_grouping_cols) # Sort for consistent order

    selected_grouping_var = st.selectbox(
        "Select grouping variable (optional):",
        grouping_variables,
        index=0, # 'None' selected by default
        key="qc_grouping_var"
    )
    if selected_grouping_var == 'None':
        st.markdown("##### Overall Distribution")
        
        # Max number of columns for plots. We'll adjust the figure size.
        num_plots_to_show = len(available_metrics)
        # Create columns - using the number of available metrics or a max of 4
        cols = st.columns(min(num_plots_to_show, 4)) # Ensure we don't try to make more than 4 columns

        plot_index = 0
        for metric_col, metric_label in available_metrics.items():
            # Place each plot within its own column
            with cols[plot_index % len(cols)]: # Cycle through columns if more than 4 plots
                # Use an expander for each plot
                with st.expander(f"**{metric_label}**", expanded=True):
                    fig_violin, ax_violin = plt.subplots(figsize=(6, 4)) # Adjusted for smaller display in columns
                    try:
                        sns.violinplot(y=obs_df[metric_col], ax=ax_violin, inner="quartile", color="skyblue")
                        ax_violin.set_title(metric_label) # Title inside expander is enough
                        ax_violin.set_ylabel(metric_label)
                        st.pyplot(fig_violin)

                        # Download button directly below the plot inside the expander
                        try:
                            fname_base = sanitize_filename(f"qc_violin_{metric_col}")
                            fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                            img_bytes = get_figure_bytes(fig_violin, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                            generate_download_button(img_bytes, filename=fname, label=f"Download Plot ({DEFAULT_PLOT_FORMAT.upper()})", mime=f"image/{DEFAULT_PLOT_FORMAT}", key=f"download_violin_{metric_col}")
                        except PlottingError as pe:
                            st.error(f"Error preparing {metric_label} download: {pe}")
                        except Exception as dle:
                            st.error(f"Unexpected error during {metric_label} download: {dle}")
                            logger.error(f"{metric_label} download error: {dle}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error generating violin plot for {metric_col}: {e}", exc_info=True)
                        st.error(f"Could not generate violin plot for {metric_label}.")
                    finally:
                        plt.close(fig_violin)
            plot_index += 1
    else: 
        st.markdown(f"##### Distribution by `{selected_grouping_var}`")
        if selected_grouping_var not in obs_df.columns or obs_df[selected_grouping_var].isnull().all():
            st.warning(f"Selected grouping variable '{selected_grouping_var}' not found or contains only missing values.")
        else:
            for metric_col, metric_label in available_metrics.items():
                with st.expander(f"**{metric_label} by {selected_grouping_var}**", expanded=True):
                    fig_violin, ax_violin = plt.subplots(figsize=(10, 6))
                    try:
                        # Filter out NaN values in the grouping column for plotting
                        plot_data = obs_df.dropna(subset=[selected_grouping_var])
                        if plot_data.empty:
                            st.info(f"No data to plot for {metric_label} when grouped by '{selected_grouping_var}' (after dropping NaNs).")
                            plt.close(fig_violin)
                            continue

                        # Sort unique categories naturally for better visualization
                        unique_categories = plot_data[selected_grouping_var].unique()
                        sorted_categories = sorted(unique_categories, key=natural_sort_key)
                        
                        # Ensure the order is maintained in the plot
                        plot_data[selected_grouping_var] = pd.Categorical(plot_data[selected_grouping_var], categories=sorted_categories, ordered=True)

                        sns.violinplot(
                            x=selected_grouping_var,
                            y=metric_col,
                            data=plot_data,
                            ax=ax_violin,
                            inner="quartile",
                            palette="viridis"
                        )
                        ax_violin.set_title(f"{metric_label} by {selected_grouping_var}")
                        ax_violin.set_xlabel(selected_grouping_var)
                        ax_violin.set_ylabel(metric_label)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig_violin)

                        # Download button
                        try:
                            fname_base = sanitize_filename(f"qc_violin_{metric_col}_by_{selected_grouping_var}")
                            fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                            img_bytes = get_figure_bytes(fig_violin, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                            generate_download_button(img_bytes, filename=fname, label=f"Download Plot ({DEFAULT_PLOT_FORMAT.upper()})", mime=f"image/{DEFAULT_PLOT_FORMAT}", key=f"download_violin_{metric_col}_by_{selected_grouping_var}")
                        except PlottingError as pe:
                            st.error(f"Error preparing {metric_label} download: {pe}")
                        except Exception as dle:
                            st.error(f"Unexpected error during {metric_label} download: {dle}")
                            logger.error(f"{metric_label} download error: {dle}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error generating violin plot for {metric_col} grouped by {selected_grouping_var}: {e}", exc_info=True)
                        st.error(f"Could not generate violin plot for {metric_label} grouped by {selected_grouping_var}.")
                    finally:
                        plt.close(fig_violin)
    
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

