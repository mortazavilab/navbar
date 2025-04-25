# scanpy_viewer/tabs/summary_tab.py
import streamlit as st
import pandas as pd
import anndata as ad
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse # Import sparse
import seaborn as sns # Import seaborn for heatmap AND barplot
import re # Import regular expressions for pattern matching
import datetime # Import datetime for time-dependent info

# Use absolute imports from the assumed package root 'scanpy_viewer'
# Adjust the paths if your structure differs
try:
    from scanpy_viewer.utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
    from scanpy_viewer.config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI
except ImportError:
    # Fallback for running script directly or if structure is different
    from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
    from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI


logger = logging.getLogger(__name__)

# --- Helper function to parse well IDs ---
def parse_well_id(well_id):
    if not isinstance(well_id, str) or not re.match(r"^[A-H]([1-9]|1[0-2])$", well_id, re.IGNORECASE): 
        return None, None
    row_char = well_id[0].upper()
    col_str = well_id[1:]
    row_idx = ord(row_char) - ord('A')
    col_idx = int(col_str) - 1
    return row_idx, col_idx

# --- Helper function for natural sorting ---
def natural_sort_key(s):
    match = re.search(r'bc(\d+)_well', s)
    return int(match.group(1)) if match else 999999

# --- Function to create the heatmap data grid ---
def create_heatmap_grid(well_counts):
    heatmap_data = pd.DataFrame(np.nan, index=list('ABCDEFGH'), columns=range(1, 13))
    valid_wells_found = 0; invalid_wells = set()
    for well_id, count in well_counts.items():
        row_idx, col_idx = parse_well_id(well_id)
        if row_idx is not None and col_idx is not None:
            row_char = chr(ord('A') + row_idx)
            col_num = col_idx + 1
            heatmap_data.loc[row_char, col_num] = count
            valid_wells_found += 1
        elif well_id not in invalid_wells:
            if isinstance(well_id, str) and well_id.lower() != 'nan': 
                logger.warning(f"Skipping invalid well ID: '{well_id}'")
                invalid_wells.add(well_id)
            elif not isinstance(well_id, str): 
                logger.warning(f"Skipping non-string well ID: {well_id}") 
                invalid_wells.add(str(well_id))
    if valid_wells_found == 0:
        if invalid_wells: 
            logger.error("No valid 96-well IDs found.") 
        else: 
            logger.warning("No valid well counts found.")
        return None
    return heatmap_data
# --- End Helper Functions ---


def render_summary_tab(adata_vis):
    """Renders the dataset summary information tab."""
    # Get current time for potential future use (as requested by context)
    current_time = datetime.datetime.now()
    logger.debug(f"Rendering summary tab at {current_time}")

    if not isinstance(adata_vis, ad.AnnData):
        st.error("Summary Tab: Invalid AnnData object received.")
        logger.error("Summary Tab received an object that is not an AnnData instance.")
        return

    fig_summary_bar = None
    fig_knee = None
    plate_heatmap_fig = None

    try:
        st.subheader("Dataset Overview")
        st.write(f"**Observations (Cells):** `{adata_vis.n_obs:,}`")
        st.write(f"**Variables (Genes/Features):** `{adata_vis.n_vars:,}`")

        # --- Display Available Metadata ---
        st.markdown("---")
        st.subheader("Available Metadata & Data Slots")
        col1, col2 = st.columns(2)
        obs_df = adata_vis.obs if hasattr(adata_vis, 'obs') else pd.DataFrame()
        with col1:
            with st.expander("**Observation Columns (`adata.obs`)**", expanded=False):
                if not obs_df.empty: 
                    st.dataframe(pd.DataFrame({'Column Name': obs_df.columns, 'Data Type': obs_df.dtypes.astype(str)}), use_container_width=True, height=250, hide_index=True)
                else: 
                    st.write("No `adata.obs`.")
            with st.expander("**Embeddings (`adata.obsm`)**", expanded=False):
                 if hasattr(adata_vis, 'obsm') and adata_vis.obsm: 
                    st.json({k: f"Shape: {v.shape}" if hasattr(v, 'shape') else f"Type: {type(v).__name__}" for k, v in adata_vis.obsm.items()}, expanded=False)
                 else: 
                    st.write("No `adata.obsm`.")
        with col2:
            with st.expander("**Variable Columns (`adata.var`)**", expanded=False):
                 if hasattr(adata_vis, 'var') and not adata_vis.var.empty: 
                    st.dataframe(pd.DataFrame({'Column Name': adata_vis.var.columns, 'Data Type': adata_vis.var.dtypes.astype(str)}), use_container_width=True, height=250, hide_index=True)
                 else: 
                    st.write("No `adata.var`.")
            with st.expander("**Layers (`adata.layers`)**", expanded=False):
                   if hasattr(adata_vis, 'layers') and adata_vis.layers: 
                    st.json({k: f"Shape: {v.shape}, Dtype: {v.dtype}" if hasattr(v, 'shape') and hasattr(v, 'dtype') else f"Type: {type(v).__name__}" for k, v in adata_vis.layers.items()}, expanded=False)
                   else: 
                    st.write("No `adata.layers`.")

        # --- Process Categorical Columns ---
        single_value_attrs = {}
        multi_value_cat_cols = []
        if not obs_df.empty:
            potential_cat_cols = [col for col in obs_df.select_dtypes(include=['category', 'object', 'bool']).columns if obs_df[col].notna().any()]
            for col in potential_cat_cols:
                try:
                    col_data_non_na = obs_df[col].dropna()
                    nunique = col_data_non_na.nunique()
                    if col_data_non_na.empty: 
                        continue
                    if nunique == 1: 
                        single_value_attrs[col] = str(col_data_non_na.unique()[0])
                    elif nunique > 1: 
                        multi_value_cat_cols.append(col)
                except Exception as e: 
                    logger.warning(f"Could not process column '{col}': {e}", exc_info=True)

        # --- Constant Metadata Attributes ---
        st.markdown("---")
        st.subheader("Constant Metadata Attributes")
        st.caption("Categorical `adata.obs` columns with only one unique non-NA value.")
        if single_value_attrs: 
            st.dataframe(pd.DataFrame(single_value_attrs.items(), columns=['Attribute', 'Constant Value']), use_container_width=True, hide_index=True)
        else: 
            st.info("No constant categorical/object/boolean attributes found.")


        # --- Variable Categorical Observation Summary ---
        st.markdown("---"); st.subheader("Variable Categorical Observation Summary")

        if not multi_value_cat_cols:
            st.info("No variable categorical/object/boolean columns found for summary.")
        else:
            # --- Setup selections ---
            sorted_options = sorted(multi_value_cat_cols)
            options_list = [None] + sorted_options
            default_index_primary = 0
            primary_default_col_name = 'leiden' # Desired default primary column
            if primary_default_col_name in multi_value_cat_cols:
                try: 
                    default_index_primary = options_list.index(primary_default_col_name)
                except ValueError: 
                    default_index_primary = 0

            selected_col_primary = st.selectbox(
                "Select Primary Category (Bars):",
                options=options_list, index=default_index_primary,
                format_func=lambda x: "Select primary category..." if x is None else x,
                key="summary_cat_select_primary"
            )

            selected_col_color = None
            display_mode = 'Counts' # Default display mode

            if selected_col_primary:
                # Options for color category (exclude primary, add None)
                color_options = [None] + [opt for opt in sorted_options if opt != selected_col_primary]
                selected_col_color = st.selectbox(
                    "Select Secondary Category (Color / Stack):",
                    options=color_options, index=0, # Default to None
                    format_func=lambda x: "Single category (no stacking)" if x is None else x,
                    key="summary_cat_select_color"
                )

                if selected_col_color: # Only show if coloring category is selected
                    display_mode = st.radio(
                        "Display Mode:", ('Counts', 'Percentage (%)'), key="summary_display_mode", horizontal=True
                    )

                # --- Data Preparation & Plotting ---
                fig_summary_bar = None # Initialize figure for this plot
                try:
                    col_data_primary = obs_df[selected_col_primary].dropna()
                    nunique_primary = col_data_primary.nunique()
                    st.write(f"Primary category **`{selected_col_primary}`** has **{nunique_primary}** unique non-NA value(s).")

                    if nunique_primary == 0:
                        st.info(f"Primary category `{selected_col_primary}` has no non-NA values.")
                    else:
                        # Decide on Top N if too many primary categories
                        num_primary_cats_to_show = 20 if nunique_primary >= 100 else nunique_primary
                        is_top_n_primary = nunique_primary >= 100

                        if is_top_n_primary:
                             st.info(f"Primary category has {nunique_primary} values. Displaying Top {num_primary_cats_to_show}.")

                        # --- Data Aggregation ---
                        with st.spinner("Aggregating data..."):
                            if selected_col_color:
                                # --- Stacked Bar Chart Data ---
                                logger.debug(f"Creating stacked bar chart: Primary='{selected_col_primary}', Color='{selected_col_color}', Mode='{display_mode}'")
                                cols_to_group = [selected_col_primary, selected_col_color]
                                df_filtered = obs_df[cols_to_group].dropna()
                                if df_filtered.empty:
                                    st.warning("No overlapping non-NA data found for the selected primary and secondary categories.")
                                else:
                                    # Group by both, count, and unstack
                                    counts_stacked = df_filtered.groupby(cols_to_group).size().unstack(fill_value=0)

                                    # Sort primary categories (index) by total counts descending
                                    total_counts_primary = counts_stacked.sum(axis=1).sort_values(ascending=False)

                                    # Filter for Top N primary categories
                                    cats_to_keep = total_counts_primary.head(num_primary_cats_to_show).index
                                    counts_stacked_sorted = counts_stacked.loc[cats_to_keep]

                                    # Check color category cardinality (optional warning)
                                    nunique_color = obs_df[selected_col_color].nunique()
                                    if nunique_color > 20:
                                        st.caption(f"Warning: Secondary category `{selected_col_color}` has {nunique_color} unique values, which may make the plot legend cluttered.")

                                    # Prepare data for plotting (Counts or Percentage)
                                    if display_mode == 'Percentage (%)':
                                        data_to_plot = counts_stacked_sorted.apply(lambda x: (x / x.sum()) * 100, axis=1).fillna(0) # Calculate row-wise percentage
                                        y_label = "Percentage (%)"
                                    else: # Counts
                                        data_to_plot = counts_stacked_sorted
                                        y_label = "Count"

                                    # --- Plotting Stacked Bars ---
                                    if not data_to_plot.empty:
                                        plot_title = f"{y_label} of `{selected_col_color}` within `{selected_col_primary}`"
                                        if is_top_n_primary: plot_title += f" (Top {num_primary_cats_to_show} Primary)"

                                        fig_summary_bar, ax = plt.subplots(figsize=(12, 7)) # Wider for legend
                                        # Use pandas plot for easy stacking
                                        data_to_plot.plot(kind='bar', stacked=True, ax=ax, colormap='tab20') # Use a colormap suitable for many categories

                                        ax.set_title(plot_title)
                                        ax.set_xlabel(selected_col_primary)
                                        ax.set_ylabel(y_label)
                                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                                        # Place legend outside the plot
                                        ax.legend(title=selected_col_color, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
                                        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right margin for legend
                                        st.pyplot(fig_summary_bar)
                                    else: 
                                        st.warning("No data to plot after filtering.")

                            else:
                                # --- Simple Bar Chart Data (No secondary category selected) ---
                                logger.debug(f"Creating simple bar chart: Primary='{selected_col_primary}'")
                                value_counts_sr = col_data_primary.value_counts().sort_values(ascending=False)
                                data_to_plot = value_counts_sr.head(num_primary_cats_to_show)
                                y_label = "Count"
                                plot_title = f"Value Counts for `{selected_col_primary}`"
                                if is_top_n_primary: 
                                    plot_title += f" (Top {num_primary_cats_to_show})"

                                # --- Plotting Simple Bars ---
                                if not data_to_plot.empty:
                                     fig_summary_bar, ax = plt.subplots(figsize=(10, 6))
                                     x_labels = data_to_plot.index.astype(str)
                                     y_values = data_to_plot.values
                                     sns.barplot(x=x_labels, y=y_values, ax=ax, palette="Blues_d")
                                     ax.set_title(plot_title); ax.set_ylabel(y_label)
                                     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                                     plt.tight_layout()
                                     st.pyplot(fig_summary_bar)
                                else: 
                                    st.warning("No data to plot.")

                        # --- Download Button (Common for both plot types) ---
                        if fig_summary_bar:
                            try:
                                fname_suffix = f"_vs_{selected_col_color}" if selected_col_color else ""
                                fname_mode = f"_{display_mode.split(' ')[0].lower()}" if selected_col_color else ""
                                fname_topn = f"_top{num_primary_cats_to_show}" if is_top_n_primary else ""
                                fname_base = f"summary_{selected_col_primary}{fname_suffix}{fname_mode}{fname_topn}"
                                fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                                img_bytes = get_figure_bytes(fig_summary_bar, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                                generate_download_button(img_bytes, filename=fname, label=f"Download Chart ({DEFAULT_PLOT_FORMAT.upper()})", mime=f"image/{DEFAULT_PLOT_FORMAT}", key=f"download_summary_chart_{selected_col_primary}_{selected_col_color}_{display_mode}")
                            except PlottingError as pe: 
                                st.error(f"Error preparing summary chart download: {pe}")
                            except Exception as dle: 
                                st.error(f"Unexpected error during summary chart download: {dle}")
                                logger.error(f"Summary chart download error: {dle}", exc_info=True)

                except Exception as e:
                    logger.error(f"Error generating summary plot section for '{selected_col_primary}': {e}", exc_info=True)
                    st.error(f"Could not generate summary plot for `{selected_col_primary}`.")
                finally:
                     if fig_summary_bar: 
                        plt.close(fig_summary_bar) # Close figure specific to this section
            else:
                st.info("Select a primary category from the dropdown above to see its value counts.")


        # --- 96-Well Plate Cell Count Heatmaps ---
        st.markdown("---")
        st.subheader("Well Cell Count Heatmaps")
        if 'plate' not in obs_df.columns: 
            st.info("No `plate` column found.")
        elif obs_df.empty: 
            st.info("`adata.obs` is empty.")
        else:
            well_col_pattern = r"^bc\d+_well$"; possible_well_cols = [col for col in obs_df.columns if re.match(well_col_pattern, col)]
            if not possible_well_cols: 
                st.warning("`plate` column found, but no `bcX_well` columns found.")
            else:
                sorted_well_cols = sorted(possible_well_cols, key=natural_sort_key)
                nice_well_names = [f"barcode {col.split('bc')[1].split('_')[0]}" for col in sorted_well_cols]
                st.caption(f"Generating heatmaps for: {', '.join(nice_well_names)}")
                for well_col in sorted_well_cols:
                    barcode_number = well_col.split('bc')[1].split('_')[0]
                    nice_name = f"barcode {barcode_number}"
                    st.markdown(f"--- \n #### Heatmap for {nice_name}")
                    plate_heatmap_fig = None
                    try:
                        with st.spinner(f"Generating heatmap: `{nice_name}`..."):
                            well_counts = obs_df[well_col].astype(str).value_counts()
                            if well_counts.empty: 
                                st.warning(f"`{well_col}`: No data.")
                                continue
                            heatmap_data = create_heatmap_grid(well_counts)
                            if heatmap_data is None: 
                                st.error(f"`{well_col}`: Failed to create grid.")
                                continue
                            plate_heatmap_fig, ax = plt.subplots(figsize=(10, 6.5))
                            mask = heatmap_data.isnull()
                            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="Reds", mask=mask, linewidths=0, linecolor='none', cbar=True, cbar_kws={'label': 'Number of Cells'}, square=False, ax=ax)
                            ax.set_title(f"Cell Counts per Well ({nice_name})", pad=30)
                            ax.set_xlabel("Plate Column")
                            ax.set_ylabel("Plate Row")
                            ax.xaxis.tick_top()
                            ax.xaxis.set_label_position('top')
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=0); 
                            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                            ax.tick_params(axis='both', which='major', length=0)
                            ax.grid(False)
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                            st.pyplot(plate_heatmap_fig)
                            # Download button
                            try:
                                fname_base = f"plate_heatmap_barcode_{barcode_number}"
                                fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                                img_bytes = get_figure_bytes(plate_heatmap_fig, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                                generate_download_button(img_bytes, filename=fname, label=f"Download Heatmap ({DEFAULT_PLOT_FORMAT.upper()})", mime=f"image/{DEFAULT_PLOT_FORMAT}", key=f"download_plate_heatmap_{well_col}")
                            except PlottingError as pe: 
                                st.error(f"Error preparing plate heatmap download: {pe}")
                            except Exception as dle: 
                                st.error(f"Unexpected error during plate heatmap download: {dle}")
                                logger.error(f"Plate heatmap download error: {dle}", exc_info=True)
                    except Exception as e: 
                        logger.error(f"Error generating plate heatmap for '{well_col}': {e}", exc_info=True)
                        st.error(f"Could not generate heatmap for `{well_col}`.")
                    finally:
                        if plate_heatmap_fig: 
                            plt.close(plate_heatmap_fig)


        # --- Cell Count Distribution (Knee Plot) ---
        st.markdown("---")
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

    except Exception as main_e:
         logger.error(f"Error in summary tab rendering: {main_e}", exc_info=True)
         st.error(f"An unexpected error occurred rendering summary: {main_e}")
    finally:
         # Ensure all top-level figures are closed
         if fig_summary_bar: 
            plt.close(fig_summary_bar)
         # plate_heatmap_fig closed in its loop
         if fig_knee: 
            plt.close(fig_knee)

# --- End of render_summary_tab ---