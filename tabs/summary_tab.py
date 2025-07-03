# navbar/tabs/summary_tab.py
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

from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI
from tabs.heatmap_utils import create_plate_heatmap_fig, create_heatmap_grid, parse_well_id, natural_sort_key

logger = logging.getLogger(__name__)

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
    except Exception as e:
        logger.error(f"Error rendering summary tab: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while rendering the summary tab: {e}")  
# --- End of render_summary_tab ---