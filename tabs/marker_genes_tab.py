# scanpy_viewer/tabs/marker_genes_tab.py

import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import logging
import anndata as ad # Import AnnData
# Import analysis functions using relative path
try:
    from ..analysis.marker_analysis import get_markers_df_from_uns, calculate_rank_genes_df
    # Import utils using relative path
    from ..utils import AnalysisError, PlottingError, generate_download_button, get_figure_bytes, sanitize_filename
    from ..config import DEFAULT_N_MARKERS, MARKER_METHODS, DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI
except ImportError: # Fallback for running/testing stand-alone
    from analysis.marker_analysis import get_markers_df_from_uns, calculate_rank_genes_df
    from utils import AnalysisError, PlottingError, generate_download_button, get_figure_bytes, sanitize_filename
    from config import DEFAULT_N_MARKERS, MARKER_METHODS, DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI


logger = logging.getLogger(__name__)

# Default number of markers to show in visualizations AND tables
DEFAULT_N_MARKERS_DISPLAY = 10

def render_marker_genes_tab(adata_vis, adata_full, valid_obs_cat_cols, selected_color_var):
    """
    Renders the content for the Marker Genes tab.
    Shows filtered marker tables (with fraction bars) and allows visualization of top N markers.
    """
    if not isinstance(adata_vis, ad.AnnData) or not isinstance(adata_full, ad.AnnData):
         st.error("Invalid data provided to marker genes tab.")
         return

    st.subheader("Marker Gene Identification")

    if not valid_obs_cat_cols:
         st.warning("No suitable categorical columns found in `adata.obs` for grouping.")
         return

    # Select Group By Factor
    default_marker_index = 0
    if selected_color_var in valid_obs_cat_cols:
         try: default_marker_index = valid_obs_cat_cols.index(selected_color_var)
         except ValueError: pass

    marker_groupby_selected = st.selectbox(
        "Group By for Markers:", options=valid_obs_cat_cols, index=default_marker_index,
        key="marker_group_select_display",
        help="Select the categorical variable defining groups for marker gene analysis."
    )

    if not marker_groupby_selected:
        st.warning("Please select a categorical variable ('Group By for Markers').")
        return

    # --- Show Precomputed Markers ---
    st.markdown("---")
    st.markdown("**Precomputed Markers (from `.uns['rank_genes_groups']`)**")
    st.caption("Shows top N markers per group from the loaded file's `adata.uns`. Full results available via download.")

    precomputed_markers_df_full = None
    fig_dot_pre = None

    try:
        precomputed_markers_df_full, precomputed_key, key_match_warning = get_markers_df_from_uns(
            adata_full.uns,
            expected_groupby=marker_groupby_selected
        )

        if precomputed_markers_df_full is not None and not precomputed_markers_df_full.empty:
            if key_match_warning: st.warning(f"Warning: Precomputed markers used `groupby='{precomputed_key}'`, but selected is '{marker_groupby_selected}'. Displaying stored table.", icon="⚠️")
            else:
                 if precomputed_key == marker_groupby_selected: st.success(f"Found precomputed markers table (groupby='{precomputed_key}')")
                 else: st.info(f"Found precomputed markers table (groupby='{precomputed_key}')")

            max_genes_available_pre = precomputed_markers_df_full.groupby('group')['names'].count().max()
            n_markers_pre_display = st.slider(
                 "Number of top markers per group (Table & Plot):", 1, min(50, max_genes_available_pre),
                 min(DEFAULT_N_MARKERS_DISPLAY, max_genes_available_pre), key="n_markers_pre_display_slider",
                 help="Select N top markers (by score) per group for table and plot."
            )

            logger.debug(f"Filtering precomputed table to top {n_markers_pre_display} markers per group.")
            precomputed_markers_df_display = precomputed_markers_df_full.groupby('group').head(n_markers_pre_display).reset_index(drop=True)

            # --- MODIFIED: Add column config for progress bars ---
            column_config_pre = {}
            # Check which fraction columns exist and configure them
            # Use pct_nz_group first if available, fallback to pts
            pct_col_group = None
            if 'pct_nz_group' in precomputed_markers_df_display.columns: pct_col_group = 'pct_nz_group'
            elif 'pts' in precomputed_markers_df_display.columns: pct_col_group = 'pts'

            if pct_col_group:
                 column_config_pre[pct_col_group] = st.column_config.ProgressColumn(
                      label="Fraction Expressing (Group)",
                      help=f"Fraction of cells in the group expressing the gene (column: {pct_col_group})",
                      format="%.3f", min_value=0.0, max_value=1.0,
                 )
            # Use pct_nz_reference first if available, fallback to pts_rest
            pct_col_ref = None
            if 'pct_nz_reference' in precomputed_markers_df_display.columns: pct_col_ref = 'pct_nz_reference'
            elif 'pts_rest' in precomputed_markers_df_display.columns: pct_col_ref = 'pts_rest'

            if pct_col_ref:
                 column_config_pre[pct_col_ref] = st.column_config.ProgressColumn(
                      label="Fraction Expressing (Reference)",
                      help=f"Fraction of cells in reference groups expressing the gene (column: {pct_col_ref})",
                      format="%.3f", min_value=0.0, max_value=1.0,
                 )
            logger.debug(f"Precomputed table column config: {column_config_pre}")
            # --- End Modification ---

            # Display filtered table with configured columns
            st.dataframe(
                precomputed_markers_df_display,
                height=300,
                use_container_width=True,
                column_config=column_config_pre # Apply the config
            )

            # Download FULL table button (remains unchanged)
            try:
                 csv_bytes = precomputed_markers_df_full.to_csv(index=False).encode('utf-8')
                 fname_base = f"precomputed_markers_ALL_{precomputed_key or 'unknown_key'}"
                 fname_csv = sanitize_filename(fname_base, extension="csv")
                 generate_download_button(csv_bytes, fname_csv, "Download Full Table (CSV)", "text/csv", key="download_precomputed_markers_csv_full")
            except Exception as e: st.error(f"Failed full table download prep: {e}"); logger.error(f"Precomputed full CSV download failed: {e}", exc_info=True)

            # --- Visualize Top N Precomputed Markers (remains largely unchanged) ---
            top_markers_pre_viz_list = precomputed_markers_df_display['names'].unique().tolist()
            top_markers_pre_viz_list = [g for g in top_markers_pre_viz_list if g in adata_vis.var_names]
            if top_markers_pre_viz_list:
                with st.expander(f"Visualize Top {n_markers_pre_display} Precomputed Markers (Dot Plot)", expanded=True):
                    # [ Plotting code remains the same ]
                    try:
                        logger.info(f"Plotting dotplot: top {n_markers_pre_display} precomputed markers, group='{marker_groupby_selected}'.")
                        n_groups_display = len(precomputed_markers_df_full['group'].unique())
                        fig_dot_pre, ax_dot_pre = plt.subplots(figsize=(max(6, len(top_markers_pre_viz_list)*0.35), max(4, n_groups_display*0.4)))
                        sc.pl.dotplot(adata_vis, var_names=top_markers_pre_viz_list, groupby=marker_groupby_selected, ax=ax_dot_pre, show=False, dendrogram=True)
                        plt.xticks(rotation=90); plt.tight_layout()
                        st.pyplot(fig_dot_pre)
                        # Download Button
                        fname_base = f"precomputed_markers_dotplot_top{n_markers_pre_display}_{marker_groupby_selected}"
                        fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                        plot_bytes = get_figure_bytes(fig_dot_pre, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                        generate_download_button(plot_bytes, fname, f"Download Plot ({DEFAULT_PLOT_FORMAT.upper()})", f"image/{DEFAULT_PLOT_FORMAT}", key="download_precomputed_markers_dotplot")
                    except PlottingError as pe: st.error(f"Plot download prep error: {pe}")
                    except Exception as plot_e: st.error(f"Dot plot failed: {plot_e}"); logger.warning(f"Precomputed dotplot failed: {plot_e}", exc_info=True)
                    finally:
                        if fig_dot_pre: plt.close(fig_dot_pre)
            elif not precomputed_markers_df_full.empty: st.info("Top precomputed markers not in current data; cannot visualize.")
        else: st.info(f"No precomputed markers found in `.uns`{f' for key {precomputed_key}' if precomputed_key else ''}.")
    except AnalysisError as e: st.error(f"Error extracting precomputed: {e}"); logger.error(f"Precomputed extraction failed: {e}", exc_info=True)
    except Exception as e: st.error(f"Unexpected error (precomputed): {e}"); logger.error(f"Precomputed check error: {e}", exc_info=True)


    # --- Calculated Markers Display ---
    st.markdown("---")
    st.markdown("**Calculated Markers Results**")
    st.caption(f"Shows top N markers per group from `sc.tl.rank_genes_groups` run via the form. Full results via download.")

    calculated_markers_df_full = None
    fig_dot_calc = None

    if st.session_state.get('calculated_markers_error'): st.error(f"Calculation Failed: {st.session_state.calculated_markers_error}")
    elif st.session_state.get('calculated_markers_result_df') is not None:
        calculated_markers_df_full = st.session_state.calculated_markers_result_df
        params = st.session_state.get('calculated_markers_params', {}); calc_method = params.get('method', 'N/A'); calc_groupby = params.get('groupby', 'N/A')
        st.markdown(f"**Results (Method: {calc_method}, GroupBy: {calc_groupby})**")

        if isinstance(calculated_markers_df_full, pd.DataFrame) and not calculated_markers_df_full.empty:
            max_genes_available_calc = calculated_markers_df_full.groupby('group')['names'].count().max()
            n_markers_calc_display = st.slider(
                 "Number of top markers per group (Table & Plot):", 1, min(50, max_genes_available_calc),
                 min(DEFAULT_N_MARKERS_DISPLAY, max_genes_available_calc), key="n_markers_calc_display_slider",
                 help="Select N top markers (by score) per group for table and plot."
            )

            logger.debug(f"Filtering calculated table to top {n_markers_calc_display} markers per group.")
            calculated_markers_df_display = calculated_markers_df_full.groupby('group').head(n_markers_calc_display).reset_index(drop=True)

            # --- MODIFIED: Add column config for progress bars ---
            column_config_calc = {}
            # Check which fraction columns exist and configure them
            pct_col_group_calc = None
            if 'pct_nz_group' in calculated_markers_df_display.columns: pct_col_group_calc = 'pct_nz_group'
            elif 'pts' in calculated_markers_df_display.columns: pct_col_group_calc = 'pts'

            if pct_col_group_calc:
                 column_config_calc[pct_col_group_calc] = st.column_config.ProgressColumn(
                      label="Fraction Expressing (Group)",
                      help=f"Fraction of cells in the group expressing the gene (column: {pct_col_group_calc})",
                      format="%.3f", min_value=0.0, max_value=1.0,
                 )

            pct_col_ref_calc = None
            if 'pct_nz_reference' in calculated_markers_df_display.columns: pct_col_ref_calc = 'pct_nz_reference'
            elif 'pts_rest' in calculated_markers_df_display.columns: pct_col_ref_calc = 'pts_rest'

            if pct_col_ref_calc:
                 column_config_calc[pct_col_ref_calc] = st.column_config.ProgressColumn(
                      label="Fraction Expressing (Reference)",
                      help=f"Fraction of cells in reference groups expressing the gene (column: {pct_col_ref_calc})",
                      format="%.3f", min_value=0.0, max_value=1.0,
                 )
            logger.debug(f"Calculated table column config: {column_config_calc}")
            # --- End Modification ---

            # Display filtered table with configured columns
            st.dataframe(
                calculated_markers_df_display,
                height=300,
                use_container_width=True,
                column_config=column_config_calc # Apply the config
            )

            # Download FULL table button (remains unchanged)
            try:
                csv_bytes = calculated_markers_df_full.to_csv(index=False).encode('utf-8')
                fname_base = f"calculated_markers_ALL_{calc_groupby}_{calc_method}"
                fname_csv = sanitize_filename(fname_base, extension="csv")
                generate_download_button(csv_bytes, fname_csv, "Download Full Table (CSV)", "text/csv", key="download_calculated_markers_csv_full")
            except Exception as e: st.error(f"Failed full table download prep: {e}"); logger.error(f"Calculated markers full CSV download error: {e}", exc_info=True)

            # --- Visualize Top N Calculated Markers (remains largely unchanged) ---
            top_markers_calc_viz_list = calculated_markers_df_display['names'].unique().tolist()
            top_markers_calc_viz_list = [g for g in top_markers_calc_viz_list if g in adata_vis.var_names]
            if top_markers_calc_viz_list:
                with st.expander(f"Visualize Top {n_markers_calc_display} Calculated Markers (Dot Plot)", expanded=True):
                    # [ Plotting code remains the same ]
                    try:
                        logger.info(f"Plotting dotplot: top {n_markers_calc_display} calculated markers, group='{calc_groupby}'.")
                        n_groups_display_calc = len(calculated_markers_df_full['group'].unique())
                        fig_dot_calc, ax_dot_calc = plt.subplots(figsize=(max(6, len(top_markers_calc_viz_list)*0.35), max(4, n_groups_display_calc*0.4)))
                        sc.pl.dotplot(adata_vis, var_names=top_markers_calc_viz_list, groupby=calc_groupby, ax=ax_dot_calc, show=False, dendrogram=True, use_raw=params.get('use_raw', False))
                        plt.xticks(rotation=90); plt.tight_layout()
                        st.pyplot(fig_dot_calc)
                        # Download Button
                        fname_base_plot = f"calculated_markers_dotplot_top{n_markers_calc_display}_{calc_groupby}_{calc_method}"
                        fname_plot = sanitize_filename(fname_base_plot, extension=DEFAULT_PLOT_FORMAT)
                        plot_bytes = get_figure_bytes(fig_dot_calc, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                        generate_download_button(plot_bytes, fname_plot, f"Download Plot ({DEFAULT_PLOT_FORMAT.upper()})", f"image/{DEFAULT_PLOT_FORMAT}", key="download_calculated_markers_dotplot")
                    except PlottingError as pe: st.error(f"Plot download prep error: {pe}")
                    except Exception as plot_e: st.error(f"Dot plot failed: {plot_e}"); logger.warning(f"Calculated dotplot failed: {plot_e}", exc_info=True)
                    finally:
                        if fig_dot_calc: plt.close(fig_dot_calc)
            elif not calculated_markers_df_full.empty: st.info("Top calculated markers not in current data; cannot visualize.")
        else: st.warning("Marker calculation done, but no results generated or invalid.")
    elif not st.session_state.get('calculated_markers_error'): st.info("Use form in main panel to calculate markers.")