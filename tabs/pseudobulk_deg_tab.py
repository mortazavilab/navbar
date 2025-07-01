# navbar/tabs/pseudobulk_deg_tab.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict
import anndata as ad

from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError, get_adata_hash
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI, MIN_DESEQ_COUNT_SUM
from analysis.deg_analysis import PYDESEQ2_INSTALLED, prepare_metadata_and_run_deseq
from aggregation import cached_aggregate_adata
from utils import AggregationError, AnalysisError, FactorNotFoundError

logger = logging.getLogger(__name__)

def render_pseudobulk_deg_tab(adata_vis, valid_obs_cat_cols, dynamic_layer_options):
    st.subheader("Pseudobulk Differential Expression (pyDESeq2)")
    
    if not PYDESEQ2_INSTALLED:
        st.warning("`pydeseq2` is not installed. Please `pip install pydeseq2`.")
        return
    
    if not isinstance(adata_vis, ad.AnnData):
        st.error("Invalid data provided to DEG tab.")
        return
    if not valid_obs_cat_cols:
        st.warning("No suitable categorical columns found in `adata.obs` for grouping.")
        return

    adata_vis_hash = get_adata_hash(adata_vis)
    st.markdown("**1. Define Analysis Factors**")

    # Replicate = sample-level
    deg_replicate_factor = st.selectbox(
        "Replicate Factor (required):", options=valid_obs_cat_cols, key="deg_replicate_select_form",
        help="Variable identifying replicates (e.g., sample or mouse id).",
        width=500
    )
    # Comparison is typically genotype (or condition)
    deg_comparison_factor = st.selectbox(
        "Comparison Factor (required):", options=[c for c in valid_obs_cat_cols if c != deg_replicate_factor],
        key="deg_comparison_factor_form",
        help="Variable defining the comparison groups (e.g., genotype, treatment).",
        width=500
    )
    # Cell type for which to run DE (do per type)
    deg_celltype_factor = st.selectbox(
        "Cell Type Factor (e.g. celltype or general_celltype)", options=[c for c in valid_obs_cat_cols if c not in (deg_replicate_factor, deg_comparison_factor)],
        key="deg_celltype_factor_form",
        width=500
    )
    selected_celltype_level = st.selectbox(
        "Choose celltype for analysis:", options=adata_vis.obs[deg_celltype_factor].dropna().unique().tolist(),
        key="deg_celltype_level_form",
        width=500
    )
    # Counts data layer
    default_deg_layer_idx = 0
    for layername in ['cellbender_counts', 'raw_counts', 'counts', 'Auto-Select']:
        if layername in dynamic_layer_options:
            default_deg_layer_idx = dynamic_layer_options.index(layername)
            break
    deg_layer_key = st.selectbox(
        "Data Source for Aggregation (Counts):", options=dynamic_layer_options,
        index=default_deg_layer_idx, key="deg_layer_select_form",
        help="Select matrix/layer with raw counts for DESeq2 (aggregation by sum).",
        width=500
    )

    # Step 2: DEG Calculation form
    with st.form("deg_form", width=600):
        st.markdown("**2. Select Comparison Groups**", width=450)
        group1_level = st.selectbox(
            f"{deg_comparison_factor} group 1:", options=sorted(adata_vis.obs[deg_comparison_factor].dropna().unique().astype(str)),
            key="deg_g1_level",
            width=400
        )
        group2_level = st.selectbox(
            f"{deg_comparison_factor} group 2:", options=sorted(adata_vis.obs[deg_comparison_factor].dropna().unique().astype(str)),
            key="deg_g2_level",
            width=400
        )
        # Validation
        deg_form_valid = (group1_level != group2_level)
        if not deg_form_valid:
            st.warning("Group 1 and Group 2 must differ.", width=450)

        st.checkbox("Show Advanced DEG Options", key="show_advanced_deg") # State key
        min_sum_filter = MIN_DESEQ_COUNT_SUM # Default
        min_nonzero_samples = 2 # Default
        
        if st.session_state.get("show_advanced_deg"):
            with st.expander("Advanced Options"):
                min_sum_filter = st.number_input(
                    "Min Gene Count Sum Filter:", min_value=0, value=MIN_DESEQ_COUNT_SUM, 
                    step=5, key="deg_min_count_form",
                    help="Filter genes with total count across pseudobulk samples < this value."
                )
                min_nonzero_samples = st.number_input(
                    "Min Non-zero Samples:", min_value=1, value=2,
                    step=1, key="deg_min_nonzero_form",
                    help="Minimum number of samples that must have non-zero counts for a gene."
                )

        #run_deg_button = st.form_submit_button("Run Pseudobulk DEG", disabled=not deg_form_valid)
        run_deg_button = st.form_submit_button("Run Pseudobulk DEG")

        if run_deg_button:
            st.session_state.deg_results_df = None
            st.session_state.deg_error = None
            st.session_state.deg_params_display = None

            try:
                # 1. SUBSET TO THE CELL TYPE OF INTEREST **BEFORE AGGREGATION**
                celltype_mask = (adata_vis.obs[deg_celltype_factor].astype(str) == str(selected_celltype_level))
                adata_celltype = adata_vis[celltype_mask].copy()
                # AGGREGATE BY [replicate]
                deg_aggregation_keys = [deg_replicate_factor]
                agg_keys_tuple = tuple(deg_aggregation_keys)
                logger.info(f"Requesting Aggregation for DEG by keys: {agg_keys_tuple}")

                with st.spinner("Aggregating data..."):
                    adata_agg_deg = cached_aggregate_adata(
                        _adata_ref=adata_celltype,
                        _adata_ref_hash=adata_vis_hash, # Technically, the mask makes cache less useful
                        grouping_vars_tuple=agg_keys_tuple,
                        selected_layer_key=deg_layer_key,
                        agg_func='sum'
                    )
                logger.info(f"Aggregation complete. Shape: {adata_agg_deg.shape}")

                # RE-ATTACH COMPARISON FACTOR (genotype etc.) to new aggregated .obs using replicate
                meta_cols = [deg_replicate_factor, deg_comparison_factor]
                sample_meta = (adata_vis.obs[meta_cols]
                               .drop_duplicates(subset=deg_replicate_factor)
                               .set_index(deg_replicate_factor))
                adata_agg_deg.obs = (
                    adata_agg_deg.obs.join(sample_meta, on=deg_replicate_factor, rsuffix='_meta')
                )
                if adata_agg_deg.obs[deg_comparison_factor].isnull().any():
                    st.warning("Some pseudobulk samples could not be assigned to a comparison group! Check your metadata.")

                # Prepare group1/group2 levels config for DESeq2
                group1_levels = OrderedDict({deg_comparison_factor: group1_level})
                group2_levels = OrderedDict({deg_comparison_factor: group2_level})
                deg_comparison_factors = [deg_comparison_factor]

                # Step 2: Run DESeq2
                with st.spinner("Running pyDESeq2..."):
                    deg_results_df = prepare_metadata_and_run_deseq(
                        adata_agg_deg,
                        comparison_factors=deg_comparison_factors,
                        replicate_factor=deg_replicate_factor,
                        group1_levels=group1_levels,
                        group2_levels=group2_levels,
                        min_count_sum_filter=min_sum_filter,
                        min_nonzero_samples=min_nonzero_samples
                    )
                st.session_state.deg_results_df = deg_results_df
                st.session_state.deg_params_display = {
                    'group1': group1_levels,
                    'group2': group2_levels,
                    'celltype': selected_celltype_level
                }
                logger.info(f"DESeq2 analysis complete. {len(deg_results_df) if deg_results_df is not None else 0} genes.")
                st.success("DEG analysis complete. Results below.")

            except (AggregationError, AnalysisError, FactorNotFoundError, ValueError, ImportError, TypeError) as e:
                st.session_state.deg_error = f"Pseudobulk DEG Error: {e}"
                logger.error(f"Pseudobulk DEG failed: {e}", exc_info=True)
            except Exception as e:
                st.session_state.deg_error = f"An unexpected error occurred during Pseudobulk DEG: {e}"
                logger.error(f"Unexpected Pseudobulk DEG error: {e}", exc_info=True)

    _render_deg_results()

def _render_deg_results():
    st.markdown("---")
    st.markdown("#### DEG Results (pyDESeq2)")

    deg_results = st.session_state.get('deg_results_df')
    stored_params = st.session_state.get('deg_params_display')
    default_params = {'group1': OrderedDict(), 'group2': OrderedDict()}
    deg_params = stored_params if stored_params is not None else default_params
    group1_levels = deg_params['group1']
    group2_levels = deg_params['group2']

    if deg_results is None:
        if st.session_state.get('deg_error'):
            st.error(f"Pseudobulk DEG Error: {st.session_state.deg_error}")
        else:
            st.info("Run Pseudobulk DEG using the form above to see results here.")
        return
    if not isinstance(deg_results, pd.DataFrame):
        st.error(f"Internal Error: DEG results are not a DataFrame (type: {type(deg_results)}).")
        return
    if deg_results.empty:
        st.warning("No DEGs found.")
        return

    padj_threshold = 0.05
    if 'padj' in deg_results.columns and 'log2FoldChange' in deg_results.columns:
        n_sig = (deg_results['padj'] < padj_threshold).sum()
        n_tested = len(deg_results)
        st.metric(f"Significant Genes (padj < {padj_threshold})", f"{n_sig} / {n_tested}")
    else:
        st.warning("Missing padj/log2FoldChange for summary stats.")

    group1_desc = ", ".join([f"{k}={v}" for k, v in group1_levels.items()])
    group2_desc = ", ".join([f"{k}={v}" for k, v in group2_levels.items()])
    st.caption(f"**Group 1 (Numerator):** {group1_desc}")
    st.caption(f"**Group 2 (Denominator):** {group2_desc}")

    column_config = {
    "pvalue": st.column_config.NumberColumn(
        "P-value (Scientific Notation)",
        help="Large numbers displayed in scientific notation",
        format="scientific", # Set the format to scientific notation
        ),
    "padj": st.column_config.NumberColumn(
        "Adjusted P-value (padj)",
        help="Adjusted p-value after multiple testing correction",
        format="scientific", # Set the format to scientific notation
        )
    }
    st.dataframe(deg_results, column_config=column_config)

    # Download
    try:
        csv_bytes = deg_results.to_csv().encode('utf-8')
        fname_base = "pseudobulk_deg_results"
        fname_csv = sanitize_filename(fname_base, extension="csv")
        generate_download_button(csv_bytes, filename=fname_csv, label="Download Results Table (CSV)",
                                mime="text/csv", key="download_deg_results_csv")
    except Exception as e:
        st.error(f"Failed to prepare results table for download: {e}")
    
    # --- Volcano Plot ---
    with st.expander("Volcano Plot"):
        required_cols_volcano = ['log2FoldChange', 'padj']
        if not all(col in deg_results.columns for col in required_cols_volcano):
            st.warning(f"Cannot generate Volcano plot: Required columns missing ({required_cols_volcano}).")
        else:
            fig_volcano = None # Initialize figure variable
            try:
                logger.info("Generating Volcano plot.")
                fig_volcano, ax_volcano = plt.subplots(figsize=(7, 6)) # Adjust size

                # Prepare data for plotting: handle NaNs and zeros for log transform
                plot_df = deg_results[['log2FoldChange', 'padj']].copy()
                plot_df = plot_df.dropna(subset=['log2FoldChange', 'padj']) # Drop rows with NaN in essential columns

                if plot_df.empty:
                     st.warning("No valid data points (after removing NaNs) to generate Volcano plot.")
                else:
                    # Replace padj=0 with a very small number for log transform
                    min_padj = plot_df.loc[plot_df['padj'] > 0, 'padj'].min() if (plot_df['padj'] > 0).any() else 1e-300
                    # Ensure min_padj is a float before multiplication
                    min_padj = float(min_padj) if min_padj is not None else 1e-300
                    plot_df['padj_clipped'] = plot_df['padj'].clip(lower=min_padj * 0.1) # Clip zeros slightly below min positive

                    # Calculate -log10(padj)
                    plot_df['-log10padj'] = -np.log10(plot_df['padj_clipped'])
                    # Cap -log10padj if it becomes extremely large (optional)
                    max_log_val = np.percentile(plot_df['-log10padj'][np.isfinite(plot_df['-log10padj'])], 99.5) if plot_df['-log10padj'].notna().any() else 300
                    max_log_val = max(max_log_val, 10) # Ensure cap is at least 10
                    plot_df['-log10padj'] = plot_df['-log10padj'].clip(upper=max_log_val)

                    # Basic volcano plot - all points
                    ax_volcano.scatter(
                        plot_df['log2FoldChange'],
                        plot_df['-log10padj'],
                        alpha=0.4, s=10, c='grey', label='Non-significant', rasterized=True # Rasterize for large numbers of points
                    )

                    # Highlight significant points (padj < threshold)
                    padj_threshold = 0.05
                    sig_df = plot_df[plot_df['padj'] < padj_threshold]
                    ax_volcano.scatter(
                        sig_df['log2FoldChange'],
                        sig_df['-log10padj'],
                        alpha=0.6, s=15, c='red', label=f'Significant (padj < {padj_threshold})', rasterized=True
                    )

                    # Add lines for thresholds
                    padj_line_val = -np.log10(padj_threshold)
                    ax_volcano.axhline(padj_line_val, ls='--', color='grey', lw=0.8)
                    ax_volcano.axvline(0, ls='-', color='black', lw=0.5) # Center line at LFC=0

                    # Optional LFC threshold lines (make configurable?)
                    lfc_vis_thresh = 1.0
                    ax_volcano.axvline(lfc_vis_thresh, ls=':', color='grey', lw=0.8)
                    ax_volcano.axvline(-lfc_vis_thresh, ls=':', color='grey', lw=0.8)

                    # Labels and Title
                    ax_volcano.set_xlabel("Log2 Fold Change (Group 1 vs Group 2)")
                    ax_volcano.set_ylabel("-Log10 Adjusted P-value")
                    ax_volcano.set_title("Volcano Plot")
                    ax_volcano.legend(fontsize=8)
                    ax_volcano.grid(True, which='major', linestyle=':', linewidth='0.5', color='lightgrey')

                    st.pyplot(fig_volcano)
                    logger.info("Volcano plot displayed.")

                    # Download Volcano Plot
                    try:
                        fname_base_volcano = "pseudobulk_deg_volcano"
                        fname_volcano = sanitize_filename(fname_base_volcano, extension=DEFAULT_PLOT_FORMAT)
                        img_bytes_volcano = get_figure_bytes(fig_volcano, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                        generate_download_button(
                            img_bytes_volcano,
                            filename=fname_volcano,
                            label=f"Download Volcano Plot ({DEFAULT_PLOT_FORMAT.upper()})",
                            mime=f"image/{DEFAULT_PLOT_FORMAT}",
                            key="download_deg_volcano_plot"
                        )
                    except PlottingError as pe:
                        st.error(f"Error preparing Volcano plot for download: {pe}")
                        logger.error(f"Volcano plot download prep failed: {pe}", exc_info=True)
                    except Exception as dle:
                        st.error(f"An unexpected error occurred during Volcano plot download preparation: {dle}")
                        logger.error(f"Volcano Download button generation error: {dle}", exc_info=True)

            except ValueError as ve:
                # Catches issues like all NaN/Inf after processing
                st.error(f"Could not generate Volcano plot due to data issue: {ve}. Check input data and filtering.")
                logger.warning(f"Volcano plot failed due to ValueError: {ve}", exc_info=True)
                if fig_volcano: plt.close(fig_volcano)
            except Exception as e:
                st.error(f"Could not generate Volcano plot: {e}")
                logger.warning(f"Volcano plot failed: {e}", exc_info=True)
                if fig_volcano: plt.close(fig_volcano) # Ensure figure is closed
