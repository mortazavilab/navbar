# scanpy_viewer/tabs/pseudobulk_deg_tab.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict # Keep for type hint maybe
from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI
# Check if pyDESeq2 is installed via the analysis module's flag
from analysis.deg_analysis import PYDESEQ2_INSTALLED

logger = logging.getLogger(__name__)

def render_pseudobulk_deg_tab(deg_results, group1_levels, group2_levels):
    """Renders the content for the Pseudobulk DEG tab AFTER calculation."""

    if not PYDESEQ2_INSTALLED:
        # This message should ideally be shown near the run button,
        # but double-check here in case state is inconsistent.
        st.warning("`pydeseq2` library not installed. DEG analysis is unavailable.", icon="⚠️")
        return

    st.markdown("---") # Separator from the form above
    st.markdown("#### DEG Results (pyDESeq2)")

    # Handle case where DEG run failed
    if deg_results is None:
        if st.session_state.get('deg_error'):
            st.error(f"Pseudobulk DEG Error: {st.session_state.deg_error}")
        else:
            st.info("Run Pseudobulk DEG using the button above to see results here.")
        return

    # Handle case where DEG ran but produced no results
    if not isinstance(deg_results, pd.DataFrame):
         st.error(f"Internal Error: DEG results are not a DataFrame (type: {type(deg_results)}).")
         return
    if deg_results.empty:
        st.warning("DEG analysis ran but produced no results (empty table). This might happen if no genes passed filtering or if the contrast was invalid.")
        return

    # --- Display Summary Stats ---
    try:
        padj_threshold = 0.05 # Standard threshold
        if 'padj' not in deg_results.columns:
             raise KeyError("'padj' column not found in DEG results.")
        if 'log2FoldChange' not in deg_results.columns:
             raise KeyError("'log2FoldChange' column not found in DEG results.")

        n_genes_tested = deg_results.shape[0]
        # Handle potential NaNs in padj before comparison
        sig_mask = deg_results['padj'].notna() & (deg_results['padj'] < padj_threshold)
        n_sig = sig_mask.sum()

        lfc_threshold_up = 0 # Basic threshold for up/down - could be configurable
        lfc_threshold_down = 0

        # Handle potential NaNs in LFC
        lfc_notna_mask = deg_results['log2FoldChange'].notna()
        up_mask = sig_mask & lfc_notna_mask & (deg_results['log2FoldChange'] > lfc_threshold_up)
        down_mask = sig_mask & lfc_notna_mask & (deg_results['log2FoldChange'] < lfc_threshold_down)
        n_up = up_mask.sum()
        n_down = down_mask.sum()

        st.metric(f"Significant Genes (padj < {padj_threshold})", f"{n_sig} / {n_genes_tested}")

        # Display compared groups clearly
        group1_desc = ", ".join([f"{k}={v}" for k, v in group1_levels.items()])
        group2_desc = ", ".join([f"{k}={v}" for k, v in group2_levels.items()])
        st.caption(f"**Group 1 (Numerator):** {group1_desc}")
        st.caption(f"**Group 2 (Denominator):** {group2_desc}")

        st.write(f"Up-regulated in Group 1 vs Group 2 (LFC > {lfc_threshold_up}): **{n_up}**")
        st.write(f"Down-regulated in Group 1 vs Group 2 (LFC < {lfc_threshold_down}): **{n_down}**")


    except KeyError as e:
        st.warning(f"Could not calculate summary statistics - results table might be missing column: {e}")
        logger.warning(f"DEG summary stats failed due to missing column: {e}")
    except Exception as e:
        st.warning(f"Could not calculate summary statistics due to error: {e}")
        logger.warning(f"DEG summary stats failed: {e}", exc_info=True)

    # --- Display Results Table ---
    st.dataframe(deg_results)

    # Download Results Table
    try:
        csv_bytes = deg_results.to_csv().encode('utf-8')
        fname_base = "pseudobulk_deg_results_G1vG2" # Basic name
        fname_csv = sanitize_filename(fname_base, extension="csv")
        generate_download_button(
            csv_bytes,
            filename=fname_csv,
            label="Download Results Table (CSV)",
            mime="text/csv",
            key="download_deg_results_csv"
        )
    except Exception as e:
        st.error(f"Failed to prepare results table for download: {e}")
        logger.error(f"DEG results CSV download prep failed: {e}", exc_info=True)


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
