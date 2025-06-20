# navbar/tabs/gsea_tab.py

import streamlit as st
import logging
from collections import OrderedDict

from utils import get_adata_hash, sanitize_filename
from analysis.gsea_analysis import run_gsea_prerank, GSEAPY_INSTALLED, AVAILABLE_GENE_SETS, RANK_GROUP_COL, RANK_SCORE_COL

logger = logging.getLogger(__name__)

def render_gsea_tab():
    """Render the GSEA/Pathway Analysis tab with form and results display."""
    
    st.header("Pathway Enrichment Analysis (GSEA Prerank)")

    if not GSEAPY_INSTALLED:
        st.warning("`gseapy` library not installed. Pathway analysis is unavailable. Please install it (`pip install gseapy`) and restart.", icon="⚠️")
        return
    
    if 'calculated_markers_result_df' not in st.session_state or st.session_state.calculated_markers_result_df is None:
        st.info("📊 Please calculate marker genes in the 'Marker Genes' tab first to enable Pathway Analysis.")
        return

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
        marker_groups = [""]  # Add empty option
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
        gsea_permutations = 100  # Keep low for web app speed
        gsea_ranking_metric = marker_params.get('ranking_metric', RANK_SCORE_COL)  # Use default from markers calc

        if st.session_state.show_advanced_gsea:
            with st.expander("Advanced GSEA Options"):
                # Allow overriding the ranking metric if marker df has multiple (e.g., logfc)
                possible_metrics = [col for col in st.session_state.calculated_markers_result_df.columns if col in ['scores', 'logfoldchanges', 'pvals_adj']]  # Example potential metrics
                if not possible_metrics: 
                    possible_metrics = [RANK_SCORE_COL]  # Fallback
                gsea_ranking_metric = st.selectbox(
                    "Ranking Metric:", 
                    options=possible_metrics, 
                    index=possible_metrics.index(gsea_ranking_metric) if gsea_ranking_metric in possible_metrics else 0, 
                    key="gsea_metric_select"
                )

                gsea_min_size = st.number_input("Min Gene Set Size:", min_value=1, value=15, key="gsea_min_size")
                gsea_max_size = st.number_input("Max Gene Set Size:", min_value=10, value=500, key="gsea_max_size")
                gsea_permutations = st.number_input(
                    "Number of Permutations:", 
                    min_value=10, 
                    max_value=1000, 
                    value=100, 
                    step=10, 
                    key="gsea_perms", 
                    help="Higher numbers increase runtime but improve p-value accuracy."
                )

        # Submit Button
        run_gsea_button = st.form_submit_button("Run Pathway Analysis")

        if run_gsea_button:
            st.session_state.active_tab = "📊 Pathway Analysis"
            st.session_state.gsea_results_df = None  # Clear previous results
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
                        )
                        st.session_state.gsea_results_df = gsea_results
                        st.session_state.gsea_params_display = {  # Store params for context
                            'group': gsea_selected_group,
                            'gene_sets': gsea_gene_sets,
                            'ranking_metric': gsea_ranking_metric,
                            'min_size': gsea_min_size,
                            'max_size': gsea_max_size,
                            'permutations': gsea_permutations
                        }
                        st.success(f"GSEA Prerank complete for group '{gsea_selected_group}'. Results below.")
                        logger.info(f"GSEA analysis successful for group {gsea_selected_group}")

                    except Exception as e:
                        st.session_state.gsea_error = f"Pathway Analysis Error: {e}"
                        logger.error(f"GSEA analysis failed: {e}", exc_info=True)
                    finally:
                        pass

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
