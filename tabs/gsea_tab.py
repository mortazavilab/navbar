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
        gsea_gene_sets = st.multiselect(
            "Select Gene Set Libraries:",
            options=AVAILABLE_GENE_SETS,
            default=["GO_Biological_Process_2025"],  # Default selection
            key="gsea_library_multiselect_form",
            help="Choose one or more gene set databases (e.g., KEGG, GO) for enrichment analysis."
        )

        # Advanced Options (Optional)
        st.checkbox("Show Advanced GSEA Options", key="show_advanced_gsea")
        gsea_min_size = 5
        gsea_max_size = 15000
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
        top_n_default = int(st.session_state.get("n_top_markers_for_gsea", 10))
        max_possible_n = 300
        gsea_n_top_markers = st.number_input(
            "Number of Top Markers to Use for GSEA:",
            min_value=5,
            max_value=max_possible_n,
            value=top_n_default,
            step=1,
            key="gsea_n_top_markers",
            help=f"Select the number of top markers to use for GSEA. Maximum is {max_possible_n}."
        )   

        # Submit Button
        run_gsea_button = st.form_submit_button("Run Pathway Analysis")

        if run_gsea_button:
            st.session_state.active_tab = "📊 Pathway Analysis"
            st.session_state.gsea_results_dict = {}
            st.session_state.gsea_error = None

            if not gsea_selected_group:
                st.warning("Please select a Group/Cluster to analyze.")
                st.session_state.gsea_error = "No group selected."
            elif not gsea_gene_sets:
                st.warning("Please select a valid Gene Set Library.")
                st.session_state.gsea_error = "Invalid or unavailable gene set library selected."
            else:
                with st.spinner(f"Running GSEA Prerank for group '{gsea_selected_group}' on '{gsea_gene_sets}'..."):
                    try:
                        # Get hash of marker df for caching
                        marker_df = st.session_state.calculated_markers_result_df
                        ranking_metric = gsea_ranking_metric
                        n_top_markers = int(gsea_n_top_markers)

                        # Filter to the selected group and top N by metric
                        group_mask = marker_df[RANK_GROUP_COL] == gsea_selected_group
                        marker_df_group = marker_df[group_mask].copy()
                        marker_df_group = marker_df_group.sort_values(by=ranking_metric, ascending=False).head(n_top_markers)
                        marker_df_hash_val = get_adata_hash(marker_df_group) if marker_df_group is not None else "no_markers"

                        gsea_results_dict = {}
                        for gene_set in gsea_gene_sets:
                            try:
                                gsea_results = run_gsea_prerank(
                                _marker_results_df_ref=marker_df_group,
                                marker_df_hash=marker_df_hash_val,
                                selected_group=gsea_selected_group,
                                gene_sets=gene_set,
                                ranking_metric=ranking_metric,
                                min_size=gsea_min_size,
                                max_size=gsea_max_size,
                                permutation_num=gsea_permutations
                                )
                                gsea_results_dict[gene_set] = gsea_results
                            except Exception as e:
                                gsea_results_dict[gene_set] = None
                                logger.error(f"GSEA failed for gene set '{gene_set}': {e}", exc_info=True)
                            finally:
                                pass
                        
                        st.session_state.gsea_results_dict = gsea_results_dict
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

    if st.session_state.get('gsea_results_dict') is not None:
        gsea_results_dict = st.session_state.get('gsea_results_dict', {})
        gsea_params = st.session_state.get('gsea_params_display', {})
        if gsea_params:
            st.write(f"Displaying results for Group: **{gsea_params.get('group', 'N/A')}**")

        any_results = False
        for gene_set, gsea_df in gsea_results_dict.items():
            if gsea_df is not None and not gsea_df.empty:
                # Filter for significant results (e.g., FDR < 0.25 or p < 0.05)
                sig_mask = None
                for col in ['fdr', 'FDR q-val', 'FDR', 'qval', 'q_val', 'padj']:
                    if col in gsea_df.columns:
                        sig_mask = gsea_df[col] < 0.25
                        break
                if sig_mask is None:
                    for col in ['pval', 'p_val', 'NOM p-val', 'p-val', 'pvalue']:
                        if col in gsea_df.columns:
                            sig_mask = gsea_df[col] < 0.05
                            break
                if sig_mask is not None:
                    sig_df = gsea_df[sig_mask]
                else:
                    sig_df = gsea_df

                if not sig_df.empty:
                    any_results = True
                    with st.expander(f"Gene Set: {gene_set} ({len(sig_df)} significant pathways)", expanded=False):
                        st.dataframe(
                            sig_df,
                            use_container_width=True,
                            column_config={
                                "NES": st.column_config.NumberColumn("NES", format="%.3f"),
                                "p_val": st.column_config.NumberColumn("p-value", format="%.2E"),
                                "fdr": st.column_config.NumberColumn("FDR q-value", format="%.2E"),
                                "genes": st.column_config.ListColumn("Core Enrichment Genes", width="large")
                            }
                        )
                        # Download button for each gene set
                        csv = sig_df.to_csv(index=False).encode('utf-8')
                        fname = sanitize_filename(f"gsea_prerank_{gsea_params.get('group', 'group')}_{gene_set}.csv")
                        st.download_button(
                            label=f"Download {gene_set} Results as CSV",
                            data=csv,
                            file_name=fname,
                            mime='text/csv',
                        )
        if not any_results:
            st.info("No significant pathways found for any selected gene set with the current settings.")
    else:
        st.write("Run Pathway Analysis using the form above to see results.")
