# scanpy_viewer/tabs/pseudobulk_pca_tab.py

import streamlit as st
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import logging
import anndata as ad # Import AnnData
from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI

logger = logging.getLogger(__name__)

def render_pseudobulk_pca_tab(pca_result, pca_grouping_vars):
    """Renders the content for the Pseudobulk PCA tab AFTER calculation."""
    if not isinstance(pca_result, ad.AnnData) and pca_result is not None: # Check if it's None or AnnData
         st.error("Invalid PCA result data provided to PCA tab.")
         return

    st.markdown("---") # Separator from the form above
    st.markdown("#### PCA Results")

    if pca_result is None:
        # Check for error stored in session state from the calculation step
        if st.session_state.get('pca_error'):
            st.error(f"Pseudobulk PCA Error: {st.session_state.pca_error}")
        else:
            st.info("Run Pseudobulk PCA using the button above to see results here.")
        return

    # If we have results, display them
    st.write(f"PCA performed on aggregated data. Input shape for PCA (after potential HVG filter): `{pca_result.shape}`")

    # Allow coloring PCA plot by the aggregation variables present in the result
    # These should have been copied to .obs during aggregation
    pca_color_options = [col for col in pca_grouping_vars if col in pca_result.obs.columns]
    if not pca_color_options:
        # Fallback if grouping vars somehow missing - use any categorical/object column
        pca_color_options = [col for col in pca_result.obs.columns if pd.api.types.is_categorical_dtype(pca_result.obs[col]) or pd.api.types.is_object_dtype(pca_result.obs[col])]
        if not pca_color_options: # If still none, use first column as last resort
            pca_color_options = pca_result.obs.columns.tolist() if not pca_result.obs.empty else [""]

    # Try to find the first original grouping var as default, else first option
    default_pca_color = pca_grouping_vars[0] if pca_grouping_vars and pca_grouping_vars[0] in pca_color_options else (pca_color_options[0] if pca_color_options else "")
    default_idx = pca_color_options.index(default_pca_color) if default_pca_color in pca_color_options else 0

    pca_plot_color = st.selectbox(
        "Color PCA Plot By:",
        options=pca_color_options,
        index=default_idx,
        key="pca_plot_color_select"
    )

    if not pca_plot_color: # Handle case where no color options are available
         st.warning("No suitable variables found in aggregated data to color PCA plot by.")
         return

    fig_pca = None # Initialize figure variable
    try:
        logger.info(f"Plotting pseudobulk PCA colored by '{pca_plot_color}'.")
        fig_pca, ax_pca = plt.subplots(figsize=(7, 6)) # Adjust size
        sc.pl.pca(
            pca_result,
            color=pca_plot_color,
            ax=ax_pca,
            show=False,
            size=60, # Adjust point size
            legend_loc='right margin',
            legend_fontsize=8,
            title=f"Pseudobulk PCA | Color: {pca_plot_color}"
        )
        st.pyplot(fig_pca)
        logger.info("PCA plot displayed.")

        # Download Plot
        try:
            fname_base = f"pseudobulk_pca_{pca_plot_color}"
            fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
            img_bytes_pca = get_figure_bytes(fig_pca, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
            generate_download_button(
                img_bytes_pca,
                filename=fname,
                label=f"Download PCA Plot ({DEFAULT_PLOT_FORMAT.upper()})",
                mime=f"image/{DEFAULT_PLOT_FORMAT}",
                key=f"download_pca_plot_{pca_plot_color}"
            )
        except PlottingError as pe:
             st.error(f"Error preparing PCA plot for download: {pe}")
             logger.error(f"PCA plot download preparation failed: {pe}", exc_info=True)
        except Exception as dle:
             st.error(f"An unexpected error occurred during PCA plot download preparation: {dle}")
             logger.error(f"PCA Download button generation error: {dle}", exc_info=True)


        # Display PCA Variance Ratio
        with st.expander("PCA Variance Explained"):
            if 'pca' in pca_result.uns and 'variance_ratio' in pca_result.uns['pca']:
                variance_ratio = pca_result.uns['pca']['variance_ratio']
                if variance_ratio is not None and len(variance_ratio) > 0:
                     pc_labels = [f'PC{i+1}' for i in range(variance_ratio.size)]

                     variance_ratio_df = pd.DataFrame({
                         'PC': [f'PC{i+1}' for i in range(variance_ratio.size)],
                         'Variance Ratio': variance_ratio
                     })
                     variance_ratio_df['Cumulative Variance'] = variance_ratio_df['Variance Ratio'].cumsum()

                    # Convert 'PC' column to an ordered Categorical type
                     # This explicitly defines the sorting order for pandas and potentially Streamlit
                     variance_ratio_df['PC'] = pd.Categorical(
                         variance_ratio_df['PC'],
                         categories=pc_labels, # Use the generated list as the order
                         ordered=True
                     )
                     
                     col_var1, col_var2 = st.columns(2)
                     with col_var1:
                          st.dataframe(variance_ratio_df, height=300)
                     with col_var2:
                          st.line_chart(variance_ratio_df, x='PC', y=['Variance Ratio', 'Cumulative Variance'])
                else:
                     st.warning("PCA variance ratio information is empty or invalid.")
            else:
                st.warning("PCA variance information not found in `adata.uns['pca']`.")
    except KeyError as ke:
        st.error(f"PCA Plotting Error: Variable '{ke}' not found in PCA results.")
        logger.error(f"PCA plot KeyError: {ke}", exc_info=True)
        if fig_pca: plt.close(fig_pca)
    except ValueError as ve:
        st.error(f"PCA Plotting Error: {ve}. Check plot parameters or data.")
        logger.error(f"PCA plot ValueError: {ve}", exc_info=True)
        if fig_pca: plt.close(fig_pca)
    except Exception as e:
        st.error(f"An unexpected error occurred during PCA plotting: {e}")
        logger.error(f"PCA plotting failed: {e}", exc_info=True)
        if fig_pca: plt.close(fig_pca) # Ensure figure is closed   
    # --- Add this section for PC Loadings ---
    st.markdown("---") # Separator
    st.markdown("#### Principal Component Loadings")

    # Check if loadings are available
    if 'PCs' in pca_result.varm:
        loadings = pca_result.varm['PCs'] # Shape: (n_vars, n_comps)
        var_names = pca_result.var_names

        # Check if loadings matrix is empty or var_names are empty
        if loadings is None or loadings.shape[0] == 0 or loadings.shape[1] == 0 or var_names.empty:
            st.warning("PCA loadings data is empty or invalid.")
        else:
            num_pcs = loadings.shape[1]
            pc_options = [f'PC{i+1}' for i in range(num_pcs)]

            # Dropdown to select PC
            selected_pc_label = st.selectbox(
                "Select Principal Component to view loadings:",
                options=pc_options,
                key="pc_loading_select"
            )

            if selected_pc_label:
                fig_loadings_dist = None
                try:
                    # Get the index of the selected PC (PC1 is index 0, etc.)
                    pc_index = pc_options.index(selected_pc_label)

                    # Create a DataFrame with gene names and loadings for the selected PC
                    loadings_df = pd.DataFrame({
                        'Gene': var_names,
                        'Loading': loadings[:, pc_index]
                    })

                    # Sort by loading value
                    loadings_df_sorted = loadings_df.sort_values(by='Loading', ascending=False)
                    plot_df = loadings_df_sorted.copy() # Keep a copy for plotting
                    # Add a rank column for the x-axis of the plot
                    plot_df['Rank'] = range(1, len(plot_df) + 1)

                    st.markdown("##### Loadings Distribution Plot")
                    fig_loadings_dist, ax_loadings_dist = plt.subplots(figsize=(8, 4)) # Adjust size
                    ax_loadings_dist.scatter(
                        plot_df['Rank'],
                        plot_df['Loading'],
                        alpha=0.6, # Use alpha for potentially overlapping points
                        s=10 # Adjust point size
                    )
                    ax_loadings_dist.axhline(0, color='grey', lw=0.8, linestyle='--') # Add line at zero
                    ax_loadings_dist.set_title(f"Gene Loadings Distribution for {selected_pc_label}")
                    ax_loadings_dist.set_xlabel("Gene Rank (Sorted by Loading)")
                    ax_loadings_dist.set_ylabel("Loading Value")
                    ax_loadings_dist.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig_loadings_dist)

                    # Get top 50 positive loadings
                    top_50 = loadings_df_sorted.head(50).reset_index(drop=True)
                    # Get bottom 50 negative loadings (tail is lowest, so sort ascending for display)
                    bottom_50 = loadings_df_sorted.tail(50).sort_values(by='Loading', ascending=True).reset_index(drop=True)
                    
                    st.write(f"**Loadings for {selected_pc_label}:**")

                    col_top, col_bottom = st.columns(2)

                    with col_top:
                        st.markdown(f"**Top 50 Positive Loadings** (Genes driving {selected_pc_label}+)")
                        st.dataframe(top_50, height=400, use_container_width=True) # Adjust height as needed

                    with col_bottom:
                        st.markdown(f"**Bottom 50 Negative Loadings** (Genes driving {selected_pc_label}-)")
                        st.dataframe(bottom_50, height=400, use_container_width=True) # Adjust height as needed

                    # Optional: Add download buttons for the loadings tables
                    try:
                        fname_top = sanitize_filename(f"pca_loadings_{selected_pc_label}_top50", ".csv")
                        generate_download_button(
                            top_50.to_csv(index=False).encode('utf-8'),
                            filename=fname_top,
                            label=f"Download Top 50 Loadings ({selected_pc_label})",
                            mime="text/csv",
                            key=f"download_top_loadings_{selected_pc_label}"
                        )
                        fname_bottom = sanitize_filename(f"pca_loadings_{selected_pc_label}_bottom50", ".csv")
                        generate_download_button(
                            bottom_50.to_csv(index=False).encode('utf-8'),
                            filename=fname_bottom,
                            label=f"Download Bottom 50 Loadings ({selected_pc_label})",
                            mime="text/csv",
                            key=f"download_bottom_loadings_{selected_pc_label}"
                        )
                    except Exception as dle:
                         st.error(f"An unexpected error occurred during loadings download preparation: {dle}")
                         logger.error(f"Loadings Download button generation error: {dle}", exc_info=True)


                except IndexError:
                    st.error(f"Could not find loadings for {selected_pc_label}. Index out of bounds.")
                    logger.error(f"IndexError accessing loadings for PC index {pc_index}", exc_info=True)
                except Exception as e:
                    st.error(f"An error occurred while processing loadings: {e}")
                    logger.error(f"Error processing loadings for {selected_pc_label}: {e}", exc_info=True)

    else:
        st.warning("PCA loadings (components) information not found in `adata.varm['PCs']`.")
    # --- End of section ---
