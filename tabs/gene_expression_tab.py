# scanpy_viewer/tabs/gene_expression_tab.py

import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import logging
import anndata as ad # Import AnnData
from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI

logger = logging.getLogger(__name__)

def render_gene_expression_tab(adata_vis, selected_embedding):
    """Renders the content for the Gene Expression tab."""
    if not isinstance(adata_vis, ad.AnnData):
         st.error("Invalid data provided to gene expression tab.")
         return

    st.subheader(f"Gene Expression on Embedding ({selected_embedding})")

    if not selected_embedding:
        st.warning("Please select an embedding in the sidebar.")
        return

    if selected_embedding not in adata_vis.obsm:
        st.error(f"Selected embedding '{selected_embedding}' not found in `adata.obsm` keys: {list(adata_vis.obsm.keys())}")
        return

    gene_input = st.text_input("Enter Gene Name(s) (comma-separated):", key="gene_expr_input")
    selected_genes_raw = [g.strip() for g in gene_input.split(',') if g.strip()]

    # Validate genes against adata_vis.var_names
    valid_genes = []
    invalid_genes = []
    if selected_genes_raw:
        gene_list_adata = adata_vis.var_names.tolist()
        for gene in selected_genes_raw:
            if gene in gene_list_adata:
                valid_genes.append(gene)
            else:
                invalid_genes.append(gene)

    if invalid_genes:
        st.warning(f"Genes not found in data's `adata.var_names`: {', '.join(invalid_genes)}")

    if valid_genes:
        n_genes_plot = len(valid_genes)
        # Adjust layout: create a grid
        max_cols = 3
        n_rows = (n_genes_plot + max_cols - 1) // max_cols
        n_cols = min(n_genes_plot, max_cols)

        fig_gene = None # Initialize figure variable
        try:
            logger.info(f"Plotting expression for genes: {valid_genes} on embedding '{selected_embedding}'")
            fig_gene, axs_gene = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
            axs_flat = axs_gene.flatten() # Flatten the axes array for easy iteration

            plot_kwargs = {
                'basis': selected_embedding,
                'show': False, # Important
                'color_map': 'viridis', # Or another map like 'plasma'
                'legend_loc': None, # Usually no legend needed for continuous expression
                'size': 15 # Adjust point size
            }

            for i, gene in enumerate(valid_genes):
                ax = axs_flat[i]
                # Use sc.pl.embedding - check data source (.X or .raw.X if applicable?)
                # Assuming gene expression is in adata_vis.X unless specified otherwise
                sc.pl.embedding(adata_vis, color=gene, ax=ax, title=gene, **plot_kwargs)


            # Hide unused subplots if any
            for j in range(i + 1, len(axs_flat)):
                axs_flat[j].set_visible(False)

            plt.tight_layout(pad=0.5) # Add padding
            st.pyplot(fig_gene)
            logger.info("Gene expression plot displayed.")

            # Download button for the composite figure
            try:
                gene_fname_part = sanitize_filename('_'.join(valid_genes))
                fname_base = f"gene_expr_{selected_embedding}_{gene_fname_part}"
                fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
                img_bytes = get_figure_bytes(fig_gene, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
                generate_download_button(
                    img_bytes,
                    filename=fname,
                    label=f"Download Plot(s) ({DEFAULT_PLOT_FORMAT.upper()})",
                    mime=f"image/{DEFAULT_PLOT_FORMAT}",
                    key=f"download_gene_expr_{selected_embedding}_{gene_fname_part}"
                )
            except PlottingError as pe:
                st.error(f"Error preparing plot for download: {pe}")
                logger.error(f"Gene expression plot download prep failed: {pe}", exc_info=True)
            except Exception as dle:
                st.error(f"An unexpected error occurred during plot download preparation: {dle}")
                logger.error(f"Download button generation error: {dle}", exc_info=True)


        except KeyError as ke:
            # This could happen if a gene name is somehow invalid despite checks
            st.error(f"Plotting Error: Gene '{ke}' caused an issue.")
            logger.error(f"Gene expression plot KeyError: {ke}", exc_info=True)
            if fig_gene: plt.close(fig_gene)
        except ValueError as ve:
            st.error(f"Plotting Error: {ve}. Check data range or plot parameters.")
            logger.error(f"Gene expression plot ValueError: {ve}", exc_info=True)
            if fig_gene: plt.close(fig_gene)
        except Exception as e:
            st.error(f"An unexpected error occurred during gene expression plotting: {e}")
            logger.error(f"Gene expression plot failed: {e}", exc_info=True)
            if fig_gene: plt.close(fig_gene) # Ensure figure is closed

    elif gene_input and not valid_genes:
        # If user entered text but none were valid
        st.warning("None of the entered genes were found in the dataset's `var_names`.")
    elif not gene_input:
        st.info("Enter gene names above to plot their expression on the selected embedding.")
