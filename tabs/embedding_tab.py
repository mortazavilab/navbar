# navbar/tabs/embedding_tab.py

import streamlit as st
import scanpy as sc
import matplotlib.pyplot as plt
import logging
import anndata as ad # Import AnnData
from utils import generate_download_button, get_figure_bytes, sanitize_filename, PlottingError
from config import DEFAULT_PLOT_FORMAT, SAVE_PLOT_DPI

logger = logging.getLogger(__name__)

def render_embedding_tab(adata_vis, selected_embedding, selected_color_var):
    """Renders the content for the Embedding Plot tab."""
    if not isinstance(adata_vis, ad.AnnData):
         st.error("Invalid data provided to embedding tab.")
         return

    st.subheader(f"Embedding ({selected_embedding}) colored by `{selected_color_var}`")

    if not selected_embedding or not selected_color_var:
        st.warning("Please select an embedding and a categorical variable for coloring in the sidebar.")
        return

    if selected_embedding not in adata_vis.obsm:
        st.error(f"Selected embedding '{selected_embedding}' not found in `adata.obsm` keys: {list(adata_vis.obsm.keys())}")
        return
    if selected_color_var not in adata_vis.obs.columns:
        st.error(f"Selected coloring variable '{selected_color_var}' not found in `adata.obs` columns: {adata_vis.obs.columns.tolist()}")
        return

    # Determine legend location based on number of categories
    try:
        num_categories = adata_vis.obs[selected_color_var].nunique()
        if num_categories < 15:
            legend_loc = 'on data'
            legend_fontsize = 6
        else:
            legend_loc = 'right margin'
            legend_fontsize = 5 # Smaller font for more categories
    except Exception as e:
        logger.warning(f"Could not determine number of categories for legend placement: {e}")
        legend_loc = 'right margin' # Default fallback
        legend_fontsize = 6

    # Generate plot
    fig_embedding = None # Initialize figure variable
    try:
        logger.info(f"Plotting embedding '{selected_embedding}' colored by '{selected_color_var}'")
        fig_embedding, ax_embedding = plt.subplots(figsize=(7, 6)) # Adjust figsize as needed
        sc.pl.embedding(
            adata_vis,
            basis=selected_embedding,
            color=selected_color_var,
            ax=ax_embedding,
            show=False, # Important: Don't show the plot directly
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            title=f"{selected_embedding} | {selected_color_var}", # Add title
            size=20 # Adjust point size if needed (default can be small)
        )
        st.pyplot(fig_embedding)
        logger.info("Embedding plot displayed.")

        # Download button
        try:
            fname_base = f"embedding_{selected_embedding}_{selected_color_var}"
            fname = sanitize_filename(fname_base, extension=DEFAULT_PLOT_FORMAT)
            img_bytes = get_figure_bytes(fig_embedding, format=DEFAULT_PLOT_FORMAT, dpi=SAVE_PLOT_DPI)
            generate_download_button(
                img_bytes,
                filename=fname,
                label=f"Download Plot ({DEFAULT_PLOT_FORMAT.upper()})",
                mime=f"image/{DEFAULT_PLOT_FORMAT}",
                key=f"download_embedding_{selected_embedding}_{selected_color_var}" # Unique key
            )
        # Catch specific plotting errors vs general errors
        except PlottingError as pe:
             st.error(f"Error preparing plot for download: {pe}")
             logger.error(f"Plot download preparation failed: {pe}", exc_info=True)
        except Exception as dle:
             st.error(f"An unexpected error occurred during plot download preparation: {dle}")
             logger.error(f"Download button generation error: {dle}", exc_info=True)

    except KeyError as ke:
         st.error(f"Plotting Error: Variable '{ke}' not found in AnnData object.")
         logger.error(f"Embedding plot KeyError: {ke}", exc_info=True)
         if fig_embedding: plt.close(fig_embedding) # Close figure if created before error
    except ValueError as ve:
         st.error(f"Plotting Error: {ve}. This might be due to incompatible data types or plot parameters.")
         logger.error(f"Embedding plot ValueError: {ve}", exc_info=True)
         if fig_embedding: plt.close(fig_embedding)
    except Exception as e:
        st.error(f"An unexpected error occurred during embedding plotting: {e}")
        logger.error(f"Embedding plot failed: {e}", exc_info=True)
        if fig_embedding: plt.close(fig_embedding) # Ensure figure is closed on error