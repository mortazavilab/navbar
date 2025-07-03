# navbar/config.py

import logging

# --- General Configuration ---
MAX_CELLS = 20000  # Max cells for visualization after subsampling
DEFAULT_N_MARKERS = 5 # Default number of markers to show/calculate
DEFAULT_PCA_COMPONENTS = 50 # Default max PCA components
MIN_DESEQ_COUNT_SUM = 10 # Min total count for a gene across samples for DESeq2
N_HVG_PSEUDOBULK = 3000 # Number of HVGs to use for pseudobulk analysis

# --- UI Defaults / Choices ---
AGGREGATION_LAYER_OPTIONS = ["Auto-Select", "adata.X", "adata.raw.X"] # Base options
MARKER_METHODS = ['wilcoxon', 't-test', 'logreg']
PLOT_FILE_FORMATS = ["png", "pdf", "svg"]
DEFAULT_PLOT_FORMAT = "png"
DEFAULT_PLOT_DPI = 150
SAVE_PLOT_DPI = 300
DEFAULT_EMBEDDING = 'X_pca' # Default embedding to look for

# --- Logging ---
LOGGING_LEVEL = "WARNING" # e.g., DEBUG, INFO, WARNING, ERROR
# Use date in filenames?
USE_DATE_IN_FILENAMES = False # Set to True to include date in download filenames

# --- Caching ---
CACHE_MAX_ENTRIES = 5 # Max entries for Streamlit memory cache
