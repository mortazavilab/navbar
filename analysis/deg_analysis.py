# navbar/analysis/deg_analysis.py
import pandas as pd
import numpy as np
import scipy.sparse
import logging
import anndata as ad
from collections import OrderedDict

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    PYDESEQ2_INSTALLED = True
except ImportError:
    PYDESEQ2_INSTALLED = False
    class DeseqDataSet: pass
    class DeseqStats: pass

from utils import AnalysisError, FactorNotFoundError, create_group_string

logger = logging.getLogger(__name__)

def extract_counts_matrix(adata_agg, possible_layers=["sum", "counts"]):
    # Prefer 'sum' or 'counts' layers if available, else fallback to .X
    for layer in possible_layers:
        if layer in adata_agg.layers:
            return adata_agg.layers[layer]
    return adata_agg.X

def prepare_metadata_and_run_deseq(adata_agg, comparison_factors, replicate_factor,
                                   group1_levels, group2_levels, min_count_sum_filter=10,
                                   min_nonzero_samples=0):
    if not PYDESEQ2_INSTALLED:
        raise ImportError("pydeseq2 is not installed. Please install it.")

    if adata_agg is None or not isinstance(adata_agg, ad.AnnData):
        raise AnalysisError("Input must be an AnnData object.")
    if adata_agg.obs.empty or adata_agg.X is None:
        raise AnalysisError("Aggregated AnnData missing obs or X.")

    metadata = adata_agg.obs.copy()
    # Only one comparison factor: e.g. "Genotype"
    comp_factor = comparison_factors[0]
    if comp_factor not in metadata.columns or replicate_factor not in metadata.columns:
        raise FactorNotFoundError(
            f"'{comp_factor}' and/or '{replicate_factor}' not found in aggregated metadata. Found: {list(metadata.columns)}"
        )

    # Set to categorical/string
    metadata[comp_factor] = metadata[comp_factor].astype(str)
    metadata[replicate_factor] = metadata[replicate_factor].astype(str)
    metadata['comparison_group'] = metadata[comp_factor].astype(str)

    # Ensure group strings exist
    groups = metadata['comparison_group'].unique().tolist()
    g1, g2 = group1_levels[comp_factor], group2_levels[comp_factor]
    if g1 not in groups or g2 not in groups:
        raise ValueError(f"Chosen groups '{g1}' or '{g2}' not present in aggregated obs. Found: {groups}")

    # Ensure enough replicates per group
    rep_counts = metadata.groupby('comparison_group')[replicate_factor].nunique()
    low_rep = rep_counts[rep_counts < 2]
    if not low_rep.empty:
        logger.warning(f"Some groups have < 2 replicates: {dict(low_rep)}")

    # Full-rank check
    design_df = pd.get_dummies(metadata[['comparison_group', replicate_factor]], drop_first=True)
    if np.linalg.matrix_rank(design_df.values) < design_df.shape[1]:
        raise AnalysisError(
            "The design matrix for DESeq2 is not full rank (comparison and replicate are not independent for these samples)."
        )

    design_factors_list = [replicate_factor, 'comparison_group']
    design_formula = f"~ {replicate_factor} + comparison_group"
    logger.info(f"Using DESeq2 design formula: {design_formula}")

    # Prepare counts
    matrix = extract_counts_matrix(adata_agg)
    counts_df = pd.DataFrame(
        matrix.toarray() if scipy.sparse.issparse(matrix) else matrix,
        index=adata_agg.obs_names, columns=adata_agg.var_names
    )
    
    is_integer = np.all(np.equal(np.mod(counts_df.values, 1), 0))
    if not is_integer:
        logger.warning("Counts matrix has non-integer values. Rounding to int.")
        counts_df = counts_df.round(0).astype(int)
    if (counts_df < 0).any().any():
        raise AnalysisError("Counts matrix has negative values.")

    # Gene filtering
    if min_count_sum_filter > 0:
        sums = counts_df.sum(axis=0)
        counts_df = counts_df.loc[:, sums >= min_count_sum_filter]
    if min_nonzero_samples > 0:
        nonzero_counts = (counts_df > 0).sum(axis=0)
        counts_df = counts_df.loc[:, nonzero_counts >= min_nonzero_samples]
    if counts_df.shape[1] < 1:
        raise AnalysisError("No genes pass filtering.")

    # Align metadata
    metadata = metadata.loc[counts_df.index]

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design= "~ comparison_group",
        refit_cooks=True,
        quiet=False
    )
    logger.info(f"Running DESeq2 analysis (dds.deseq2()) using formula: {design_formula}...")
    dds.deseq2()

    contrast_list = ["comparison_group", g1, g2]
    stat_res = DeseqStats(dds, contrast=contrast_list, quiet=False)
    stat_res.summary()
    results_df = stat_res.results_df
    if results_df is None or results_df.empty:
        results_df = pd.DataFrame()
    else:
        results_df = results_df.sort_values("padj")
        # Merge gene symbols if present
        if 'gene_symbols' in adata_agg.var.columns:
            results_df = results_df.merge(
                adata_agg.var[['gene_symbols']], left_index=True, right_index=True, how='left'
            )
    del dds, stat_res
    return results_df
