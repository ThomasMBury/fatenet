#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:10:43 2023

Get PCA components in Weinreb-Klein data subject to gene knockout and over-expression

@author: tbury
"""

import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)

import anndata as ad
import scipy.sparse as sp

# Import HVG data (top 3000 most variable genes)
filename = "../../data/adata_hvg.h5ad"
adata = sc.read_h5ad(filename)

# Get cell-gene matrix
df_cell_gene = pd.DataFrame.sparse.from_spmatrix(adata.X)
df_cell_gene.index.name = "Cell index"
df_cell_gene = df_cell_gene.reset_index()

# Import pseudotime values and cluster names and merge into cell-gene matrix
# Only using values for clusters of undiff and neutrophil cells - reduced size of dataset
df_pseudo = pd.read_csv("../../data/pseudotime_wk.txt", sep="\t")
df_cluster = pd.read_csv("../../data/clusters_wk.txt", sep="\t")
df_cluster["Cell index"] = df_cluster.index

df = df_cell_gene.merge(df_pseudo, on="Cell index")
df = df.merge(df_cluster[["Cell index", "Cell type annotation"]], on="Cell index")

# Compute PCA
sc.tl.pca(adata, n_comps=10)
X_pca = adata.obsm["X_pca"]
W = adata.varm["PCs"]  # weight matrix (eigenvectors in gene space)


# -------
# Knockout genes
# -------


def knockout_genes(n_genes_knockout):
    """
    Find most expressed genes in first PCA comp.
    Knockout top n_genes_knockout by setting their expression to zero in cell-gene matrix
    Recompute PCA and return top PCA components

    """
    # Find most expressed genes in first PCA comp
    v = W[:, 0]
    sorted_args = abs(v).argsort()[
        ::-1
    ]  # use absolute value of v as can have negative values
    top_genes = sorted_args[:n_genes_knockout]
    top_gene_names = adata.var_names[top_genes].values
    print("Top genes")
    print(top_gene_names)

    # Make node mask where indices of zeroed columns are marked as True
    node_mask = np.zeros(W.shape[0])
    node_mask[top_genes] = 1
    node_mask = node_mask.astype("bool")

    # Knockout top genes in cell-gene matrix (set to zero)
    print("Knockout top genes")
    X_knockout = adata.X.copy()
    X_knockout = X_knockout @ sp.diags((~node_mask).astype(int))

    # Do new PCA transformation
    print("Do PCA transformation")
    adata_knockout = ad.AnnData(X_knockout, dtype=X_knockout.dtype)
    sc.tl.pca(adata_knockout, n_comps=10)
    pca_top_comp = adata_knockout.obsm["X_pca"]
    df_pca_knockout = pd.DataFrame(pca_top_comp[:, 0])

    return df_pca_knockout


for n_genes_knockout in [5, 10, 20, 30, 40, 50]:
    # for n_genes_knockout in [0]:
    df_pca_knockout = knockout_genes(n_genes_knockout)
    # Export PCA knockout
    df_pca_knockout.to_csv(
        "output/df_pca_knockout_{}.csv".format(n_genes_knockout), index=False
    )
    print("PCA computed for n_genes_knockout = {}".format(n_genes_knockout))


# -------
# Over-express genes
# -------


def overexpress_genes(n_genes_express):
    """
    Find most expressed genes in first PCA comp.
    Overexpress top n_genes_knockout by setting their multiplying their values
    by exp_multiplier in the cell-gene matrix.
    Recompute PCA and return top PCA components

    """

    exp_multiplier = 2

    # Find most expressed genes in first PCA comp
    v = W[:, 0]
    sorted_args = abs(v).argsort()[
        ::-1
    ]  # use absolute value of v as can have negative values
    top_genes = sorted_args[:n_genes_express]

    # Make node mask where indices of overexpressed columns are marked as True
    node_mask = np.ones(W.shape[0])
    node_mask[top_genes] = exp_multiplier

    # Express top genes in cell-gene matrix (set to zero)
    print("Express top genes")
    X_express = adata.X.copy()
    X_express = X_express @ sp.diags((node_mask).astype(int))

    # Do new PCA transformation
    print("Do PCA transformation")
    adata_express = ad.AnnData(X_express, dtype=X_express.dtype)
    sc.tl.pca(adata_express, n_comps=10)
    pca_top_comp = adata_express.obsm["X_pca"]
    df_pca_overexpress = pd.DataFrame(pca_top_comp[:, 0])

    return df_pca_overexpress


for n_genes_express in [0, 5, 10, 20, 30, 40, 50]:

    pca_top = overexpress_genes(n_genes_express)
    df_pca_overexpress = overexpress_genes(n_genes_express)
    # Export PCA knockout
    df_pca_overexpress.to_csv(
        "output/df_pca_overexpress_{}.csv".format(n_genes_express), index=False
    )
    print("PCA computed for n_genes_express = {}".format(n_genes_express))
