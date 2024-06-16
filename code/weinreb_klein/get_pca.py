#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:10:43 2023

Get PCA components in Weinreb-Klein data 

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
# Rescale pseudotime values to between 0 and 1
df_pseudo["pseudotime"] = df_pseudo["pseudotime"] / df_pseudo["pseudotime"].max()

df_cluster = pd.read_csv("../../data/clusters_wk.txt", sep="\t")
df_cluster["Cell index"] = df_cluster.index

df = df_cell_gene.merge(df_pseudo, on="Cell index")
df = df.merge(df_cluster[["Cell index", "Cell type annotation"]], on="Cell index")

# Compute PCA
sc.tl.pca(adata, n_comps=10)
X_pca = adata.obsm["X_pca"]
W = adata.varm["PCs"]  # weight matrix (eigenvectors in gene space)
exp_var = adata.uns["pca"]["variance"]  # explained variance for each PCA comp
print("Explained varaiance in top PCA components")
print(exp_var)

# Export top PCA components
df_pca = pd.DataFrame(X_pca[:, :10])
df_pca.to_csv("output/df_pca_10.csv", index=False)

# Get gene names of most expressed genes in first PCA comp

# Find most expressed genes in first PCA comp
n_top_genes = 10
v = W[:, 0]
sorted_args = abs(v).argsort()[
    ::-1
]  # use absolute value of v as can have negative values
top_genes = sorted_args[:n_top_genes]
top_gene_names = adata.var_names[top_genes].values
print("Most expressed genes in first PCA comp")
print(top_gene_names)
