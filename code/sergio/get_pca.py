#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:06:12 2022

- Load in adata (cell x gene matrix)
- Preprocess
- Do PCA on a single type of cell

@author: tbury
"""

import numpy as np
import pandas as pd
import scanpy as sc

# Read in cell-gene matrix
df = pd.read_csv("../../data/cells_genes_sergio.csv", header=None)

# Read in pseudotime and append to df
df_pseudo = pd.read_csv("../../data/pseudotime_sergio.csv")
df["pseudotime"] = df_pseudo["dpt_pseudotime"]

# Read in cluster names
df_cluster = pd.read_csv("../../data/clusters_sergio.csv")

# Extract section of cell-gene matrix where tipping trajectory is
tstart = 0.32
tend = 0.70
df_tipping = df.query("pseudotime >= @tstart and pseudotime <= @tend")

# Put into adata format
adata_full = sc.AnnData(X=df.drop("pseudotime", axis=1), dtype=np.float32)
adata_tipping = sc.AnnData(X=df_tipping.drop("pseudotime", axis=1), dtype=np.float32)

# Normalise
sc.pp.normalize_total(adata_full, target_sum=1e4)  # Normalize counts per cell
sc.pp.log1p(adata_full)  # Map x->log(x+1)

sc.pp.normalize_total(adata_tipping, target_sum=1e4)  # Normalize counts per cell
sc.pp.log1p(adata_tipping)  # Map x->log(x+1)

# Compute PCA components and basis on tipping section
sc.tl.pca(adata_tipping, n_comps=50)

# Get transformed data using PCA basis from tipping section
X = adata_full.X
w = adata_tipping.varm["PCs"]
X_pca = np.matmul(X, w)

# Dataframe of first 10 PCA comps with pseudotime
df_pca = pd.DataFrame(X_pca[:, :10])
df_pca["pseudotime"] = df["pseudotime"].values
df_pca["cluster"] = df_cluster["leiden"].values
df_pca = df_pca.sort_values("pseudotime")
df_pca["pseudotime_index"] = np.arange(len(df_pca))
df_pca = df_pca.melt(
    id_vars=["pseudotime", "pseudotime_index", "cluster"],
    var_name="pca_comp",
    value_name="value",
)

# Export data of PCA components over pseudotime
df_pca.to_csv("output/df_pca.csv", index=False)
