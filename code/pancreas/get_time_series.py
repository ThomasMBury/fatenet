#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:06:12 2022

- Load in adata (cell x gene matrix)
- Preprocess
- Do PCA to extract main components
- Export PCA components

@author: tbury
"""

import numpy as np
import pandas as pd
import scanpy as sc

df_pseudo = pd.read_csv("../../data/pancreas/pancreas_pseudo.csv").set_index("index")

# Import cluster data
df_cluster = pd.read_csv("../../data/pancreas/pancreas_clusters.csv", index_col="index")

adata = sc.read_csv(
    "../../data/pancreas/pancreas_cells_genes.csv",
    delimiter=",",
    first_column_names=True,
    dtype="float32",
)

sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts per cell
sc.pp.log1p(adata)  # Map x->log(x+1)
sc.tl.pca(adata, n_comps=50)  # compute PCA

# Extract time series of PCA comps of genes
# Dataframe of PCA comps with pseudotime
df = pd.DataFrame(adata.obsm["X_pca"][:, :10])
df.index = adata.obs.index
df["pseudotime"] = df_pseudo["dpt_pseudotime"]
df["cluster"] = df_cluster["clusters"]
# df['class'] = adata.obs['leiden'].values
df = df.sort_values("pseudotime")
df["pseudotime_index"] = np.arange(len(df))
# df = df.melt(id_vars = ['pseudotime','class','pseudotime_index'], var_name='PCA comp', value_name='value')
df = df.melt(
    id_vars=["pseudotime", "pseudotime_index", "cluster"],
    var_name="PCA comp",
    value_name="value",
)

# Export data of PCA components over pseudotime
df.to_csv("output/df_pca.csv", index=False)
