#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:15:44 2022

Compute EWS in WK data

@author: tbury
"""

import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

import ewstools
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model

import sklearn.metrics as metrics
import scipy.stats as stats


def zscore(s, window, thresh=2, return_all=False):
    roll = s.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)
    z = s.sub(avg).div(std)
    m = z.between(-thresh, thresh)

    if return_all:
        return z, avg, std, m
    return s.where(m, avg)


# Read in PCA data and pseudotime
df_pca = pd.read_csv("output/df_pca_10.csv")
df_pca["Cell index"] = df_pca.index
df_pseudo = pd.read_csv("../../data/pseudotime_wk.txt", sep="\t")
# Rescale pseudotime values to between 0 and 1
df_pseudo["pseudotime"] = df_pseudo["pseudotime"] / df_pseudo["pseudotime"].max()


# Import cluster names and merge
df_cluster = pd.read_csv("../../data/clusters_wk.txt", sep="\t")
df_cluster["Cell index"] = df_cluster.index
df = df_pseudo.merge(df_pca, on="Cell index")
df = df.merge(df_cluster[["Cell index", "Cell type annotation"]], on="Cell index")
df.sort_values(by="pseudotime", inplace=True)

# Only 1 baso entry- remove it
df = df.drop(53210, axis=0)
df["Cell type annotation"].value_counts()

# Export trajectory data
df[["0", "pseudotime", "Cell type annotation"]].to_csv("output/df_traj_forced.csv")

# Consider only undif cells for EWS
df_undiff = df[df["Cell type annotation"] == "Undifferentiated"]

# -------------
# Compute EWS for forced trajectory
# -------------

## Compute EWS for specific class and PCA comp
span = 0.2  # span of Lowess Filter

# Load models
print("Load DL classifiers")
path = "../dl_train/models/"

import glob

list_model_names_1 = [f.split("/")[-1] for f in glob.glob(path + "20240612-18*m1*")]
list_model_names_2 = [f.split("/")[-1] for f in glob.glob(path + "20240612-18*m2*")]

list_models_1 = []
list_models_2 = []

for model_name in list_model_names_1:
    m = load_model(path + model_name)
    list_models_1.append(m)

for model_name in list_model_names_2:
    m = load_model(path + model_name)
    list_models_2.append(m)
print("ML models loaded")

# Transition time (observed from data when cell type significantly changes)
transition = 0.6

pca_comp = "0"
remove_outliers = True
df_select = df[["pseudotime", pca_comp, "Cell index"]].copy()

# Remove outliers and plot new trajectory
df_select[pca_comp + "_ro"] = zscore(df_select[pca_comp], window=50)

# Downsample the data
df_down = df_select.iloc[77::100]

if remove_outliers:
    series = df_down.set_index("pseudotime")[pca_comp + "_ro"]
else:
    series = df_down.set_index("pseudotime")[pca_comp]

pre_tran_len = len(series[series.index < transition])
start_idx = max(pre_tran_len - 499, 0)
s = series.iloc[start_idx:]

# Compute EWS
print("Compute EWS for PCA comp {}".format(pca_comp))
# time series increment for DL classifier
inc = (series.index[1] - series.index[0]) * 4

ts = ewstools.TimeSeries(s, transition=transition)
ts.detrend(method="Lowess", span=span)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)

# Apply classifiers
print("Apply classifiers")
for idx, m1 in enumerate(list_models_1):
    ts.apply_classifier_inc(m1, inc=inc, name=f"c1_{idx}", verbose=0)
for idx, m2 in enumerate(list_models_2):
    ts.apply_classifier_inc(m2, inc=inc, name=f"c2_{idx}", verbose=0)

# Export data
df_dl_forced = ts.dl_preds
df_ews_forced = ts.state
df_ews_forced = df_ews_forced.join(ts.ews)
df_dl_forced.to_csv("output/ews/df_dl_forced.csv", index=False)
df_ews_forced.to_csv("output/ews/df_ews_forced.csv")

# Get cell indices during first transition phase of DL probabilities
cell_idx = df_down.query("0.28<=pseudotime<=0.32")["Cell index"].values
pseudo = df_down.query("0.28<=pseudotime<=0.32")["pseudotime"].values
df_cell_info = df_cluster[df_cluster["Cell index"].isin(cell_idx)].copy()
df_cell_info["pseudotime"] = pseudo
df_cell_info.to_csv("output/cell_id_warning.csv", index=False)

# ---------------
# Compute EWS for a null
# Sample randomly from first 20% of residuals and add to trend of raw time series
# ---------------

np.random.seed(8)
pre_tran_len = len(series[series.index < transition])
start_idx = max(pre_tran_len - 499, 0)
s = series.iloc[start_idx:]

# Get pre-transition series
series_pre = s.iloc[:pre_tran_len]
ts = ewstools.TimeSeries(series_pre)
ts.detrend(method="Lowess", span=span)
resids = ts.state["residuals"]
resids_sample = resids.iloc[: int(len(series_pre) * 0.2)].sample(
    n=len(series_pre), replace=True, ignore_index=True
)
series_null = ts.state["smoothing"] + resids_sample.values
series_null.name = "0"
series_null.to_csv("output/df_traj_null.csv")

# time series increment for DL classifier
inc = (series_null.index[1] - series_null.index[0]) * 4

ts = ewstools.TimeSeries(series_null)
ts.detrend(method="Lowess", span=0.2)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)

# Apply classifiers
print("Apply classifiers")
for idx, m1 in enumerate(list_models_1):
    ts.apply_classifier_inc(m1, inc=inc, name=f"c1_{idx}", verbose=0)
for idx, m2 in enumerate(list_models_2):
    ts.apply_classifier_inc(m2, inc=inc, name=f"c2_{idx}", verbose=0)

# Export data
df_ews_null = ts.state
df_ews_null = df_ews_null.join(ts.ews)
df_dl_null = ts.dl_preds
df_dl_null.to_csv("output/ews/df_dl_null.csv", index=False)
df_ews_null.to_csv("output/ews/df_ews_null.csv", index=False)
