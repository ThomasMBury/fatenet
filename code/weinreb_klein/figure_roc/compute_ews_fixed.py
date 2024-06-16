#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:15:44 2022

Compute EWS at fixed point(s) in time over all possible samples of WK data.
Compute 
    - DL prediction
    - Variance
    - Lag-1 AC 
    - Entropy

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

np.random.seed(0)


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
df_pca = pd.read_csv("../output/df_pca_10.csv")
df_pca["Cell index"] = df_pca.index
df_pseudo = pd.read_csv("../../../data/pseudotime_wk.txt", sep="\t")
# Rescale pseudotime values to between 0 and 1
df_pseudo["pseudotime"] = df_pseudo["pseudotime"] / df_pseudo["pseudotime"].max()

# Import cluster names and merge
df_cluster = pd.read_csv("../../../data/clusters_wk.txt", sep="\t")
df_cluster["Cell index"] = df_cluster.index
df = df_pseudo.merge(df_pca, on="Cell index")
df = df.merge(df_cluster[["Cell index", "Cell type annotation"]], on="Cell index")
df.sort_values(by="pseudotime", inplace=True)

# Only 1 baso entry- remove it
df = df.drop(53210, axis=0)
df["Cell type annotation"].value_counts()

# Consider only undif cells for EWS
df_undiff = df[df["Cell type annotation"] == "Undifferentiated"]

# -------------
# Compute EWS for forced trajectory
# -------------

## Compute EWS for specific class and PCA comp
span = 0.2  # span of Lowess Filter

# list_sample_numbers = np.random.choice(np.arange(100), 10, replace=False)
list_sample_numbers = np.arange(100)

# Load models
print("Load DL classifiers")
path = "../../dl_train/models/"

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
eval_times = np.arange(0.3, 0.601, 0.05)

pca_comp = "0"
df_select = df[["pseudotime", pca_comp]].copy()

# Remove outliers
df_select[pca_comp] = zscore(df_select[pca_comp], window=50)

list_df = []
list_ktau = []

for sample_number in list_sample_numbers:
    # Downsample the data
    df_down = df_select.iloc[sample_number::100]
    series = df_down.set_index("pseudotime")[pca_comp]

    pre_tran_len = len(series[series.index < transition])
    start_idx = max(pre_tran_len - 499, 0)
    s = series.iloc[start_idx:]

    for eval_time in eval_times:
        # Compute EWS for forced trajectory
        ts = ewstools.TimeSeries(s, transition=transition)
        ts.detrend(method="Lowess", span=span)
        # ts.detrend(method='Gaussian', bandwidth=0.2)

        # Compute conventional indiciators
        ts.compute_var(rolling_window=0.25)
        ts.compute_auto(rolling_window=0.25, lag=1)
        ts.compute_entropy(rolling_window=0.25, method="sample")
        ts.compute_entropy(rolling_window=0.25, method="kolmogorov")
        # Compute kendall tau up to eval time
        ts.compute_ktau(tmax=eval_time)
        dict_ktau = ts.ktau
        dict_ktau["sample_number"] = sample_number
        dict_ktau["eval_time"] = eval_time
        dict_ktau["type"] = "forced"
        list_ktau.append(dict_ktau)

        # Apply classifiers
        for idx, m1 in enumerate(list_models_1):
            ts.apply_classifier(m1, tmin=0, tmax=eval_time, name=f"m1_{idx}", verbose=0)
        for idx, m2 in enumerate(list_models_2):
            ts.apply_classifier(m2, tmin=0, tmax=eval_time, name=f"m2_{idx}", verbose=0)

        df_dl = ts.dl_preds
        df_dl["sample_number"] = sample_number
        df_dl["eval_time"] = eval_time
        df_dl["type"] = "forced"
        list_df.append(df_dl)

        # ---------------
        # Compute EWS for a null
        # Sample randomly from first 20% of residuals and add to trend of raw time series
        # ---------------

        # Get pre-transition series
        series_pre = s.iloc[:pre_tran_len]
        ts = ewstools.TimeSeries(series_pre)
        ts.detrend(method="Lowess", span=span)
        resids = ts.state["residuals"]
        resids_sample = resids.iloc[: int(len(series_pre) * 0.2)].sample(
            n=len(series_pre), replace=True, ignore_index=True
        )
        series_null = ts.state["smoothing"] + resids_sample.values

        ts = ewstools.TimeSeries(series_null)
        ts.detrend(method="Lowess", span=0.2)

        # Compute conventional indiciators
        ts.compute_var(rolling_window=0.25)
        ts.compute_auto(rolling_window=0.25, lag=1)
        ts.compute_entropy(rolling_window=0.25, method="sample")
        ts.compute_entropy(rolling_window=0.25, method="kolmogorov")
        # Compute kendall tau up to eval time
        ts.compute_ktau(tmax=eval_time)
        dict_ktau = ts.ktau
        dict_ktau["sample_number"] = sample_number
        dict_ktau["eval_time"] = eval_time
        dict_ktau["type"] = "neutral"
        list_ktau.append(dict_ktau)

        # Apply classifiers
        for idx, m1 in enumerate(list_models_1):
            ts.apply_classifier(m1, tmin=0, tmax=eval_time, name=f"m1_{idx}", verbose=0)
        for idx, m2 in enumerate(list_models_2):
            ts.apply_classifier(m2, tmin=0, tmax=eval_time, name=f"m2_{idx}", verbose=0)

        df_dl = ts.dl_preds
        df_dl["sample_number"] = sample_number
        df_dl["eval_time"] = eval_time
        df_dl["type"] = "neutral"
        list_df.append(df_dl)

    print("EWS computed for sample number {}".format(sample_number))

df_dl = pd.concat(list_df).reset_index(drop=True)
df_ktau = pd.DataFrame(list_ktau)

# Average over classifiers
df_dl = df_dl.groupby(["sample_number", "eval_time", "type"]).mean(numeric_only=True)
df_dl["any_bif"] = df_dl[1] + df_dl[2] + df_dl[3]

# Export predictions
df_dl.to_csv("output/df_dl_{}.csv".format(len(list_sample_numbers)))
df_ktau.to_csv("output/df_ktau_{}.csv".format(len(list_sample_numbers)), index=False)
