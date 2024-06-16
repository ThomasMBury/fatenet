#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:15:44 2022

Compute EWS over data with genes knocked out

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

list_protocols = ["knockout_{}".format(ko) for ko in [0, 5, 10, 20, 30, 40, 50]]
list_df_dl = []
list_df_ews = []


for protocol in list_protocols:
    # Read in PCA data, pseudotime and cluster names
    df_pca = pd.read_csv("output/df_pca_{}.csv".format(protocol))
    df_pca["Cell index"] = df_pca.index
    df_pseudo = pd.read_csv("../../data/pseudotime_wk.txt", sep="\t")
    # Rescale pseudotime values to between 0 and 1
    df_pseudo["pseudotime"] = df_pseudo["pseudotime"] / df_pseudo["pseudotime"].max()

    df_cluster = pd.read_csv("../../data/clusters_wk.txt", sep="\t")
    df_cluster["Cell index"] = df_cluster.index

    # Merge dataframes
    df = df_pseudo.merge(df_pca, on="Cell index")
    df = df.merge(df_cluster[["Cell index", "Cell type annotation"]], on="Cell index")
    df.sort_values(by="pseudotime", inplace=True)

    # Only 1 baso entry- remove it
    df = df.drop(53210, axis=0)
    df["Cell type annotation"].value_counts()

    # -------------
    # Compute EWS for forced trajectory
    # -------------

    ## Compute EWS for specific class and PCA comp
    span = 0.2  # span of Lowess Filter

    # Transition time (observed from data when cell type significantly changes)
    transition = 0.6

    pca_comp = "0"
    remove_outliers = True
    df_select = df[["pseudotime", pca_comp]].copy()

    # Remove outliers
    df_select[pca_comp + "_ro"] = zscore(df_select[pca_comp], window=50)

    # Downsample the data
    df_down = df_select.iloc[::100]

    if remove_outliers:
        series = df_down.set_index("pseudotime")[pca_comp + "_ro"]
    else:
        series = df_down.set_index("pseudotime")[pca_comp]

    pre_tran_len = len(series[series.index < transition])
    start_idx = max(pre_tran_len - 499, 0)
    s = series.iloc[start_idx:]

    # Compute EWS
    # time series increment for DL classifier
    inc = (series.index[1] - series.index[0]) * 2

    ts = ewstools.TimeSeries(s, transition=transition)
    ts.detrend(method="Lowess", span=span)
    # ts.detrend(method='Gaussian', bandwidth=0.2)
    ts.compute_var(rolling_window=0.25)
    ts.compute_auto(rolling_window=0.25, lag=1)

    # Tmax values for each time series segment
    tmax_vals = np.arange(0.4, 0.61, inc)
    for tmax in tmax_vals:

        # Apply classifiers
        for idx, m1 in enumerate(list_models_1):
            ts.apply_classifier(m1, name=f"c1_{idx}", tmin=0, tmax=tmax, verbose=0)
        for idx, m2 in enumerate(list_models_2):
            ts.apply_classifier(m2, name=f"c2_{idx}", tmin=0, tmax=tmax, verbose=0)

    # Collect data
    df_dl_forced = ts.dl_preds
    df_dl_forced["protocol"] = protocol
    df_ews_forced = ts.state
    df_ews_forced = df_ews_forced.join(ts.ews)
    df_ews_forced["protocol"] = protocol

    # Add to lists
    list_df_dl.append(df_dl_forced)
    list_df_ews.append(df_ews_forced)

    print("EWS computed for protocol {}".format(protocol))

# Concat data
df_dl = pd.concat(list_df_dl)
df_ews = pd.concat(list_df_ews)

# Export data
df_dl.to_csv("output/ews/df_dl_knockout.csv", index=False)
df_ews.to_csv("output/ews/df_ews_knockout.csv", index=False)
