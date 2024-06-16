#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:52:37 2022

Compute early warning signals in first PCA component of sergio data

@author: tbury
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

import ewstools

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model


def zscore(s, window, thresh=2, return_all=False):
    roll = s.rolling(window=window, min_periods=1, center=True)
    avg = roll.mean()
    std = roll.std(ddof=0)
    z = s.sub(avg).div(std)
    m = z.between(-thresh, thresh)

    if return_all:
        return z, avg, std, m
    return s.where(m, avg)


# Import PCA data
df = pd.read_csv("output/df_pca.csv")

span = 0.1  # span of Lowess Filter
inc = 2

## Load models
print("Load ML models")
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

cluster = 5
pca_comp = 0
transition = 0.703

# # # ----------------
# # # Compute EWS for raw time series
# # # -----------------

print("Compute EWS in raw")
df_select = df.query("pca_comp==@pca_comp and cluster==@cluster").copy()

# Remove outliers
df_select["value_ro"] = zscore(df_select["value"], window=50)
series = df_select.reset_index()["value_ro"]

# Compute EWS
ts = ewstools.TimeSeries(series)
ts.detrend(method="Lowess", span=span)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)

# Apply classifiers
print("Apply classifiers")
for idx, m1 in enumerate(list_models_1):
    ts.apply_classifier_inc(m1, inc=inc, name=f"c1_{idx}", verbose=0)
for idx, m2 in enumerate(list_models_2):
    ts.apply_classifier_inc(m2, inc=inc, name=f"c2_{idx}", verbose=0)

# Add pseudotime col
df_dl = ts.dl_preds
df_dl["pseudotime_index"] = df_dl["time"]
df_dl["pseudotime"] = df_select["pseudotime"].iloc[df_dl["time"].values].values

df_ews = ts.state
df_ews = df_ews.join(ts.ews).reset_index()
df_ews["pseudotime_index"] = df_ews["time"]
df_ews["pseudotime"] = df_select["pseudotime"].iloc[df_ews["time"].values].values

# Export
df_dl.to_csv("output/ews/df_dl.csv", index=False)
df_ews.to_csv("output/ews/df_ews.csv", index=False)

# ----------------
# Compute EWS for null time series - sample randomly from first 20% of residuals
# and add to trend
# -----------------
print("Compute EWS in null")

np.random.seed(10)

series = df_select.reset_index()["value_ro"]
# series = df_select.reset_index()["value"]

ts = ewstools.TimeSeries(series)
ts.detrend(method="Lowess", span=span)
resids = ts.state["residuals"]
resids_sample = resids.iloc[: int(len(series) * 0.2)].sample(
    n=len(series), replace=True, ignore_index=True
)
series_null = ts.state["smoothing"] + resids_sample
series_null.to_csv("output/df_null.csv")

# Compute EWS
ts = ewstools.TimeSeries(series_null)
ts.detrend(method="Lowess", span=span)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)


# Apply classifiers
print("Apply classifiers")
for idx, m1 in enumerate(list_models_1):
    ts.apply_classifier_inc(m1, inc=inc, name=f"c1_{idx}", verbose=0)
for idx, m2 in enumerate(list_models_2):
    ts.apply_classifier_inc(m2, inc=inc, name=f"c2_{idx}", verbose=0)

# Add pseudotime col
df_dl_null = ts.dl_preds
df_dl_null["pseudotime_index"] = df_dl_null["time"]
df_dl_null["pseudotime"] = (
    df_select["pseudotime"].iloc[df_dl_null["time"].values].values
)

df_ews_null = ts.state
df_ews_null = df_ews_null.join(ts.ews).reset_index()
df_ews_null["pseudotime_index"] = df_ews_null["time"]
df_ews_null["pseudotime"] = (
    df_select["pseudotime"].iloc[df_ews_null["time"].values].values
)

# Export
df_dl_null.to_csv("output/ews/df_dl_null.csv", index=False)
df_ews_null.to_csv("output/ews/df_ews_null.csv", index=False)
