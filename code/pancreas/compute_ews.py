#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:52:37 2022

Compute EWS in pancreas data

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


np.random.seed(1)

# Import PCA data
df = pd.read_csv("output/df_pca.csv")

span = 0.2  # span of Lowess Filter

# Load models
print("Load DL classifiers")
path = "../dl_train/models/"

## Load models

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

cell_class = "Fev+"
transition = 0.93
pca_comp = 0

# ----------------
# Compute EWS for transition time series
# -----------------

df_select = (
    df.query(
        "`cluster`==@cell_class and `PCA comp`==@pca_comp and pseudotime<=@transition"
    )
    .reset_index(drop=True)
    .copy()
)

# # Remove outliers
df_select["value_ro"] = zscore(df_select["value"], window=50)
df_select = df_select.iloc[-470:].reset_index(drop=True)
series = df_select["value_ro"]
inc = (series.index[1] - series.index[0]) * 4

# Compute EWS
ts = ewstools.TimeSeries(series)
ts.detrend(method="Lowess", span=span)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)

# Apply classifiers
print("Apply classifiers")
for idx, c1 in enumerate(list_models_1):
    ts.apply_classifier_inc(c1, inc=inc, name=f"c1_{idx}", verbose=0)
for idx, c2 in enumerate(list_models_2):
    ts.apply_classifier_inc(c2, inc=inc, name=f"c2_{idx}", verbose=0)


# Add pseudotime col
df_dl = ts.dl_preds
df_dl["pseudotime_index"] = df_dl["time"]
df_dl["pseudotime"] = df_select["pseudotime"].iloc[df_dl["time"].values].values

df_ews = ts.state
df_ews = df_ews.join(ts.ews).reset_index()
df_ews["pseudotime_index"] = df_ews["time"]
df_ews["pseudotime"] = df_select["pseudotime"].iloc[df_ews["time"].values].values

# ----------------
# Compute EWS for null time series - sample randomly from first 20% of data
# -----------------

# shuffle
series = df_select["value"]
series_shuffled = series.iloc[: int(len(series) * 0.2)].sample(
    n=len(series), replace=True, ignore_index=True
)
series_shuffled.index = series.index

ts = ewstools.TimeSeries(series_shuffled)
ts.detrend(method="Lowess", span=span)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)
print("Apply classifiers")
for idx, c1 in enumerate(list_models_1):
    ts.apply_classifier_inc(c1, inc=inc, name=f"c1_{idx}", verbose=0)
for idx, c2 in enumerate(list_models_2):
    ts.apply_classifier_inc(c2, inc=inc, name=f"c2_{idx}", verbose=0)


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

# Export output data
df_dl.to_csv("output/df_dl_forced.csv", index=False)
df_dl_null.to_csv("output/df_dl_null.csv", index=False)
df_ews.to_csv("output/df_ews_forced.csv", index=False)
df_ews_null.to_csv("output/df_ews_null.csv", index=False)
