#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17 Feb 2022

Visualize training data

@author: tbury
"""

import time

import numpy as np
import pandas as pd
import plotly.express as px
import ewstools

df = pd.read_parquet("output_def/df_train.parquet")

# Plot fold forced
df_plot = df[df["bif_type"] == "fold_forced"]
tsid_vals = np.random.choice(df_plot["tsid"].unique(), size=10)
df_plot = df_plot[df_plot["tsid"].isin(tsid_vals)]
fig = px.line(df_plot, x="time", y="x", facet_col="tsid", facet_col_wrap=4)
fig.update_layout(height=2000)
fig.write_html("figures/fig_training_fold_forced.html")

# Plot fold null
df_plot = df[df["bif_type"] == "fold_null"].iloc[: 10 * 500]
fig = px.line(df_plot, x="time", y="x", facet_col="tsid", facet_col_wrap=4)
fig.update_layout(height=2000)
fig.write_html("figures/fig_training_fold_null.html")

# Plot tc forced
df_plot = df[df["bif_type"] == "tc_forced"].iloc[: 10 * 500]
fig = px.line(df_plot, x="time", y="x", facet_col="tsid", facet_col_wrap=4)
fig.update_layout(height=2000)
fig.write_html("figures/fig_training_tc_forced.html")

# Plot tc_null
df_plot = df[df["bif_type"] == "tc_null"].iloc[: 10 * 500]
fig = px.line(df_plot, x="time", y="x", facet_col="tsid", facet_col_wrap=4)
fig.update_layout(height=2000)
fig.write_html("figures/fig_training_tc_null.html")

# Plot pf_forced
df_plot = df[df["bif_type"] == "pf_forced"].iloc[: 10 * 500]
fig = px.line(df_plot, x="time", y="x", facet_col="tsid", facet_col_wrap=4)
fig.update_layout(height=2000)
fig.write_html("figures/fig_training_pf_forced.html")

# Plot pf_null
df_plot = df[df["bif_type"] == "pf_null"].iloc[: 10 * 500]
fig = px.line(df_plot, x="time", y="x", facet_col="tsid", facet_col_wrap=4)
fig.update_layout(height=2000)
fig.write_html("figures/fig_training_pf_null.html")


# Compute EWS
s = df[df["tsid"] == 1].set_index("time")["x"]
ts = ewstools.TimeSeries(s)
ts.compute_var()
ts.compute_auto()
# ts.compute_entropy()
fig = ts.make_plotly()
fig.write_html("figures/fig_ews_fold_forced.html")
