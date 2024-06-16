#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:30:01 2021

- Make fig of performance vs. epoch for val and train sets

@author: tbury
"""


import numpy as np

np.random.seed(0)

import pandas as pd
import ewstools

import os

import plotly.express as px
import plotly.graph_objects as go


# Import data
df1 = pd.read_csv("wandb_download/training_accuracy_ncied.csv")
df2 = pd.read_csv("wandb_download/val_accuracy_ncied.csv")
df3 = pd.read_csv("wandb_download/training_accuracy_yjbfz.csv")
df4 = pd.read_csv("wandb_download/val_accuracy_yjbfz.csv")

df1["value"] = df1["20240522-124437_m1_ncied - epoch/accuracy"]
df1["type"] = "Network 1 training accuracy"

df2["value"] = df2["20240522-124437_m1_ncied - epoch/val_accuracy"]
df2["type"] = "Network 1 validation accuracy"

df3["value"] = df3["20240522-124506_m2_yjbfz - epoch/accuracy"]
df3["type"] = "Network 2 training accuracy"

df4["value"] = df4["20240522-124506_m2_yjbfz - epoch/val_accuracy"]
df4["type"] = "Network 2 validation accuracy"

df_plot = pd.concat([df1, df2, df3, df4])

fig = px.line(df_plot, x="Step", y="value", color="type")

fig.update_legends(title="")
fig.update_xaxes(title="epoch")
fig.update_yaxes(title="accuracy")


fig.write_html("figures/training_performance.html")
