#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:30:01 2021

- Get performance metrics for DL classifiers on test set
- Plot confusion matrices

@author: tbury
"""

import numpy as np

np.random.seed(0)

import pandas as pd
import ewstools

import os

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from tensorflow.keras.models import load_model

import time

start = time.time()

filepath_data = "../training_data/output_80k/"

model_name_1 = "20240522-124437_m1_ncied"
model_name_2 = "20240522-124506_m2_yjbfz"


# Function to preprocess training/test data
ts_len = 500


def prepare_series(series, model_type):
    """
    Prepare raw series data for training.

    Parameters:
        series: pd.Series of length ts_len
    """

    # Length of sequence to extract
    L = np.random.choice(np.arange(50, ts_len + 1))

    # Start time of sequence to extract
    if model_type == 1:
        t0 = np.random.choice(np.arange(0, ts_len - L + 1))
    elif model_type == 2:
        t0 = ts_len - L

    seq = series.iloc[t0 : t0 + L]

    # Normalise the sequence by mean of absolute values
    mean = seq.abs().mean()
    seq_norm = seq / mean

    # Prepend with zeros to make sequence of length ts_len
    series_out = pd.concat(
        [pd.Series(np.zeros(ts_len - L)), seq_norm], ignore_index=True
    )

    # Keep original index
    series_out.index = series.index
    return series_out


# -----------
# Confusion matrix for model type 1 - full classificaiton problem
# ------------

# Import model
model_type = 1
model = load_model("models/" + model_name_1)

# Import test data and apply preprocessing
df_test = pd.read_parquet(filepath_data + "df_test.parquet")
# tsid_select = np.random.choice(df_test['tsid'].unique(), 1000, replace=False)
# df_test = df_test.query('tsid in @tsid_select')

ts_pad = df_test.groupby("tsid")["x"].transform(prepare_series, model_type)
df_test["x_pad"] = ts_pad

# Put into numpy array with shape (samples, timesteps, features)
inputs_test = df_test["x_pad"].to_numpy().reshape(-1, ts_len, 1)
targets_test = df_test["type"].iloc[::ts_len].to_numpy().reshape(-1, 1)


# Get predictions
preds_prob = model.predict(inputs_test)  # prediction probabilities
preds = np.argmax(preds_prob, axis=1)  # class prediction (max prob)

# Show performance stats
# print(classification_report(targets_test, preds, digits=3))
print("Performance of model 1\n")
print("F1 score: {:.3f}".format(f1_score(targets_test, preds, average="macro")))
print("Accuracy: {:3f}".format(accuracy_score(targets_test, preds)))
print("Confusion matrix: \n")
print(confusion_matrix(targets_test, preds))

# Make confusion matrix plot
class_names = ["Null", "Fold", "TC", "PF"]
disp = ConfusionMatrixDisplay.from_predictions(
    targets_test, preds, display_labels=class_names, cmap=plt.cm.Blues, normalize="true"
)
ax = disp.ax_
ax.images[0].colorbar.remove()
plt.text(x=-1.3, y=-0.1, s="A", fontdict={"size": 14})
plt.savefig("figures/cm_mtype_{}.png".format(model_type), bbox_inches="tight", dpi=300)


# -----------
# Confusion matrix for model type 1 - binary classificaiton problem
# ------------


def full_to_binary(bif_class):
    if bif_class >= 1:
        out = 1
    else:
        out = 0
    return out


targets_test_binary = np.array(list(map(full_to_binary, targets_test)))
preds_binary = np.array(list(map(full_to_binary, preds)))

# Show performance stats
# print(classification_report(targets_test, preds, digits=3))
print("Performance of model 1\n")
print(
    "F1 score: {:.3f}".format(
        f1_score(targets_test_binary, preds_binary, average="macro")
    )
)
print("Accuracy: {:3f}".format(accuracy_score(targets_test_binary, preds_binary)))
print("Confusion matrix: \n")
print(confusion_matrix(targets_test_binary, preds_binary))

# Make confusion matrix plot
class_names = ["Null", "Bifurcation"]
disp = ConfusionMatrixDisplay.from_predictions(
    targets_test_binary,
    preds_binary,
    display_labels=class_names,
    cmap=plt.cm.Blues,
    normalize="true",
)
ax = disp.ax_
ax.images[0].colorbar.remove()
plt.yticks(rotation=90, va="center")
plt.text(x=-0.9, y=-0.35, s="C", fontdict={"size": 14})
plt.ylabel(ylabel="True label", labelpad=15)
plt.savefig(
    "figures/cm_mtype_{}_binary.png".format(model_type), bbox_inches="tight", dpi=300
)


# -----------
# Confusion matrix for model type 2 - full classificaiton problem
# ------------


# Import model
model_type = 2
model = load_model("models/" + model_name_2)

# Import test data and apply preprocessing
df_test = pd.read_parquet(filepath_data + "df_test.parquet")
# tsid_select = np.random.choice(df_test['tsid'].unique(), 1000, replace=False)
# df_test = df_test.query('tsid in @tsid_select')

ts_pad = df_test.groupby("tsid")["x"].transform(prepare_series, model_type)
df_test["x_pad"] = ts_pad

# Put into numpy array with shape (samples, timesteps, features)
inputs_test = df_test["x_pad"].to_numpy().reshape(-1, ts_len, 1)
targets_test = df_test["type"].iloc[::ts_len].to_numpy().reshape(-1, 1)

# Get predictions
preds_prob = model.predict(inputs_test)  # prediction probabilities
preds = np.argmax(preds_prob, axis=1)  # class prediction (max prob)

# Show performance stats
# print(classification_report(targets_test, preds, digits=3))
print("Performance of model 2\n")
print("F1 score: {:.3f}".format(f1_score(targets_test, preds, average="macro")))
print("Accuracy: {:3f}".format(accuracy_score(targets_test, preds)))
print("Confusion matrix: \n")
print(confusion_matrix(targets_test, preds))

# Make confusion matrix plot
class_names = ["Null", "Fold", "TC", "PF"]
disp = ConfusionMatrixDisplay.from_predictions(
    targets_test, preds, display_labels=class_names, cmap=plt.cm.Blues, normalize="true"
)
ax = disp.ax_
ax.images[0].colorbar.remove()
plt.text(x=-1.3, y=-0.1, s="B", fontdict={"size": 14})
plt.savefig("figures/cm_mtype_{}.png".format(model_type), bbox_inches="tight", dpi=300)


# -----------
# Confusion matrix for model type 2 - binary classificaiton problem
# ------------


def full_to_binary(bif_class):
    if bif_class >= 1:
        out = 1
    else:
        out = 0
    return out


targets_test_binary = np.array(list(map(full_to_binary, targets_test)))
preds_binary = np.array(list(map(full_to_binary, preds)))

# Show performance stats
# print(classification_report(targets_test, preds, digits=3))
print("Performance of model 2\n")
print(
    "F1 score: {:.3f}".format(
        f1_score(targets_test_binary, preds_binary, average="macro")
    )
)
print("Accuracy: {:3f}".format(accuracy_score(targets_test_binary, preds_binary)))
print("Confusion matrix: \n")
print(confusion_matrix(targets_test_binary, preds_binary))

# Make confusion matrix plot
class_names = ["Null", "Bifurcation"]
disp = ConfusionMatrixDisplay.from_predictions(
    targets_test_binary,
    preds_binary,
    display_labels=class_names,
    cmap=plt.cm.Blues,
    normalize="true",
)
ax = disp.ax_
ax.images[0].colorbar.remove()
plt.yticks(rotation=90, va="center")
plt.text(x=-0.9, y=-0.35, s="D", fontdict={"size": 14})
plt.ylabel(ylabel="True label", labelpad=15)
plt.savefig(
    "figures/cm_mtype_{}_binary.png".format(model_type), bbox_inches="tight", dpi=300
)


# -----------
# Make combined figure of confusion matrices
# -----------

from PIL import Image


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


img_1 = Image.open("figures/cm_mtype_1.png")
img_1_binary = Image.open("figures/cm_mtype_1_binary.png")
img_2 = Image.open("figures/cm_mtype_2.png")
img_2_binary = Image.open("figures/cm_mtype_2_binary.png")

total_width = img_1.width * 2
total_height = img_1.height + img_1_binary.height

right_margin = int((total_width / 2 - img_1_binary.width))

img_1_binary = add_margin(img_1_binary, 0, right_margin, 0, 0, (255, 255, 255))
img_2_binary = add_margin(img_2_binary, 0, right_margin, 0, 0, (255, 255, 255))

dst = Image.new("RGB", (2 * img_1.width, img_1.height + img_1_binary.height))

dst.paste(img_1, (0, 0))
dst.paste(img_2, (img_1.width, 0))
dst.paste(img_1_binary, (0, img_1.height))
dst.paste(img_2_binary, (img_1.width, img_1.height))

dst.save("figures/fig_confusion.png")

end = time.time()
print("Script took {:0.1f} seconds".format(end - start))


# # -----------
# # Plot of training accuracy over time
# # -----------

# df_history_1 = pd.read_csv("training_history/df_history_{}.csv".format(model_name_1))
# df_history_1["model"] = 1

# df_history_2 = pd.read_csv("training_history/df_history_{}.csv".format(model_name_2))
# df_history_2["model"] = 2

# df_plot = pd.concat([df_history_1, df_history_2])
# df_plot["iter"] = df_plot.index

# df_plot = df_plot.melt(
#     id_vars=["iter", "model"],
#     value_vars=["accuracy", "val_accuracy"],
#     var_name="metric",
#     value_name="val",
# )

# fig = px.line(df_plot, x="iter", y="val", color="model", line_dash="metric")
# fig.write_html("figures/fig_history.html")
