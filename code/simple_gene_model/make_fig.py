#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

Make fig with rows
- bifurcation diagram
- simulation
- FateNet predictions

Columns:
- saddle-node bifurcation
- pitchfork bifurcation
- null

@author: tbury
"""

import time

start_time = time.time()

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

pio.kaleido.scope.mathjax = None

sigma = 0.05
path = f"output/sigma_{sigma}/"

# Load in data
df_fold = pd.read_csv(path + "df_fold_forced.csv")
df_pf = pd.read_csv(path + "df_pf_forced.csv")
df_null = pd.read_csv(path + "df_null.csv")

df_fold_bif = pd.read_csv(path + "df_fold_bif.csv")
df_pf_bif = pd.read_csv(path + "df_pf_bif.csv")
df_null_bif = pd.read_csv(path + "df_null_bif.csv")


# DL probability for *any* bifurcation
bif_labels = ["1", "2", "3"]
df_fold["any"] = df_fold[bif_labels].dropna().sum(axis=1)
df_pf["any"] = df_pf[bif_labels].dropna().sum(axis=1)
df_null["any"] = df_null[bif_labels].dropna().sum(axis=1)


# Time column for bif data
m1start = 1
m1end = 4.75
m = 750 / (m1end - m1start)

df_fold_bif["time"] = df_fold_bif["m1"].apply(lambda x: m * (x - m1start))
df_null_bif["time"] = df_null_bif["m1"].apply(lambda x: m * (x - m1start))


kstart = 1
kend = 1 / 4
m = 750 / (kend - kstart)
df_pf_bif["time"] = df_pf_bif["k"].apply(lambda x: m * (x - kstart))


# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray


col_other_bif = "gray"
dic_colours = {
    "state": "gray",
    "smoothing": "black",
    "variance": cols[1],
    "ac": cols[2],
    "dl_any": cols[0],
    "dl_fold": cols[4],
    "dl_other": col_other_bif,
    "dl_ns": col_other_bif,
    "dl_fold": cols[1],
    "dl_tc": cols[3],
    "dl_pf": cols[2],
    "dl_null": cols[0],
    "bif": "black",
}


fig_height = 550
fig_width = 800

font_size = 16
font_family = "Times New Roman"
font_size_letter_label = 16
font_size_titles = 18

linewidth = 1
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 5

marker_size = 1.6

# Opacity of DL probabilities for different bifs
opacity = 1

# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0


# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600

fig = make_subplots(
    rows=3,
    cols=3,
    shared_xaxes=True,
    vertical_spacing=0.04,
)

# ----------------
# Col 1: fold
# ------------------

col = 1

# Bifurcation plot
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="0_real"')["time"],
        y=df_fold_bif.query('variable=="0_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="3_real"')["time"],
        y=df_fold_bif.query('variable=="3_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth, "dash": "dash"},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="4_real"')["time"],
        y=df_fold_bif.query('variable=="4_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_fold_bif.query('variable=="2_real"')["time"],
        y=df_fold_bif.query('variable=="2_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=1,
)


df = df_fold
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        mode="markers",
        marker=dict(size=marker_size),
        showlegend=False,
        # line={'width':linewidth},
    ),
    row=2,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Weight for null
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["0"],
        marker_color=dic_colours["dl_null"],
        name="Null",
        showlegend=True,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_fold"],
        # showlegend=False,
        name="Fold",
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
        name="TC",
    ),
    row=3,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# ----------------
# Col 2: pf
# ------------------

col = 2


# Bifurcation plot
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="0_real"')["time"],
        y=df_pf_bif.query('variable=="0_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="2_real" and time>500')["time"],
        y=df_pf_bif.query('variable=="2_real" and time>500')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="3_real"')["time"],
        y=df_pf_bif.query('variable=="3_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth, "dash": "dash"},
    ),
    row=1,
    col=col,
)
fig.add_trace(
    go.Scatter(
        x=df_pf_bif.query('variable=="4_real"')["time"],
        y=df_pf_bif.query('variable=="4_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


df = df_pf
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        mode="markers",
        marker=dict(size=marker_size),
        showlegend=False,
        # line={'width':linewidth},
    ),
    row=2,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_pf"],
        # showlegend=False,
        name="PF",
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_tc"],
        showlegend=True,
        name="TC",
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for null
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["0"],
        marker_color=dic_colours["dl_null"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# ----------------
# Col 3: null
# ------------------

col = 3


# Bifurcation plot
fig.add_trace(
    go.Scatter(
        x=df_null_bif.query('variable=="4_real"')["time"],
        y=df_null_bif.query('variable=="4_real"')["value"],
        marker_color=dic_colours["bif"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=1,
    col=col,
)


df = df_null
# Trace for trajectory
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        mode="markers",
        marker=dict(size=marker_size),
        # line={'width':linewidth},
    ),
    row=2,
    col=col,
)

# Trace for smoothing
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["smoothing"],
        marker_color=dic_colours["smoothing"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)

# Weight for Fold
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["1"],
        marker_color=dic_colours["dl_fold"],
        name="Fold",
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)

# Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["2"],
        marker_color=dic_colours["dl_tc"],
        showlegend=False,
        # showlegend=False,
        # name='Other',
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["3"],
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        # name='Pitchfork',
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# Weight for null
fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["0"],
        marker_color=dic_colours["dl_null"],
        showlegend=False,
        line={"width": linewidth},
        opacity=opacity,
    ),
    row=3,
    col=col,
)


# --------------
# Shapes
# --------------

list_shapes = []


# Vertical lines for where transitions occur
t_transition = 501

#  Line for fold bif
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 0.65,
    "xref": "x",
    "yref": "paper",
    "line": {"width": 1.5, "dash": "dot"},
}
list_shapes.append(shape)

#  Line for PF bif
shape = {
    "type": "line",
    "x0": t_transition,
    "y0": 0,
    "x1": t_transition,
    "y1": 0.65,
    "xref": "x2",
    "yref": "paper",
    "line": {"width": 1.5, "dash": "dot"},
}
list_shapes.append(shape)

fig["layout"].update(shapes=list_shapes)

# --------------
# Add annotations
# ----------------------

list_annotations = []

# Letter labels for each panel
import string

label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1, 10)]
axes_numbers[0] = ""
idx = 0
for axis_number in axes_numbers:
    label_annotation = dict(
        x=0.01,
        y=1.00,
        text="({})".format(label_letters[idx]),
        xref="x{} domain".format(axis_number),
        yref="y{} domain".format(axis_number),
        showarrow=False,
        font=dict(color="black", size=font_size_letter_label),
    )
    list_annotations.append(label_annotation)
    idx += 1


# Bifurcation titles
y_pos = 1.06
title_fold = dict(
    x=0.5,
    y=y_pos,
    text="Fold",
    xref="x domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)

title_pf = dict(
    x=0.5,
    y=y_pos,
    text="Pitchfork",
    xref="x2 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)

title_null = dict(
    x=0.5,
    y=y_pos,
    text="Null",
    xref="x3 domain",
    yref="paper",
    showarrow=False,
    font=dict(color="black", size=font_size_titles),
)


# Arrows to indiciate rolling window
axes_numbers = [7, 8, 9]
arrowhead = 1
arrowsize = 2
arrowwidth = 0.5

for axis_number in axes_numbers:
    # Make right-pointing arrow
    annotation_arrow_right = dict(
        x=0,  # arrows' head
        y=0.1,  # arrows' head
        ax=100,  # arrows' tail
        ay=0.1,  # arrows' tail
        xref="x{}".format(axis_number),
        yref="y{} domain".format(axis_number),
        axref="x{}".format(axis_number),
        ayref="y{} domain".format(axis_number),
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor="black",
    )
    # Make left-pointing arrow
    annotation_arrow_left = dict(
        ax=0,  # arrows' head
        y=0.1,  # arrows' head
        x=100,  # arrows' tail
        ay=0.1,  # arrows' tail
        xref="x{}".format(axis_number),
        yref="y{} domain".format(axis_number),
        axref="x{}".format(axis_number),
        ayref="y{} domain".format(axis_number),
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=arrowhead,
        arrowsize=arrowsize,
        arrowwidth=arrowwidth,
        arrowcolor="black",
    )

    # Append to annotations
    list_annotations.append(annotation_arrow_left)
    list_annotations.append(annotation_arrow_right)


# list_annotations.append(label_annotation)

list_annotations.append(title_fold)
list_annotations.append(title_pf)
list_annotations.append(title_null)


fig["layout"].update(annotations=list_annotations)


# -------------
# Axes properties
# -----------

# Global y axis properties
fig.update_yaxes(
    showline=True,
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=yaxes_standoff,
)

# Global x axis properties
fig.update_xaxes(
    range=[0, 750],
    showline=True,
    linecolor="black",
    linewidth=linewidth_axes,
    mirror=False,
    showgrid=False,
    automargin=False,
    title_standoff=xaxes_standoff,
)


# Specific x axes properties
fig.update_xaxes(
    title="Pseudotime",
    ticks="outside",
    tickwidth=tickwidth,
    ticklen=ticklen,
    row=3,
)

# Specific y axes properties
fig.update_yaxes(title="Gene 1", row=1, col=1)
fig.update_yaxes(title="Gene 1", row=2, col=1)
fig.update_yaxes(title="FateNet", row=3, col=1)


fig.update_yaxes(range=[0, 4.5], row=1, col=1)
fig.update_yaxes(range=[0, 4.5], row=2, col=1)

fig.update_yaxes(range=[0, 2], row=1, col=2)
fig.update_yaxes(range=[0, 2], row=2, col=2)


fig.update_yaxes(range=[0, 1], row=1, col=3)
fig.update_yaxes(range=[0, 1], row=2, col=3)


fig.update_yaxes(range=[-0.05, 1.05], row=3)


# General layout properties
fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin={"l": 50, "r": 5, "b": 50, "t": 35},
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)

fig.update_layout(
    legend=dict(
        yanchor="bottom",
        y=0.05,
        xanchor="right",
        x=1,
    )
)

fig.update_traces(connectgaps=True)

fig.write_image(f"figures/freedman_model_preds_sigma_{sigma}.png", scale=2)
fig.write_image(f"figures/freedman_model_preds_sigma_{sigma}.pdf", scale=2)

print("Exported figure", f"figures/freedman_model_preds_sigma_{sigma}.png")
