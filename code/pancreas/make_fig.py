#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

Make fig with rows
- Pancreas PCA 1st component
- DL predictions (averaged over top 10 PCA)

Columns:
- Cell trajectory up to bifurcation (forced)
- Shuffled trajectory (null)

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

# -----------------
# Read in trajectory data
# -----------------
df_traj = pd.read_csv("output/df_pca.csv")

# ------------
# Read in EWS
# ------------

df_ews_forced = pd.read_csv("output/df_ews_forced.csv")
df_dl_forced = pd.read_csv("output/df_dl_forced.csv")
df_dl_forced = df_dl_forced.groupby("time").mean(numeric_only=True).reset_index()
df_dl_forced = df_dl_forced.query("pseudotime_index>=100")

df_ews_null = pd.read_csv("output/df_ews_null.csv")
df_dl_null = pd.read_csv("output/df_dl_null.csv")
df_dl_null = df_dl_null.groupby("time").mean(numeric_only=True).reset_index()
df_dl_null = df_dl_null.query("pseudotime_index>=100")

# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

col_other_bif = "gray"
dic_colours = {
    "state": "gray",
    "dl_any": cols[0],
    "dl_pd": col_other_bif,
    "dl_ns": cols[1],
    "dl_fold": cols[1],
    "dl_trans": cols[3],
    "dl_pf": cols[2],
    "dl_null": cols[0],
    "bif": "black",
}

fig_height = 400
fig_width = 700

font_size = 16
font_family = "Times New Roman"
font_size_letter_label = 16
font_size_titles = 18

linewidth = 1
linewidth_axes = 0.5
tickwidth = 0.5
ticklen = 5
marker_size = 2.5

# Opacity of DL probabilities for different bifs
opacity = 1

# dist from axis to axis label
xaxes_standoff = 0
yaxes_standoff = 0

# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600


# plot as a function of pseudotime or pseudotime index
xvar = "pseudotime"
# xvar = "pseudotime_index"

# Transition time is taken as time where Fev+ ends
cluster = "Fev+"
pca_comp = 0
transition = df_traj.query("cluster==@cluster")["pseudotime_index"].iloc[-1]
transition_pseudo = df_traj.query("cluster==@cluster")["pseudotime"].iloc[-1]
transition_pseudo = 0.924
df_dl_forced = df_dl_forced.query("pseudotime<=@transition_pseudo")

fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    vertical_spacing=0.04,
)

# ----------------
# Col 1: forced
# ----------------

col = 1

# Trajectory of Fev+ cells
cluster = "Fev+"
df_plot = df_traj.query(
    "cluster==@cluster and pseudotime_index<=@transition and `PCA comp`==@pca_comp"
).copy()
# Reset pseudotime index to start at zero
first_idx = df_plot["pseudotime_index"].iloc[0]
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color="gray",
        showlegend=True,
        legend="legend",
        name="Fev+",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)


# Trajectory of Alpha cells
cluster = "Alpha"
df_plot = df_traj.query("cluster==@cluster and `PCA comp`==@pca_comp").copy()
# Reset pseudotime index to start at zero
first_idx = df_plot["pseudotime_index"].iloc[0]
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color=cols[6],
        showlegend=True,
        legend="legend",
        name="Alpha",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# Trajectory of Delta cells
cluster = "Delta"
df_plot = df_traj.query("cluster==@cluster and `PCA comp`==@pca_comp").copy()
# Reset pseudotime index to start at zero
first_idx = df_plot["pseudotime_index"].iloc[0]
df_plot["pseudotime_index"] = df_plot["pseudotime_index"] - first_idx
fig.add_trace(
    go.Scatter(
        x=df_plot[xvar],
        y=df_plot["value"],
        marker_color=cols[5],
        showlegend=True,
        legend="legend",
        name="Delta",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# DL Weight for null
fig.add_trace(
    go.Scatter(
        x=df_dl_forced[xvar],
        y=df_dl_forced["0"],
        # mode='lines',
        marker_color=dic_colours["dl_null"],
        showlegend=True,
        legend="legend2",
        name="Null",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)

# DL Weight for fold
fig.add_trace(
    go.Scatter(
        x=df_dl_forced[xvar],
        y=df_dl_forced["1"],
        # mode='lines',
        marker_color=dic_colours["dl_fold"],
        showlegend=True,
        name="Fold",
        legend="legend2",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)
# DL Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df_dl_forced[xvar],
        y=df_dl_forced["2"],
        # mode='lines',
        marker_color=dic_colours["dl_trans"],
        showlegend=True,
        name="TC",
        legend="legend2",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)
# DL Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df_dl_forced[xvar],
        y=df_dl_forced["3"],
        # mode='lines',
        marker_color=dic_colours["dl_pf"],
        showlegend=True,
        legend="legend2",
        name="PF",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# ----------------
# Col 2: null
# ----------------

col = 2

# Trajectory
fig.add_trace(
    go.Scatter(
        x=df_ews_null[xvar],
        y=df_ews_null["state"],
        marker_color=dic_colours["state"],
        showlegend=False,
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)


# DL Weight for null
fig.add_trace(
    go.Scatter(
        x=df_dl_null[xvar],
        y=df_dl_null["0"],
        # mode='lines',
        marker_color=dic_colours["dl_null"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)

# DL Weight for fold
fig.add_trace(
    go.Scatter(
        x=df_dl_null[xvar],
        y=df_dl_null["1"],
        # mode='lines',
        marker_color=dic_colours["dl_fold"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)
# DL Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df_dl_null[xvar],
        y=df_dl_null["2"],
        # mode='lines',
        marker_color=dic_colours["dl_trans"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)
# DL Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df_dl_null[xvar],
        y=df_dl_null["3"],
        # mode='lines',
        marker_color=dic_colours["dl_pf"],
        showlegend=False,
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)

# --------
# Annotations
# ---------

list_annotations = []

# Letter labels for each panel
import string

label_letters = string.ascii_lowercase

axes_numbers = [str(n) for n in np.arange(1, 5)]
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


# Arrows to indiciate rolling window
axes_numbers = [3, 4]
arrowhead = 1
arrowsize = 2
arrowwidth = 0.5
for axis_number in axes_numbers:
    # Make right-pointing arrow
    annotation_arrow_right = dict(
        x=0.82,  # arrows' head
        y=0.1,  # arrows' head
        ax=0.85,  # arrows' tail
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
        ax=0.82,  # arrows' head
        y=0.1,  # arrows' head
        x=0.85,  # arrows' tail
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


fig["layout"].update(annotations=list_annotations)


fig.add_annotation(
    x=0.08,
    y=1.08,
    text="Bifurcation trajectory",
    showarrow=False,
    font=dict(size=18),
    xref="paper",
    yref="paper",
)


fig.add_annotation(
    x=0.85,
    y=1.08,
    text="Null trajectory",
    showarrow=False,
    font=dict(size=18),
    xref="paper",
    yref="paper",
)

list_shapes = []
# Vertical lines for where transitions occur

#  Line for transition col1
shape = {
    "type": "line",
    "x0": transition_pseudo if xvar == "pseudotime" else transition,
    "y0": 0,
    "x1": transition_pseudo if xvar == "pseudotime" else transition,
    "y1": 1,
    "xref": "x",
    "yref": "paper",
    "line": {"width": 1.5, "dash": "dot"},
}
list_shapes.append(shape)

fig["layout"].update(shapes=list_shapes)


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
    range=[0.82, 0.98],
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
    row=2,
    # col=1,
)

# # Specific y axes properties
fig.update_yaxes(title="First PCA", row=1, col=1)
fig.update_yaxes(title="FateNet", row=2, col=1)
fig.update_yaxes(range=[-15, 20], row=1)


# General layout properties
fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin={"l": 60, "r": 5, "b": 60, "t": 35},
    font=dict(size=font_size, family=font_family),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(255,255,255,1)",
)


fig.update_layout(
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.7,
        xanchor="right",
        x=1,
        font=dict(size=14),
        itemsizing="constant",
    )
)

fig.update_layout(
    legend2=dict(
        yanchor="bottom",
        y=0.15,
        xanchor="right",
        x=1,
        font=dict(size=14),
    )
)

# fig.update_traces(connectgaps=True)
fig.write_image("figures/fig_pancreas.png", scale=2)
fig.write_image("figures/fig_pancreas.pdf", scale=2)
