#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:23:35 2022

Make fig with rows
- Weinreb Klein PCA 0 (dot plot)
- DL predictions

Columns:
- Cell trajectory up to transition (forced)
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

df_traj = pd.read_csv("../output/df_traj_forced.csv")
df_null = pd.read_csv("../output/df_traj_null.csv")

# ------------
# Read in EWS
# ------------

df_ews_forced = pd.read_csv("../output/ews/df_ews_forced.csv")
df_ews_forced["sample"] = 1
df_ews_null = pd.read_csv("../output/ews/df_ews_null.csv")
df_ews_null["sample"] = 1

df_dl_forced = pd.read_csv("../output/ews/df_dl_forced.csv")
df_dl_forced = df_dl_forced.groupby(["time"]).mean(numeric_only=True).reset_index()
df_dl_forced["any"] = df_dl_forced[["1", "2", "3"]].sum(axis=1)
df_dl_forced["sample"] = 1
df_dl_forced = df_dl_forced.query("time>0.1")


df_dl_null = pd.read_csv("../output/ews/df_dl_null.csv")
df_dl_null = df_dl_null.groupby(["time"]).mean(numeric_only=True).reset_index()
df_dl_null["any"] = df_dl_null[["1", "2", "3"]].sum(axis=1)
df_dl_null["sample"] = 1
df_dl_null = df_dl_null.query("time>0.1")


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
    "dl_pd": col_other_bif,
    "dl_ns": col_other_bif,
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
font_size_titles = 16

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

fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    vertical_spacing=0.04,
)

# ----------------
# Col 1: forced
# ------------------

col = 1
sample_plot = 1

# Trajectory of undiff cells
fig.add_trace(
    go.Scatter(
        x=df_traj.query('`Cell type annotation`=="Undifferentiated"')[
            "pseudotime"
        ].iloc[::100],
        y=df_traj.query('`Cell type annotation`=="Undifferentiated"')["0"].iloc[::100],
        marker_color=dic_colours["state"],
        showlegend=True,
        name="Undifferentiated",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# Trajectory of neutrophils
fig.add_trace(
    go.Scatter(
        x=df_traj.query('`Cell type annotation`=="Neutrophil"')["pseudotime"].iloc[
            ::100
        ],
        y=df_traj.query('`Cell type annotation`=="Neutrophil"')["0"].iloc[::100],
        marker_color=cols[4],
        showlegend=True,
        name="Neutrophil",
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# # Residuals
# fig.add_trace(
#     go.Scatter(x=df_ews_forced.query('sample==@sample_plot')['pseudotime'],
#                y=df_ews_forced.query('sample==@sample_plot')['residuals'],
#                marker_color='gray',
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=2,col=col,
#     )


# # Variance
# fig.add_trace(
#     go.Scatter(x=df_ews_forced['pseudotime'],
#                y=df_ews_forced['variance'],
#                marker_color='gray',
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )


# # DL Weight for any
# fig.add_trace(
#     go.Scatter(x=df_dl_forced['time'],
#                 y=df_dl_forced['any'],
#                 marker_color=dic_colours['dl_any'],
#                 showlegend=True,
#                 name='Any bif',
#                 line={'width':linewidth},
#                 ),
#     row=3,col=col,
#     )

# DL Weight for null
fig.add_trace(
    go.Scatter(
        x=df_dl_forced.query("sample==@sample_plot")["time"],
        y=df_dl_forced.query("sample==@sample_plot")["0"],
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
        x=df_dl_forced.query("sample==@sample_plot")["time"],
        y=df_dl_forced.query("sample==@sample_plot")["1"],
        marker_color=dic_colours["dl_fold"],
        showlegend=True,
        legend="legend2",
        name="Fold",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# DL Weight for transcritical
fig.add_trace(
    go.Scatter(
        x=df_dl_forced.query("sample==@sample_plot")["time"],
        y=df_dl_forced.query("sample==@sample_plot")["2"],
        marker_color=dic_colours["dl_trans"],
        showlegend=True,
        legend="legend2",
        name="TC",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)

# DL Weight for pitchfork
fig.add_trace(
    go.Scatter(
        x=df_dl_forced.query("sample==@sample_plot")["time"],
        y=df_dl_forced.query("sample==@sample_plot")["3"],
        marker_color=dic_colours["dl_pf"],
        showlegend=True,
        legend="legend2",
        name="PF",
        line={"width": linewidth},
    ),
    row=2,
    col=col,
)


# ##### Mean and std of DL weights
# df_temp = df_dl_forced.copy()
# df_temp['time'] = df_temp['time']-df_temp['time'].mod(500)
# df_dl_forced_mean = df_temp.groupby(['time']).mean().reset_index()
# df_dl_forced_std = df_temp.groupby('time').std().reset_index()


# # DL Weight for null
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                 y=df_dl_forced_mean['0'],
#                 marker_color=dic_colours['dl_null'],
#                 showlegend=True,
#                 name='Null',
#                 line={'width':linewidth},
#                 ),
#     row=3,col=col,
#     )
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                 y=df_dl_forced_mean['0']+df_dl_forced_std['0'],
#                 marker_color=dic_colours['dl_null'],
#                 showlegend=True,
#                 name='Null',
#                 line={'width':linewidth},
#                 ),
#     row=3,col=col,
#     )

# # DL Weight for fold
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                y=df_dl_forced_mean['1'],
#                marker_color=dic_colours['dl_fold'],
#                showlegend=True,
#                name='Fold',
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                y=df_dl_forced_mean['1']+df_dl_forced_std['1'],
#                marker_color=dic_colours['dl_fold'],
#                showlegend=True,
#                name='Fold',
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )

# # DL Weight for transcritical
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                y=df_dl_forced_mean['2'],
#                marker_color=dic_colours['dl_trans'],
#                showlegend=True,
#                name='TC',
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                y=df_dl_forced_mean['2'] + df_dl_forced_std['2'],
#                marker_color=dic_colours['dl_trans'],
#                showlegend=True,
#                name='TC',
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )
# # DL Weight for pitchfork
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                y=df_dl_forced_mean['3'],
#                marker_color=dic_colours['dl_pf'],
#                showlegend=True,
#                name='PF',
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )
# fig.add_trace(
#     go.Scatter(x=df_dl_forced_mean['time'],
#                y=df_dl_forced_mean['3']+df_dl_forced_std['3'],
#                marker_color=dic_colours['dl_pf'],
#                showlegend=True,
#                name='PF',
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )


# ----------------
# Col 2: null
# ------------------

col = 2
sample_plot_null = 1

# Trajectory
fig.add_trace(
    go.Scatter(
        x=df_null["pseudotime"],
        y=df_null["0"],
        marker_color=dic_colours["state"],
        showlegend=False,
        mode="markers",
        marker=dict(size=marker_size),
    ),
    row=1,
    col=col,
)

# # Binned time series
# fig.add_trace(
#     go.Scatter(x=df_ews_null['time'],
#                y=df_ews_null['state'],
#                marker_color=cols[0],
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=2,col=col,
#     )


# # Smoothing
# fig.add_trace(
#     go.Scatter(x=df_ews_null['time'],
#                y=df_ews_null['smoothing'],
#                marker_color='black',
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=2,col=col,
#     )


# # Residuals
# fig.add_trace(
#     go.Scatter(x=df_ews_null.query('sample==@sample_plot_null')['time'],
#                y=df_ews_null.query('sample==@sample_plot_null')['residuals'],
#                marker_color='gray',
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=2,col=col,
#     )


# # Variance
# fig.add_trace(
#     go.Scatter(x=df_ews_null.groupby('time').mean().index,
#                y=df_ews_null.groupby('time').mean()['variance'],
#                marker_color='gray',
#                showlegend=False,
#                line={'width':linewidth},
#                ),
#     row=3,col=col,
#     )


# # DL Weight for any
# fig.add_trace(
#     go.Scatter(x=df_dl_null['time'],
#                 y=df_dl_null['any'],
#                 marker_color=dic_colours['dl_any'],
#                 showlegend=False,
#                 name='Any bif',
#                 line={'width':linewidth},
#                 ),
#     row=3,col=col,
#     )

# DL Weight for null
fig.add_trace(
    go.Scatter(
        x=df_dl_null.query("sample==@sample_plot_null")["time"],
        y=df_dl_null.query("sample==@sample_plot_null")["0"],
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
        x=df_dl_null.query("sample==@sample_plot_null")["time"],
        y=df_dl_null.query("sample==@sample_plot_null")["1"],
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
        x=df_dl_null.query("sample==@sample_plot_null")["time"],
        y=df_dl_null.query("sample==@sample_plot_null")["2"],
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
        x=df_dl_null.query("sample==@sample_plot_null")["time"],
        y=df_dl_null.query("sample==@sample_plot_null")["3"],
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
        x=0,  # arrows' head
        y=0.1,  # arrows' head
        ax=0.1,  # arrows' tail
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
        x=0.1,  # arrows' tail
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


list_shapes = []
# Vertical lines for where transitions occur

#  Line for transition col1
shape = {
    "type": "line",
    "x0": 0.6,
    "y0": 0,
    "x1": 0.6,
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
    # range=[0,750],
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
)

# Specific y axes properties
fig.update_yaxes(title="First PCA", row=1, col=1)
# fig.update_yaxes(title='Binned data',row=2,col=1)
# fig.update_yaxes(title='Residuals',row=2,col=1)
# fig.update_yaxes(title='Variance',row=3,col=1)
fig.update_yaxes(title="FateNet", row=2, col=1)


fig.update_yaxes(range=[-10, 14], row=1)
# fig.update_yaxes(range=[-3.5,3.5],row=2)

fig.update_yaxes(range=[-0.05, 1.1], row=2)

fig.update_xaxes(range=[0, 0.9], row=2, col=1)
fig.update_xaxes(range=[0, 0.9], row=2, col=2)

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
        yanchor="bottom",
        orientation="v",
        y=0.8,
        xanchor="right",
        x=1,
        itemsizing="constant",
    )
)


fig.update_layout(
    legend2=dict(
        yanchor="bottom",
        orientation="v",
        y=0.1,
        xanchor="right",
        x=1,
    )
)

fig.update_traces(connectgaps=True)

# Export as temp image
fig.write_image("fig_weinreb_klein.png", scale=2)
fig.write_image("fig_weinreb_klein.pdf", scale=2)
