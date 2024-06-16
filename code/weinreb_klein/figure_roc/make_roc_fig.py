#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:30:10 2023

Get DL predictions on WK data downsampled at 100 different phases.
Generate csp. nulls and DL predictions.
Construct ROC curve

@author: tbury
"""


import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

# Import PIL for image tools
from PIL import Image

sample_size = 100

df_dl = pd.read_csv("output/df_dl_{}.csv".format(sample_size))
df_ktau = pd.read_csv("output/df_ktau_{}.csv".format(sample_size))

df_dl["truth_value"] = df_dl["type"].apply(lambda x: 1 if x == "forced" else 0)
df_ktau["truth_value"] = df_ktau["type"].apply(lambda x: 1 if x == "forced" else 0)

## Get data on ML favoured bifurcation for each forced trajectory
bif_labels = [str(i) for i in np.arange(1, 4)]
df_dl["fav_bif"] = df_dl[bif_labels].idxmax(axis=1)

df_dl_forced = df_dl.query("truth_value==1")

# Count each bifurcation choice for forced trajectories
counts = df_dl[df_dl["truth_value"] == 1]["fav_bif"].value_counts()
df_counts = pd.DataFrame(index=bif_labels)
df_counts.index.name = "bif_id"
df_counts["count"] = counts
# Nan as 0
df_counts.fillna(value=0, inplace=True)


def roc_compute(truth_vals, indicator_vals):
    # Compute ROC curve and threhsolds using sklearn
    fpr, tpr, thresholds = metrics.roc_curve(truth_vals, indicator_vals)

    # Compute AUC (area under curve)
    auc = metrics.auc(fpr, tpr)

    # Put into a DF
    dic_roc = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}
    df_roc = pd.DataFrame(dic_roc)

    return df_roc


# Initiliase list for ROC dataframes for predicting May fold bifurcation
list_roc = []

# Assign indicator and truth values for ML prediction
indicator_vals = df_dl["any_bif"]
truth_vals = df_dl["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "DL bif"
list_roc.append(df_roc)


# Assign indicator and truth values for variance
indicator_vals = df_ktau["variance"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "Variance"
list_roc.append(df_roc)


# Assign indicator and truth values for lag-1 AC
indicator_vals = df_ktau["ac1"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "Lag-1 AC"
list_roc.append(df_roc)


# Assign indicator and truth values for sample-entropy-0
indicator_vals = -df_ktau["sample-entropy-0"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "sample-entropy-0"
list_roc.append(df_roc)

# Assign indicator and truth values for sample-entropy-1
indicator_vals = -df_ktau["sample-entropy-1"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "sample-entropy-1"
list_roc.append(df_roc)


# Assign indicator and truth values for sample-entropy-2
indicator_vals = -df_ktau["sample-entropy-2"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "sample-entropy-2"
list_roc.append(df_roc)


# Assign indicator and truth values for kolmogorov
indicator_vals = -df_ktau["kolmogorov-entropy-0"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "kolmogorov-entropy-0"
list_roc.append(df_roc)


# Assign indicator and truth values for sample-entropy-2
indicator_vals = -df_ktau["kolmogorov-entropy-1"]
truth_vals = df_ktau["truth_value"]
df_roc = roc_compute(truth_vals, indicator_vals)
df_roc["ews"] = "kolmogorov-entropy-1"
list_roc.append(df_roc)

# Concatenate roc dataframes
df_roc_full = pd.concat(list_roc, ignore_index=True)


# -----------
# Make ROC fig
# ------------

# Colour scheme
# cols = px.colors.qualitative.D3 # blue, orange, green, red, purple, brown
cols = (
    px.colors.qualitative.Plotly
)  # blue, red, green, purple, orange, cyan, pink, light green
col_grays = px.colors.sequential.gray

dic_colours = {
    "state": "gray",
    "smoothing": col_grays[2],
    "dl_bif": cols[0],
    "variance": cols[1],
    "ac": cols[2],
    "dl_fold": cols[3],
    "dl_hopf": cols[4],
    "dl_branch": cols[5],
    "dl_null": "black",
    "sample-entropy-0": cols[3],
    "sample-entropy-1": cols[7],
    "sample-entropy-2": cols[8],
    "kolmogorov-entropy-0": cols[4],
    "kolmogorov-entropy-1": cols[2],
}

# Pixels to mm
mm_to_pixel = 96 / 25.4  # 96 dpi, 25.4mm in an inch

# Nature width of single col fig : 89mm
# Nature width of double col fig : 183mm

# Get width of single panel in pixels
fig_width = 183 * mm_to_pixel / 3  # 3 panels wide
fig_height = fig_width


font_size = 10
font_family = "Times New Roman"
font_size_letter_label = 14
font_size_auc_text = 10


# AUC annotations
x_auc = 0.97
y_auc = 0.6
x_N = 0.97
y_N = 0.7
y_auc_sep = 0.065

linewidth = 0.7
linewidth_axes = 0.5
tickwidth = 0.5
linewidth_axes_inset = 0.5

axes_standoff = 0


# Scale up factor on image export
scale = 8  # default dpi=72 - nature=300-600


def make_roc_figure(df_roc, letter_label, title="", text_N=""):
    """Make ROC figure (no inset)"""

    fig = go.Figure()

    # DL prediction any bif
    df_trace = df_roc[df_roc["ews"] == "DL bif"]
    auc_dl = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            name="DL",
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["dl_bif"],
            ),
        )
    )

    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    auc_var = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            name="Var",
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["variance"],
            ),
        )
    )

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            name="AC",
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["ac"],
            ),
        )
    )

    # Sample-entrpy-0
    df_trace = df_roc[df_roc["ews"] == "sample-entropy-0"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            name="SE",
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["sample-entropy-0"],
            ),
        )
    )

    # # Sample-entrpy-1
    # df_trace = df_roc[df_roc["ews"] == "sample-entropy-1"]
    # auc_ac = df_trace.round(2)["auc"].iloc[0]
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_trace["fpr"],
    #         y=df_trace["tpr"],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(
    #             width=linewidth,
    #             color=dic_colours["sample-entropy-1"],
    #         ),
    #     )
    # )

    # Kolmogorov-entropy-0
    df_trace = df_roc[df_roc["ews"] == "kolmogorov-entropy-0"]
    auc_ac = df_trace.round(2)["auc"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            showlegend=False,
            name="KE",
            mode="lines",
            line=dict(
                width=linewidth,
                color=dic_colours["kolmogorov-entropy-0"],
            ),
        )
    )

    # # Kolmogorov-entropy-1
    # df_trace = df_roc[df_roc["ews"] == "kolmogorov-entropy-1"]
    # auc_ac = df_trace.round(2)["auc"].iloc[0]
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_trace["fpr"],
    #         y=df_trace["tpr"],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(
    #             width=linewidth,
    #             color=dic_colours["kolmogorov-entropy-1"],
    #         ),
    #     )
    # )

    # # Sample-entrpy-2
    # df_trace = df_roc[df_roc["ews"] == "sample-entropy-2"]
    # auc_ac = df_trace.round(2)["auc"].iloc[0]
    # fig.add_trace(
    #     go.Scatter(
    #         x=df_trace["fpr"],
    #         y=df_trace["tpr"],
    #         showlegend=False,
    #         mode="lines",
    #         line=dict(
    #             width=linewidth,
    #             color=dic_colours["sample-entropy-2"],
    #         ),
    #     )
    # )

    # Line y=x
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            showlegend=False,
            line=dict(
                color="black",
                dash="dot",
                width=linewidth,
            ),
        )
    )

    # --------------
    # Add labels and titles
    # ----------------------

    list_annotations = []
    list_shapes = []

    label_annotation = dict(
        # x=sum(xrange)/2,
        x=0.02,
        y=1,
        text="<b>{}</b>".format(letter_label),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_letter_label,
        ),
    )

    annotation_auc_dl = dict(
        # x=sum(xrange)/2,
        x=x_auc,
        y=y_auc,
        text="AUC={:.2f}".format(auc_dl),
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    # annotation_auc_var = dict(
    #         # x=sum(xrange)/2,
    #         x=x_auc,
    #         y=y_auc-y_auc_sep,
    #         text='A<sub>Var</sub>={:.2f}'.format(auc_var),
    #         xref='paper',
    #         yref='paper',
    #         showarrow=False,
    #         font = dict(
    #                 color = 'black',
    #                 size = font_size_auc_text,
    #                 )
    #         )

    # annotation_auc_ac = dict(
    #         # x=sum(xrange)/2,
    #         x=x_auc,
    #         y=y_auc-2*y_auc_sep,
    #         text='A<sub>AC</sub>={:.2f}'.format(auc_ac),
    #         xref='paper',
    #         yref='paper',
    #         showarrow=False,
    #         font = dict(
    #                 color = 'black',
    #                 size = font_size_auc_text,
    #                 )
    #         )

    annotation_N = dict(
        # x=sum(xrange)/2,
        x=x_N,
        y=y_N,
        text=text_N,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    legend_y = 0.7
    legend_spacing = 0.07
    legend_x = 0.9
    legend_circle_diam = 0.03
    legend_circle_offset_x = 0.05
    legend_circle_offset_y = 0.02

    annotation_dl = dict(
        # x=sum(xrange)/2,
        x=legend_x,
        y=legend_y,
        xanchor="left",
        text="FN",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    annotation_var = dict(
        # x=sum(xrange)/2,
        x=legend_x,
        y=legend_y - legend_spacing,
        xanchor="left",
        text="Var",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    annotation_ac = dict(
        # x=sum(xrange)/2,
        x=legend_x,
        y=legend_y - 2 * legend_spacing,
        xanchor="left",
        text="AC",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    annotation_se = dict(
        # x=sum(xrange)/2,
        x=legend_x,
        y=legend_y - 3 * legend_spacing,
        text="SE",
        xanchor="left",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    annotation_ke = dict(
        # x=sum(xrange)/2,
        x=legend_x,
        y=legend_y - 4 * legend_spacing,
        xanchor="left",
        text="KC",
        xref="x",
        yref="y",
        showarrow=False,
        font=dict(
            color="black",
            size=font_size_auc_text,
        ),
    )

    circle_dl = dict(
        type="circle",
        xref="x",
        yref="y",
        fillcolor=cols[0],
        x0=legend_x - legend_circle_offset_x,
        y0=legend_y - legend_circle_offset_y,
        x1=legend_x - legend_circle_offset_x + legend_circle_diam,
        y1=legend_y - legend_circle_offset_y + legend_circle_diam,
        line_color=cols[0],
    )

    circle_var = dict(
        type="circle",
        xref="x",
        yref="y",
        fillcolor=dic_colours["variance"],
        x0=legend_x - legend_circle_offset_x,
        y0=legend_y - legend_circle_offset_y - 1 * legend_spacing,
        x1=legend_x - legend_circle_offset_x + legend_circle_diam,
        y1=legend_y - legend_circle_offset_y + legend_circle_diam - 1 * legend_spacing,
        line_color=dic_colours["variance"],
    )

    circle_ac = dict(
        type="circle",
        xref="x",
        yref="y",
        fillcolor=dic_colours["ac"],
        x0=legend_x - legend_circle_offset_x,
        y0=legend_y - legend_circle_offset_y - 2 * legend_spacing,
        x1=legend_x - legend_circle_offset_x + legend_circle_diam,
        y1=legend_y - legend_circle_offset_y + legend_circle_diam - 2 * legend_spacing,
        line_color=dic_colours["ac"],
    )
    circle_se = dict(
        type="circle",
        xref="x",
        yref="y",
        fillcolor=dic_colours["sample-entropy-0"],
        x0=legend_x - legend_circle_offset_x,
        y0=legend_y - legend_circle_offset_y - 3 * legend_spacing,
        x1=legend_x - legend_circle_offset_x + legend_circle_diam,
        y1=legend_y - legend_circle_offset_y + legend_circle_diam - 3 * legend_spacing,
        line_color=dic_colours["sample-entropy-0"],
    )
    circle_ke = dict(
        type="circle",
        xref="x",
        yref="y",
        fillcolor=dic_colours["kolmogorov-entropy-0"],
        x0=legend_x - legend_circle_offset_x,
        y0=legend_y - legend_circle_offset_y - 4 * legend_spacing,
        x1=legend_x - legend_circle_offset_x + legend_circle_diam,
        y1=legend_y - legend_circle_offset_y + legend_circle_diam - 4 * legend_spacing,
        line_color=dic_colours["kolmogorov-entropy-0"],
    )

    title_annotation = dict(
        # x=sum(xrange)/2,
        x=0.5,
        y=1,
        text=title,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color="black", size=font_size),
    )

    # list_annotations.append(label_annotation)
    list_annotations.append(annotation_dl)
    list_annotations.append(annotation_var)
    list_annotations.append(annotation_ac)
    list_annotations.append(annotation_se)
    list_annotations.append(annotation_ke)

    # list_annotations.append(annotation_auc_var)
    # list_annotations.append(annotation_auc_ac)
    # list_annotations.append(annotation_N)
    # list_annotations.append(title_annotation)

    list_shapes.append(circle_dl)
    list_shapes.append(circle_var)
    list_shapes.append(circle_ac)
    list_shapes.append(circle_se)
    list_shapes.append(circle_ke)

    fig["layout"].update(annotations=list_annotations, shapes=list_shapes)

    # -------------
    # General layout properties
    # --------------

    # X axes properties
    fig.update_xaxes(
        title=dict(
            text="False positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickwidth=tickwidth,
        tickvals=np.arange(0, 1.1, 0.2),
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Y axes properties
    fig.update_yaxes(
        title=dict(
            text="True positive",
            standoff=axes_standoff,
        ),
        range=[-0.04, 1.04],
        ticks="outside",
        tickvals=np.arange(0, 1.1, 0.2),
        tickwidth=tickwidth,
        showline=True,
        linewidth=linewidth_axes,
        linecolor="black",
        mirror=False,
    )

    # Overall properties
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=5, b=15, t=5),
        font=dict(size=font_size, family=font_family),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        # legend=dict(
        #     x=0.6,
        #     y=1,
        #     tracegroupgap=0,
        #     font=dict(size=8),
        #     itemwidth=30,
        # ),
    )

    return fig


import seaborn as sns


def make_inset_boxplot(df_dl_forced, target_bif, save_dir):
    """
    Make inset boxplot that shows the value of the
    DL weights where the predictions are made

    """

    sns.set(
        style="ticks",
        rc={
            "figure.figsize": (2.5 * 1.05, 1.5 * 1.05),
            "axes.linewidth": 0.5,
            "axes.edgecolor": "#333F4B",
            "xtick.color": "#333F4B",
            "xtick.major.width": 0.5,
            "xtick.major.size": 3,
            "text.color": "#333F4B",
            "font.family": "Times New Roman",
            # 'font.size':20,
        },
        font_scale=1.2,
    )

    plt.figure()

    bif_types = [str(i) for i in np.arange(1, 4)]
    bif_labels = ["Fold", "TC", "PF"]
    map_bif = dict(zip(bif_types, bif_labels))
    df_plot = df_dl_forced[bif_types].melt(var_name="bif_type", value_name="DL prob")
    df_plot["bif_label"] = df_plot["bif_type"].map(map_bif)

    color_main = "#A9A9A9"
    # color_target = '#FFA15A'
    color_target = "#A9A9A9"
    col_palette = {bif: color_main for bif in bif_labels}
    col_palette[target_bif] = color_target

    b = sns.boxplot(
        df_plot,
        orient="h",
        x="DL prob",
        y="bif_label",
        width=0.8,
        palette=col_palette,
        linewidth=0.8,
        showfliers=False,
    )

    b.set(xlabel=None)
    b.set(ylabel=None)
    b.set_xticks([0, 0.5, 1])
    b.set_xticklabels(["0", "0.5", "1"])

    sns.despine(offset=3, trim=True)
    b.tick_params(left=False, bottom=True)

    fig = b.get_figure()
    # fig.tight_layout()

    fig.savefig(save_dir, dpi=330, bbox_inches="tight", pad_inches=0)


def combine_roc_inset(path_roc, path_inset, path_out):
    """
    Combine ROC plot and inset, and export to path_out
    """

    # Import image
    img_roc = Image.open(path_roc)
    img_inset = Image.open(path_inset)

    # Get height and width of frame (in pixels)
    height = img_roc.height
    width = img_roc.width

    # Create frame
    dst = Image.new("RGB", (width, height), (255, 255, 255))

    # Pasete in images
    dst.paste(img_roc, (0, 0))
    dst.paste(img_inset, (width - img_inset.width - 60, 1050))

    dpi = 96 * 8  # (default dpi) * (scaling factor)
    dst.save(path_out, dpi=(dpi, dpi))

    return


# --------
# Make fig
# --------

fig_roc = make_roc_figure(df_roc_full, "")
fig_roc.write_image("temp_roc.png", scale=scale)

# Do bif specific predictions for times 0.5 to 0.6
make_inset_boxplot(df_dl_forced.query("time>0.45"), "PD", "temp_inset.png")

# Combine figs and export
path_roc = "temp_roc.png"
path_inset = "temp_inset.png"
path_out = "fig_roc.png"

combine_roc_inset(path_roc, path_inset, path_out)

# #  Make pdf
path_out = "fig_roc.pdf".format(sample_size)
combine_roc_inset(path_roc, path_inset, path_out)
