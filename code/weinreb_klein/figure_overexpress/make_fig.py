import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


# Import data
df_overexpress = pd.read_csv("../output/ews/df_dl_overexpress.csv")

# Use predictions within a certain time frame
t_low = 0.45
t_high = 0.6
df_overexpress = df_overexpress.query("@t_low <= time <= @t_high")

# Get average predictions over classifiers and time values
df_overexpress = (
    df_overexpress.groupby(["protocol"]).mean(numeric_only=True).reset_index()
)

# Extract the number of knockouts from the protocol column
df_overexpress["num_overexpress"] = df_overexpress["protocol"].apply(
    lambda x: x.split("_")[-1]
)

# Melt the DataFrame for plotting
df_plot = df_overexpress.melt(
    id_vars=["protocol", "num_overexpress"],
    value_vars=["0", "1", "2", "3"],
    var_name="bifurcation",
    value_name="DL probability",
)

plot_colors = px.colors.qualitative.Plotly

# Create a bar chart with Plotly Express
fig = px.bar(
    df_plot,
    x="num_overexpress",
    y="DL probability",
    color="bifurcation",
    labels={
        "num_overexpress": "Number overexpressed",
        "DL probability": "DL Probability",
    },
    category_orders={"num_overexpress": ["0", "5", "10", "20", "30", "40", "50"]},
    color_discrete_sequence=[plot_colors[i] for i in [0, 1, 3, 2]],
)

# Customize the appearance of the plot
fig.update_layout(
    # Set Times New Roman font
    font=dict(family="Times New Roman", size=16),  # Set Times New Roman font
    legend=dict(title="Bifurcation"),
    paper_bgcolor="white",  # Set the background color to white
    plot_bgcolor="white",  # Set the plot area background color to white
    margin=dict(l=50, r=50, t=20, b=50),  # Adjust margin to reduce white space
    bargap=0.2,  # Adjust bargap for spacing between bars
    width=600,
    height=400,
)
fig.update_yaxes(title="FateNet")

# Change the legend names
fig.data[0].name = "Null"
fig.data[1].name = "Fold"
fig.data[2].name = "Transcritical"
fig.data[3].name = "Pitchfork"

# Export as image
fig.write_image("fig_overexpress.png", scale=2)
fig.write_image("fig_overexpress.pdf", scale=2)
