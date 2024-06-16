#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:29:12 2022

Simulate single trajectories of 2d gene model from Freedman et al.
https://journals.biologists.com/dev/article/150/11/dev201280/312613/A-dynamical-systems-treatment-of-transcriptomic

Compute FateNet predictions

@author: tbury
"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import ewstools

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)  # comment out to see TensorFlow warnings
from tensorflow.keras.models import load_model

# Load models
import glob

print("Load ML models")
path = "../dl_train/models/"

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

import plotly.express as px
import plotly.graph_objects as go

# noise amplitude
sigma = 0.05
sigma_x = sigma
sigma_y = sigma

path_out = f"output/sigma_{sigma}/"
path_out_figs = f"figures/sigma_{sigma}/"

os.makedirs(path_out, exist_ok=True)
os.makedirs(path_out_figs, exist_ok=True)


def simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=1):
    # Simulation parameters
    dt = 0.01
    t0 = 0
    tburn = 100

    # Initialise arrays to store single time-series data
    t = np.arange(t0, tmax, dt)
    g1 = np.zeros(len(t))
    g2 = np.zeros(len(t))

    # Model parameters
    if type(m1) == list:
        m1 = np.linspace(m1[0], m1[1], len(t))
    else:
        m1 = np.ones(int(tmax / dt)) * m1

    if type(k) == list:
        k = np.linspace(k[0], k[1], len(t))
    else:
        k = np.ones(int(tmax / dt)) * k

    def de_fun_g1(g1, g2, m1, k):
        return m1 / (1 + g2**2) - k * g1

    def de_fun_g2(g1, g2, m2, k):
        return m2 / (1 + c * g1**2) - k * g2

    # Create brownian increments (s.d. sqrt(dt))
    dW_x_burn = np.random.normal(
        loc=0, scale=sigma_x * np.sqrt(dt), size=int(tburn / dt)
    )
    dW_x = np.random.normal(loc=0, scale=sigma_x * np.sqrt(dt), size=len(t))

    dW_y_burn = np.random.normal(
        loc=0, scale=sigma_y * np.sqrt(dt), size=int(tburn / dt)
    )
    dW_y = np.random.normal(loc=0, scale=sigma_y * np.sqrt(dt), size=len(t))

    # Run burn-in period on x0
    for i in range(int(tburn / dt)):
        g10 = g10 + de_fun_g1(g10, g20, m1[0], k[0]) * dt + dW_x_burn[i]
        g20 = g20 + de_fun_g2(g10, g20, m2, k[0]) * dt + dW_y_burn[i]

    # Initial condition post burn-in period
    g1[0] = g10
    g2[0] = g20

    # Run simulation
    for i in range(len(t) - 1):
        g1[i + 1] = g1[i] + de_fun_g1(g1[i], g2[i], m1[i], k[i]) * dt + dW_x[i]
        g2[i + 1] = g2[i] + de_fun_g2(g1[i], g2[i], m2, k[i]) * dt + dW_y[i]

    # Store series data in a temporary DataFrame
    data = {"time": t, "g1": g1, "g2": g2}
    df = pd.DataFrame(data)

    return df


# -------------
# Saddle node trajectory
# --------------
np.random.seed(7)

ncells = 20
tmax = 750
g10 = 0
g20 = 0
m1bif = 3.6  # 3.5 for very large noise to account for noise-induced transition
m1start = 1
m1end = m1start + (m1bif - m1start) * 1.5

m2 = 3
k = 1
c = 1

list_df_traj = []

for cell in range(ncells):
    # Forced trajectory
    m1 = [m1start, m1end]
    df = simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax)
    df = df.iloc[::100]

    df["cell"] = cell
    list_df_traj.append(df)

    print("Simulation done for cell {}".format(cell))

df_traj = pd.concat(list_df_traj)

# Stitch together individual cell trajectories to make pseudotime trajectories
mat_cell_g1 = (
    df_traj[["g1", "time", "cell"]].pivot(columns=["cell"], index="time").values
)
mat_cell_g2 = (
    df_traj[["g2", "time", "cell"]].pivot(columns=["cell"], index="time").values
)

# Re-ordered to give pseduto time series
mat_pseudo_g1 = np.zeros((tmax, ncells))
mat_pseudo_g2 = np.zeros((tmax, ncells))

for j in range(ncells):
    for i in range(tmax):
        # take diagonal that loops around
        mat_pseudo_g1[i, j] = mat_cell_g1[i, (i + j) % ncells]
        mat_pseudo_g2[i, j] = mat_cell_g2[i, (i + j) % ncells]

df_pseudo_g1 = pd.DataFrame(mat_pseudo_g1)
df_pseudo_g1 = df_pseudo_g1.reset_index(names="time")
df_pseudo_g1 = df_pseudo_g1.melt(id_vars="time", var_name="pseudo", value_name="g1")

id_plot = 1

########## Compute EWS
series = df_pseudo_g1.query("pseudo==@id_plot")[["g1", "time"]].set_index("time")["g1"]
ts = ewstools.TimeSeries(series, transition=500)
ts.detrend(method="Lowess", span=0.2)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)
print("Apply classifiers")
for idx, c1 in enumerate(list_models_1):
    ts.apply_classifier_inc(c1, inc=10, name=f"c1_{idx}", verbose=0)
for idx, c2 in enumerate(list_models_2):
    ts.apply_classifier_inc(c2, inc=10, name=f"c2_{idx}", verbose=0)

# Make a temp figure to view
fig = ts.make_plotly(ens_avg=True)
fig.update_layout(height=650)
fig.write_html(path_out_figs + "ews_fold.html")

# Export ews
df_dl = ts.dl_preds.groupby("time").mean(numeric_only=True)
df_dl.index.name = "time"
df_dl = df_dl[df_dl.index > 100]
df_out = ts.state
df_out = df_out.join(ts.ews, on="time")
df_out = df_out.join(df_dl, on="time")
df_out.to_csv(path_out + "df_fold_forced.csv")

# -----------------
# Make bifurcation diagram
# -----------------

# Get roots of cubic polynomial
m1_vals = np.linspace(m1start, m1end, 1000)

list_roots_g1 = []
list_roots_g2 = []

for m1 in m1_vals:
    # print(a0, m2, k)
    # print(type(a0), type(m2), type(k))
    a0 = -m2 * (k**2)
    a1 = (k**3) + k * (m1**2)
    a2 = -2 * c * m2 * (k**2)
    a3 = 2 * c * (k**3)
    a4 = -m2 * (c**2) * (k**2)
    a5 = (c**2) * k**3

    p = [a5, a4, a3, a2, a1, a0]
    roots_g2 = np.roots(p)
    roots_g1 = m1 / (k * (1 + roots_g2**2))
    list_roots_g1.append(roots_g1)
    list_roots_g2.append(roots_g2)

ar1 = np.array(list_roots_g1)
ar2 = np.array(list_roots_g2)


df_roots_g1 = pd.DataFrame()
df_roots_g2 = pd.DataFrame()
df_roots_g1["m1"] = m1_vals
df_roots_g2["m1"] = m1_vals

for i in np.arange(5):
    df_roots_g1[str(i)] = ar1[:, i]
    df_roots_g2[str(i)] = ar2[:, i]


def get_real(col):
    np.imag(col) == 0

    np.real(col)

    real_vals = np.real(col)

    pure_real = real_vals.copy()
    pure_real[np.imag(col) != 0] = np.nan

    out = pd.Series(pure_real, index=col.index)
    return out


cols = df_roots_g1.columns
for col in cols:
    df_roots_g1["{}_real".format(col)] = get_real(df_roots_g1[col])
    df_roots_g2["{}_real".format(col)] = get_real(df_roots_g2[col])

df_plot_g1 = df_roots_g1.melt(
    id_vars="m1", value_vars=["0_real", "1_real", "2_real", "3_real", "4_real"]
)

# Export
df_plot_g1.dropna(inplace=True)
df_plot_g1.to_csv(path_out + "df_fold_bif.csv", index=False)

# -------------
# Pitchfork trajectory
# --------------

np.random.seed(8)

tmax = 750

g10 = 1
g20 = 1
m1 = 1
m2 = 1
k = [1, 1 / 4]
kbif = 1 / 2
c = 1


list_df_traj = []

for cell in range(ncells):
    # Forced trajectory
    df = simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=c)
    df = df.iloc[::100]

    df["cell"] = cell
    list_df_traj.append(df)

    print("Simulation done for cell {}".format(cell))

df_traj = pd.concat(list_df_traj)

# Stitch together individual cell trajectories to make pseudotime trajectories
mat_cell = df_traj[["g1", "time", "cell"]].pivot(columns=["cell"], index="time").values

# Re-ordered to give pseduto time series
mat_pseudo = np.zeros((tmax, ncells))
for j in range(ncells):
    for i in range(tmax):
        # take diagonal that loops around
        mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

df_pseudo = pd.DataFrame(mat_pseudo)
df_pseudo = df_pseudo.reset_index(names="time")
df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")

id_plot = 1
df_pseudo.query("pseudo==@id_plot")["x"].plot()


series = df_pseudo.query("pseudo==@id_plot")[["x", "time"]].set_index("time")["x"]
ts = ewstools.TimeSeries(series, transition=500)
# ts.detrend(method='Gaussian', bandwidth=0.2)
ts.detrend(method="Lowess", span=0.2)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)
print("Apply classifiers")
for idx, c1 in enumerate(list_models_1):
    ts.apply_classifier_inc(c1, inc=10, name=f"c1_{idx}", verbose=0)
for idx, c2 in enumerate(list_models_2):
    ts.apply_classifier_inc(c2, inc=10, name=f"c2_{idx}", verbose=0)


fig = ts.make_plotly(ens_avg=True)
fig.update_layout(height=650)
fig.write_html(path_out_figs + "ews_pf.html")

# Export ews
df_dl = ts.dl_preds.groupby("time").mean(numeric_only=True)
df_dl.index.name = "time"
df_dl = df_dl[df_dl.index > 100]
df_out = ts.state
df_out = df_out.join(ts.ews, on="time")
df_out = df_out.join(df_dl, on="time")
df_out.to_csv(path_out + "df_pf_forced.csv")


# -----------
# Bifurcation diagram for pitchfork
# -----------

# Vary k values for pitchfork
k_vals = np.linspace(k[0], k[1], 5000)

list_roots = []

for k in k_vals:
    a0 = -m2 * (k**2)
    a1 = (k**3) + k * (m1**2)
    a2 = -2 * c * m2 * (k**2)
    a3 = 2 * c * (k**3)
    a4 = -m2 * (c**2) * (k**2)
    a5 = (c**2) * k**3

    p = [a5, a4, a3, a2, a1, a0]
    roots_g2 = np.roots(p)
    roots_g1 = m1 / (k * (1 + roots_g2**2))
    list_roots.append(roots_g1)

ar = np.array(list_roots)

df_roots = pd.DataFrame()
df_roots["k"] = k_vals
for i in np.arange(5):
    df_roots[str(i)] = ar[:, i]

cols = df_roots.columns
for col in cols:
    df_roots["{}_real".format(col)] = get_real(df_roots[col])

df_plot = df_roots.melt(
    id_vars="k", value_vars=["0_real", "1_real", "2_real", "3_real", "4_real"]
)

# fig = px.line(df_plot, x="k", y="value", color="variable")
# # fig.write_html('figures/bif_pf.html')

# Export
df_plot.dropna(inplace=True)
df_plot.to_csv(path_out + "df_pf_bif.csv", index=False)


# -------------
# Null trajectory (model without a bifurcation)
# --------------

## Simulate trajectory
np.random.seed(3)

tmax = 750

g10 = 0
g20 = 0
m1bif = 3.6
# m1start = 2.5
m1start = 1
m1end = m1start + (m1bif - m1start) * 1.5
m1 = [m1start, m1end]


m2 = 3
k = 1
c = 0.1


list_df_traj = []

for cell in range(ncells):
    # Forced trajectory
    m1 = [m1start, m1end]
    df = simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=c)
    df = df.iloc[::100]

    df["cell"] = cell
    list_df_traj.append(df)

    print("Simulation done for cell {}".format(cell))

df_traj = pd.concat(list_df_traj)

# Stitch together individual cell trajectories to make pseudotime trajectories
mat_cell = df_traj[["g1", "time", "cell"]].pivot(columns=["cell"], index="time").values

# Re-ordered to give pseduto time series
mat_pseudo = np.zeros((tmax, ncells))
for j in range(ncells):
    for i in range(tmax):
        # take diagonal that loops around
        mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

df_pseudo = pd.DataFrame(mat_pseudo)
df_pseudo = df_pseudo.reset_index(names="time")
df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")

id_plot = 1
df_pseudo.query("pseudo==@id_plot")["x"].plot()

series = df_pseudo.query("pseudo==@id_plot")[["x", "time"]].set_index("time")["x"]


ts = ewstools.TimeSeries(series, transition=500)
ts.detrend(method="Lowess", span=0.2)
ts.compute_var(rolling_window=0.25)
ts.compute_auto(rolling_window=0.25, lag=1)
print("Apply classifiers")
for idx, c1 in enumerate(list_models_1):
    ts.apply_classifier_inc(c1, inc=10, name=f"c1_{idx}", verbose=0)
for idx, c2 in enumerate(list_models_2):
    ts.apply_classifier_inc(c2, inc=10, name=f"c2_{idx}", verbose=0)

# Make a temp figure to view
fig = ts.make_plotly(ens_avg=True)
fig.update_layout(height=650)
fig.write_html(path_out_figs + "ews_null.html")

# Export ews
df_dl = ts.dl_preds.groupby("time").mean(numeric_only=True)
df_dl.index.name = "time"
df_dl = df_dl[df_dl.index > 100]
df_out = ts.state
df_out = df_out.join(ts.ews, on="time")
df_out = df_out.join(df_dl, on="time")
df_out.to_csv(path_out + "df_null.csv")

# -----------------
# Get bifurcation diagram
# -----------------


# Get roots of cubic polynomial
m1_vals = np.linspace(m1start, m1end, 1000)

list_roots = []

for m1 in m1_vals:
    a0 = -m1 * (k**2)
    a1 = (k**3) + k * (m2**2)
    a2 = -2 * c * m1 * (k**2)
    a3 = 2 * c * (k**3)
    a4 = -m1 * (c**2) * (k**2)
    a5 = (c**2) * k**3

    p = [a5, a4, a3, a2, a1, a0]
    roots_g1 = np.roots(p)
    list_roots.append(roots_g1)

ar = np.array(list_roots)

df_roots = pd.DataFrame()
df_roots["m1"] = m1_vals
for i in np.arange(5):
    df_roots[str(i)] = ar[:, i]


def get_real(col):
    np.imag(col) == 0

    np.real(col)

    real_vals = np.real(col)

    pure_real = real_vals.copy()
    pure_real[np.imag(col) != 0] = np.nan

    out = pd.Series(pure_real, index=col.index)
    return out


cols = df_roots.columns
for col in cols:
    df_roots["{}_real".format(col)] = get_real(df_roots[col])

df_plot = df_roots.melt(
    id_vars="m1", value_vars=["0_real", "1_real", "2_real", "3_real", "4_real"]
)

# Export
df_plot.dropna(inplace=True)
df_plot.to_csv(path_out + "df_null_bif.csv", index=False)
