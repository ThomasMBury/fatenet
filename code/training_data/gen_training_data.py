#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17 Feb 2022

Generate training data for DL classifier.
Uses normal forms + randomly generated higher-order terms.
Reshuffle to obtain pseudotime series

Key for trajectory type:
    0 : Null trajectory
    1 : Fold trajectory
    2 : Transcritical trajectory
    3 : Pitchfork trajectory

@author: tbury
"""

import time

start = time.time()

import numpy as np
import pandas as pd

import train_funs as funs

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nsims", type=int, help="Number of simulations for each model", default=5
)
parser.add_argument("--verbose", type=int, choices=[0, 1], default=1)
parser.add_argument(
    "--ncells", type=int, default=20, help="Number of cells to simulate for each model"
)
parser.add_argument(
    "--path_out",
    type=str,
    default="output_def/",
    help="Path to store output training data",
)

args = parser.parse_args()
nsims = args.nsims
path_out = args.path_out
verbose = bool(args.verbose)
ncells = args.ncells

# Make output directory
import os

if not os.path.exists(path_out):
    os.makedirs(path_out)

# Fix random number seed for reproducibility
np.random.seed(1)

tmax = 600
tburn = 100
ts_len = 500  #  length of time series to store for training
max_order = 10  # Max polynomial degree

# Noise amplitude distribution parameters (uniform dist.)
sigma_min = 0.005
sigma_max = 0.015

# Number of standard deviations from equilibrium that defines a transition
thresh_std = 10

# List to collect time series
list_df = []
tsid = 1  # time series id

# total number of pseudotime seires will be nsims * ncells * nmodels
print("total number of time series for training data = {}".format(nsims * ncells * 4))

print("Run forced fold simulations")
sim = 1
while sim <= nsims:
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)

    # Deviation that defines transition
    dev_thresh = sigma * thresh_std

    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-0.25, -0.01)

    # Orientation of fold (positive or negative stable branch)
    orient = np.random.choice([-1, 1])

    # Eigenvalue lambda = 1-2*sqrt(-mu)
    # Lower bound csp to lambda=0
    # Upper bound csp to lambda=0.8
    bh = 0

    # Run simulation for each cell
    list_df_cell = []
    cell = 1
    while cell <= ncells:
        df_traj = funs.simulate_fold(
            bl=bl,
            bh=bh,
            tmax=tmax,
            tburn=tburn,
            sigma=sigma,
            max_order=max_order,
            dev_thresh=dev_thresh,
            return_dev=True,
            orient=orient,
        )
        df_traj["cell"] = cell

        # Drop Nans and keep only last ts_len points
        df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
        if len(df_traj) < ts_len:
            if verbose:
                print(
                    "Forced fold trajectory diverged too early with sigma={}".format(
                        sigma
                    )
                )
            continue

        df_traj["time"] = np.arange(0, ts_len)
        list_df_cell.append(df_traj)

        # if cell%10==0:
        #     print('Complete for cell {}'.format(cell))
        cell += 1

    df_cells = pd.concat(list_df_cell)

    # Create pseudotime series by combining points from each cell
    mat_cell = (
        df_cells[["time", "x", "cell"]].pivot(index="time", columns="cell").values
    )

    # Re-ordered to give pseudo time series
    mat_pseudo = np.zeros((ts_len, ncells))
    for j in range(ncells):
        for i in range(ts_len):
            # take diagonal that loops around
            mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

    df_pseudo = pd.DataFrame(mat_pseudo)
    df_pseudo = df_pseudo.reset_index(names="time")
    df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")
    df_pseudo["tsid"] = tsid + df_pseudo["pseudo"]

    df_pseudo["bif_type"] = "fold_forced"
    df_pseudo["type"] = 1
    df_pseudo["sigma"] = sigma
    list_df.append(df_pseudo)
    if verbose:
        print("Complete for tsid={}".format(tsid))
    elif tsid % 1000 == 0:
        print("Complete for tsid={}".format(tsid))
    tsid += ncells
    sim += 1


print("Run null fold simulations")
sim = 1
while sim <= nsims:
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)

    # Deviation that defines transition
    dev_thresh = sigma * thresh_std

    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-0.25, -0.01)

    bh = bl

    # Orientation of    fold (positive or negative stable branch)
    orient = np.random.choice([-1, 1])

    # Run simulation for each cell
    list_df_cell = []
    cell = 1
    while cell <= ncells:
        df_traj = funs.simulate_fold(
            bl=bl,
            bh=bh,
            tmax=tmax,
            tburn=tburn,
            sigma=sigma,
            max_order=max_order,
            dev_thresh=dev_thresh,
            return_dev=True,
            orient=orient,
        )
        df_traj["cell"] = cell

        # Drop Nans and keep only last ts_len points
        df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
        if len(df_traj) < ts_len:
            if verbose:
                print(
                    "Forced fold trajectory diverged too early with sigma={}".format(
                        sigma
                    )
                )
            continue

        df_traj["time"] = np.arange(0, ts_len)
        list_df_cell.append(df_traj)

        # if cell%10==0:
        #     print('Complete for cell {}'.format(cell))
        cell += 1

    df_cells = pd.concat(list_df_cell)

    # Create pseudotime series by combining points from each cell
    mat_cell = (
        df_cells[["time", "x", "cell"]].pivot(index="time", columns="cell").values
    )

    # Re-ordered to give pseduto time series
    mat_pseudo = np.zeros((ts_len, ncells))
    for j in range(ncells):
        for i in range(ts_len):
            # take diagonal that loops around
            mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

    df_pseudo = pd.DataFrame(mat_pseudo)
    df_pseudo = df_pseudo.reset_index(names="time")
    df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")
    df_pseudo["tsid"] = tsid + df_pseudo["pseudo"]

    df_pseudo["bif_type"] = "fold_null"
    df_pseudo["type"] = 0
    df_pseudo["sigma"] = sigma
    list_df.append(df_pseudo)
    if verbose:
        print("Complete for tsid={}".format(tsid))
    elif tsid % 1000 == 0:
        print("Complete for tsid={}".format(tsid))
    tsid += ncells
    sim += 1

df_full = pd.concat(list_df, ignore_index=True)


print("Run forced TC simulations")
sim = 1
while sim <= nsims:
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)

    # Deviation that defines transition
    dev_thresh = sigma * thresh_std

    # Draw starting value of bifurcaiton parameter at random
    # Eigenvalue lambda = 1+mu
    # Lower bound csp to lambda= 0
    # Upper bound csp to lambda=0.8

    bl = np.random.uniform(-1, -0.2)
    bh = 0

    # Run simulation for each cell
    list_df_cell = []
    cell = 1
    while cell <= ncells:
        df_traj = funs.simulate_tc(bl, bh, tmax, tburn, sigma, max_order, dev_thresh)
        df_traj["cell"] = cell

        # Drop Nans and keep only last ts_len points
        df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
        if len(df_traj) < ts_len:
            if verbose:
                print(
                    "Forced fold trajectory diverged too early with sigma={}".format(
                        sigma
                    )
                )
            continue

        df_traj["time"] = np.arange(0, ts_len)
        list_df_cell.append(df_traj)

        # if cell%10==0:
        #     print('Complete for cell {}'.format(cell))

        cell += 1

    df_cells = pd.concat(list_df_cell)

    # Create pseudotime series by combining points from each cell
    mat_cell = (
        df_cells[["time", "x", "cell"]].pivot(index="time", columns="cell").values
    )

    # Re-ordered to give pseduto time series
    mat_pseudo = np.zeros((ts_len, ncells))
    for j in range(ncells):
        for i in range(ts_len):
            # take diagonal that loops around
            mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

    df_pseudo = pd.DataFrame(mat_pseudo)
    df_pseudo = df_pseudo.reset_index(names="time")
    df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")
    df_pseudo["tsid"] = tsid + df_pseudo["pseudo"]

    df_pseudo["bif_type"] = "tc_forced"
    df_pseudo["type"] = 2
    df_pseudo["sigma"] = sigma
    list_df.append(df_pseudo)
    if verbose:
        print("Complete for tsid={}".format(tsid))
    elif tsid % 1000 == 0:
        print("Complete for tsid={}".format(tsid))
    tsid += ncells
    sim += 1


print("Run null TC simulations")
sim = 1
while sim <= nsims:
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)

    # Deviation that defines transition
    dev_thresh = sigma * thresh_std

    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-1, -0.2)
    bh = bl

    # Run simulation for each cell
    list_df_cell = []
    cell = 1
    while cell <= ncells:
        df_traj = funs.simulate_tc(bl, bh, tmax, tburn, sigma, max_order, dev_thresh)
        df_traj["cell"] = cell

        # Drop Nans and keep only last ts_len points
        df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
        if len(df_traj) < ts_len:
            if verbose:
                print(
                    "Forced fold trajectory diverged too early with sigma={}".format(
                        sigma
                    )
                )
            continue

        df_traj["time"] = np.arange(0, ts_len)
        list_df_cell.append(df_traj)

        # if cell%10==0:
        #     print('Complete for cell {}'.format(cell))
        cell += 1

    df_cells = pd.concat(list_df_cell)

    # Create pseudotime series by combining points from each cell
    mat_cell = (
        df_cells[["time", "x", "cell"]].pivot(index="time", columns="cell").values
    )

    # Re-ordered to give pseduto time series
    mat_pseudo = np.zeros((ts_len, ncells))
    for j in range(ncells):
        for i in range(ts_len):
            # take diagonal that loops around
            mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

    df_pseudo = pd.DataFrame(mat_pseudo)
    df_pseudo = df_pseudo.reset_index(names="time")
    df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")
    df_pseudo["tsid"] = tsid + df_pseudo["pseudo"]

    df_pseudo["bif_type"] = "tc_null"
    df_pseudo["type"] = 0
    df_pseudo["sigma"] = sigma
    list_df.append(df_pseudo)
    if verbose:
        print("Complete for tsid={}".format(tsid))
    elif tsid % 1000 == 0:
        print("Complete for tsid={}".format(tsid))
    tsid += ncells
    sim += 1


print("Run forced PF simulations")
sim = 1
while sim <= nsims:
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)

    # Deviation that defines transition
    dev_thresh = sigma * thresh_std

    # Draw starting value of bifurcaiton parameter at random
    # Eigenvalue lambda = 1+mu
    # Lower bound csp to lambda=0
    # Upper bound csp to lambda=0.8
    bl = np.random.uniform(-1, -0.2)
    bh = 0
    supercrit = bool(np.random.choice([0, 1]))

    # Run simulation for each cell
    list_df_cell = []
    cell = 1
    while cell <= ncells:
        df_traj = funs.simulate_pf(
            bl, bh, tmax, tburn, sigma, max_order, dev_thresh, supercrit
        )
        df_traj["cell"] = cell

        # Drop Nans and keep only last ts_len points
        df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
        if len(df_traj) < ts_len:
            if verbose:
                print(
                    "Forced PF trajectory diverged too early with sigma={}".format(
                        sigma
                    )
                )
            continue

        df_traj["time"] = np.arange(0, ts_len)
        list_df_cell.append(df_traj)

        # if cell%10==0:
        #     print('Complete for cell {}'.format(cell))
        cell += 1

    df_cells = pd.concat(list_df_cell)

    # Create pseudotime series by combining points from each cell
    mat_cell = (
        df_cells[["time", "x", "cell"]].pivot(index="time", columns="cell").values
    )

    # Re-ordered to give pseduto time series
    mat_pseudo = np.zeros((ts_len, ncells))
    for j in range(ncells):
        for i in range(ts_len):
            # take diagonal that loops around
            mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

    df_pseudo = pd.DataFrame(mat_pseudo)
    df_pseudo = df_pseudo.reset_index(names="time")
    df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")
    df_pseudo["tsid"] = tsid + df_pseudo["pseudo"]

    df_pseudo["bif_type"] = "pf_forced"
    df_pseudo["type"] = 3
    df_pseudo["sigma"] = sigma
    df_pseudo["supercrit"] = supercrit
    list_df.append(df_pseudo)
    if verbose:
        print("Complete for tsid={}".format(tsid))
    elif tsid % 1000 == 0:
        print("Complete for tsid={}".format(tsid))
    tsid += ncells
    sim += 1


print("Run null PF simulations")
sim = 1
while sim <= nsims:
    # Set noise amplitude
    sigma = np.random.uniform(sigma_min, sigma_max)

    # Deviation that defines transition
    dev_thresh = sigma * thresh_std

    # Draw starting value of bifurcaiton parameter at random
    bl = np.random.uniform(-1, -0.2)

    bh = bl
    supercrit = bool(np.random.choice([0, 1]))

    # Run simulation for each cell
    list_df_cell = []
    cell = 1
    while cell <= ncells:
        df_traj = funs.simulate_pf(
            bl, bh, tmax, tburn, sigma, max_order, dev_thresh, supercrit
        )
        df_traj["cell"] = cell

        # Drop Nans and keep only last ts_len points
        df_traj = df_traj.dropna(axis=0).iloc[-ts_len:]
        if len(df_traj) < ts_len:
            if verbose:
                print(
                    "Null PF trajectory diverged too early with sigma={}".format(sigma)
                )
            continue

        df_traj["time"] = np.arange(0, ts_len)
        list_df_cell.append(df_traj)

        # if cell%10==0:
        #     print('Complete for cell {}'.format(cell))
        cell += 1

    df_cells = pd.concat(list_df_cell)

    # Create pseudotime series by combining points from each cell
    mat_cell = (
        df_cells[["time", "x", "cell"]].pivot(index="time", columns="cell").values
    )

    # Re-ordered to give pseduto time series
    mat_pseudo = np.zeros((ts_len, ncells))
    for j in range(ncells):
        for i in range(ts_len):
            # take diagonal that loops around
            mat_pseudo[i, j] = mat_cell[i, (i + j) % ncells]

    df_pseudo = pd.DataFrame(mat_pseudo)
    df_pseudo = df_pseudo.reset_index(names="time")
    df_pseudo = df_pseudo.melt(id_vars="time", var_name="pseudo", value_name="x")
    df_pseudo["tsid"] = tsid + df_pseudo["pseudo"]

    df_pseudo["bif_type"] = "pf_null"
    df_pseudo["type"] = 0
    df_pseudo["sigma"] = sigma
    df_pseudo["supercrit"] = supercrit
    list_df.append(df_pseudo)
    if verbose:
        print("Complete for tsid={}".format(tsid))
    elif tsid % 1000 == 0:
        print("Complete for tsid={}".format(tsid))
    tsid += ncells
    sim += 1


# -------------
# Concatenate data
# --------------

df_full = pd.concat(list_df, ignore_index=True)

# Make classes of equal size : currently 5 times number of samples in null section
# Get null class
tsid_null = df_full[df_full["type"] == 0]["tsid"].unique()
# Downsample - Take random selection
tsid_null_down = np.random.choice(
    tsid_null, size=int(len(tsid_null) / 3), replace=False
)
df_null = df_full[df_full["tsid"].isin(tsid_null_down)]
df_not_null = df_full[df_full["type"] != 0]
df_full_balanced = pd.concat((df_null, df_not_null))


# Set types to save disk space
df_out = pd.DataFrame()
df_out["tsid"] = df_full_balanced["tsid"].astype("int32")
df_out["time"] = df_full_balanced["time"].astype("int32")
df_out["x"] = df_full_balanced["x"].astype("float32")
df_out["type"] = df_full_balanced["type"].astype("category")
df_out["bif_type"] = df_full_balanced["bif_type"].astype("category")


# Split into train/validation/test (include shuffle)
split_ratio = (0.95, 0.025, 0.025)
# split_ratio = (0.5, 0.25, 0.25)
tsid_values = df_out["tsid"].unique()
max_index_train = int(split_ratio[0] * len(tsid_values))
max_index_val = int((split_ratio[0] + split_ratio[1]) * len(tsid_values))

np.random.shuffle(tsid_values)
tsid_train = tsid_values[:max_index_train]
tsid_val = tsid_values[max_index_train:max_index_val]
tsid_test = tsid_values[max_index_val:]

df_out_train = df_out[df_out["tsid"].isin(tsid_train)]
df_out_val = df_out[df_out["tsid"].isin(tsid_val)]
df_out_test = df_out[df_out["tsid"].isin(tsid_test)]

# Export data
df_out_train.to_parquet(path_out + "df_train.parquet")
df_out_val.to_parquet(path_out + "df_val.parquet")
df_out_test.to_parquet(path_out + "df_test.parquet")

end = time.time()
print("Script took {:0.1f} seconds".format(end - start))
