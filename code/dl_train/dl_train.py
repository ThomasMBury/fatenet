#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:48:34 2021

Script to train DL classifier on wandb

Key for trajectory type:
    0 : Null trajectory
    1 : Fold trajectory
    2 : Transcritical trajectory
    3 : Pitchfork trajectory

Options for model_type
    1: random length U[50,ts_len] & random start time U[0,ts_len-L]
    2: random length U[50,ts_len] & start time = ts_len-L (use end of time series)

@author: tbury
"""

import time
import numpy as np
import datetime
import pytz
from typing import Tuple, List
import os
import random

# Get date and time in zone ET
newYorkTz = pytz.timezone("America/New_York")
datetime_now = datetime.datetime.now(newYorkTz).strftime("%Y%m%d-%H%M%S")

start = time.time()


# Wandb packages
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# For command line args
import tyro
from dataclasses import dataclass

import pandas as pd
import numpy as np
import ewstools

from tensorflow.random import set_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


### Define command line arguments
@dataclass
class Args:
    wandb_project_name: str = "FateNet"
    """wandb project name"""
    wandb_entity: str = "bub_lab"
    """wandb team name"""

    seed: int = 0
    """seed of the experiment"""
    model_type: int = 2
    """model type"""

    path_training: str = "../training_data/output_def/"
    """Path to import training data from"""

    detrend: bool = False
    """whether or not to detrend the training data"""

    # Hyperparameters
    pool_size: int = 2  # for maxpooling
    learning_rate: float = 0.0005
    batch_size: int = 64  # 1024
    num_epochs: int = 5
    dropout_percent: float = 0.10
    num_conv_layers: int = 1
    num_conv_filters: int = 50
    mem_cells_1: int = 50
    mem_cells_2: int = 10
    kernel_size: int = 12
    kernel_initializer: str = "lecun_normal"


# Get CLI arguments
args = tyro.cli(Args)
# Set unique run name for our session
local_id = ""
for i in range(5):
    local_id += random.choice("abcdefghijklmnopqrstuvwxyz")
args.run_name = f"{datetime_now}_m{args.model_type}_{local_id}"
print(args)

# Set seeds
np.random.seed(args.seed)
set_seed(args.seed)


# Initialise wandb
# wandb.login()
wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    name=args.run_name,
    config=vars(args),
    sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)

# Define metrics to get max accuracy (training) and max val_accurancy
wandb.define_metric("epoch/accuracy", summary="max")
wandb.define_metric("epoch/val_accuracy", summary="max")

df_train = pd.read_parquet(args.path_training + "df_train.parquet")
df_val = pd.read_parquet(args.path_training + "df_val.parquet")

# Pad and normalise the data
ts_len = 500


def prepare_series(series):
    """
    Prepare raw series data for training.

    Parameters:
        series: pd.Series of length ts_len
    """

    if args.detrend:
        # Detrend the series with a Lowess filter
        ts = ewstools.TimeSeries(series)
        ts.detrend(method="Lowess", span=0.2)
        series = ts.state["residuals"]

    # Length of sequence to extract
    L = np.random.choice(np.arange(50, ts_len + 1))

    # Start time of sequence to extract
    if args.model_type == 1:
        t0 = np.random.choice(np.arange(0, ts_len - L + 1))
    elif args.model_type == 2:
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


# Apply preprocessing to each series
ts_pad = df_train.groupby("tsid")["x"].transform(prepare_series)
df_train["x_pad"] = ts_pad

ts_pad = df_val.groupby("tsid")["x"].transform(prepare_series)
df_val["x_pad"] = ts_pad


# Put into numpy array with shape (samples, timesteps, features)
inputs_train = df_train["x_pad"].to_numpy().reshape(-1, ts_len, 1)
targets_train = df_train["type"].iloc[::ts_len].to_numpy().reshape(-1, 1)

inputs_val = df_val["x_pad"].to_numpy().reshape(-1, ts_len, 1)
targets_val = df_val["type"].iloc[::ts_len].to_numpy().reshape(-1, 1)

# print("Using {} training data samples".format(len(inputs_train)))
# print("Using {} validation data samples".format(len(inputs_val)))


# Set up NN architecture
model = Sequential()
model.add(
    Conv1D(
        filters=args.num_conv_filters,
        kernel_size=args.kernel_size,
        activation="relu",
        padding="same",
        input_shape=(ts_len, 1),
        kernel_initializer=args.kernel_initializer,
    )
)
if args.num_conv_layers == 2:
    model.add(
        Conv1D(
            filters=args.num_conv_filters,
            kernel_size=args.kernel_size,
            activation="relu",
            padding="same",
            kernel_initializer=args.kernel_initializer,
        )
    )

model.add(Dropout(args.dropout_percent))
model.add(MaxPooling1D(pool_size=args.pool_size))
model.add(
    LSTM(
        args.mem_cells_1,
        return_sequences=True,
        kernel_initializer=args.kernel_initializer,
    )
)
model.add(Dropout(args.dropout_percent))
model.add(LSTM(args.mem_cells_2, kernel_initializer=args.kernel_initializer))
model.add(Dropout(args.dropout_percent))
model.add(Dense(4, activation="softmax", kernel_initializer=args.kernel_initializer))

# Set up optimiser and checkpoints to save best model
adam = Adam(learning_rate=args.learning_rate)

# Compile Keras model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
)

# Callbacks
# model_name = "models/{}_".format(args.run_name) + "ep{epoch:02d}"
model_name = "models/{}".format(args.run_name)

callback_chk = WandbModelCheckpoint(
    model_name,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)
callback_metrics = WandbMetricsLogger(log_freq="epoch")

# print summary
print(model.summary())

# Train model
# print("Train NN with seed={} and model_type={}".format(args.seed, args.model_type))
history = model.fit(
    inputs_train,
    targets_train,
    epochs=args.num_epochs,
    batch_size=args.batch_size,
    callbacks=[callback_metrics, callback_chk],
    validation_data=(inputs_val, targets_val),
    verbose=2,
)

# Export history data (metrics evaluated on training and validation sets at each epoch)
df_history = pd.DataFrame(history.history)
df_history.to_csv(
    "training_history/df_history_{}.csv".format(args.run_name), index=False
)

end = time.time()
print("Script took {:0.1f} seconds".format(end - start))

wandb.finish()
