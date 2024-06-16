#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:08:09 2023

Combine training batches into a single file and split into training, validation and test sets

@author: tbury
"""

import numpy as np
import pandas as pd

path = "output_80k/"

seed_vals = np.arange(10)

print("Load batches")
list_df = []
for seed in seed_vals:
    df = pd.read_parquet(path + f"seed_{seed}/df_out.parquet")
    list_df.append(df)

df_full = pd.concat(list_df)

# Split into train/validation/test (include shuffle)
split_ratio = (0.95, 0.025, 0.025)
# split_ratio = (0.5, 0.25, 0.25)
tsid_values = df_full["tsid"].unique()
max_index_train = int(split_ratio[0] * len(tsid_values))
max_index_val = int((split_ratio[0] + split_ratio[1]) * len(tsid_values))

print("Shuffle")
np.random.shuffle(tsid_values)
tsid_train = tsid_values[:max_index_train]
tsid_val = tsid_values[max_index_train:max_index_val]
tsid_test = tsid_values[max_index_val:]

df_out_train = df_full[df_full["tsid"].isin(tsid_train)]
df_out_val = df_full[df_full["tsid"].isin(tsid_val)]
df_out_test = df_full[df_full["tsid"].isin(tsid_test)]

print("{} train samples".format(len(df_out_train["tsid"].unique())))
print("{} val samples".format(len(df_out_val["tsid"].unique())))
print("{} test samples".format(len(df_out_test["tsid"].unique())))

# Export data
print("Export data")
df_out_train.to_parquet(path + "df_train.parquet")
df_out_val.to_parquet(path + "df_val.parquet")
df_out_test.to_parquet(path + "df_test.parquet")
