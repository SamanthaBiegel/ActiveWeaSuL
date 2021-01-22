# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.7.6 64-bit (''thesis'': conda)'
#     language: python
#     name: python37664bitthesiscondab786fa2bf8ea4d8196bc19b4ba146cff
# ---

# +
# %load_ext autoreload
# %autoreload 2

import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath("../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -

joined_df_models = pd.read_csv("results/re_marg_random_joined.csv")
joined_df_approaches = pd.read_csv("results/awsl_hybrid.csv").rename(columns={"Strategy": "Approach"})
joined_df_approaches["Dash"] = "n"

joined_df_approaches["Approach"] = joined_df_approaches["Approach"].str.replace('Relative Entropy','Active WeaSuL')
joined_df_approaches["Approach"] = joined_df_approaches["Approach"].str.replace('Hybrid','Nashaat et al.')

optimal = joined_df_models[(joined_df_models["Strategy"] == "GM*") | (joined_df_models["Strategy"] == "DM*")]
optimal = optimal.rename(columns={"Strategy": "Approach"})
optimal["Approach"] = "*"
optimal["Dash"] = "y"

# +
accuracy_df = pd.read_csv("results/accuracy_only_AL.csv")

accuracy_df["Model"] = "Discriminative"
accuracy_df["Approach"] = "Active Learning"
accuracy_df["Metric"] = "Accuracy"
accuracy_df["Dash"] = "n"
accuracy_df = accuracy_df.drop(columns=["Unnamed: 0"])

# +
WS_baseline = joined_df_approaches[(joined_df_approaches["Active Learning Iteration"] == 0) & (joined_df_approaches["Approach"] == "Active WeaSuL")].groupby("Model").mean()["Accuracy"]

baseline_fm = pd.DataFrame({"Active Learning Iteration": list(range(31)), "Accuracy": np.repeat(WS_baseline["Discriminative"], 31)})
baseline_fm["Model"] = "Discriminative"
baseline_fm["Run"] = 0
baseline_fm["Approach"] = "Weak Supervision"
baseline_fm["Metric"] = "Accuracy"
baseline_fm["Dash"] = "n"

baseline_lm = pd.DataFrame({"Active Learning Iteration": list(range(31)), "Accuracy": np.repeat(WS_baseline["Generative"], 31)})
baseline_lm["Model"] = "Generative"
baseline_lm["Run"] = 0
baseline_lm["Approach"] = "Weak Supervision"
baseline_lm["Metric"] = "Accuracy"
baseline_lm["Dash"] = "n"

joined_df = pd.concat([joined_df_approaches, accuracy_df, optimal, baseline_lm, baseline_fm])
# -

joined_df = joined_df.rename(columns={"Active Learning Iteration": "Number of labeled points"})

joined_df

joined_df.to_csv("results/combined_baselines.csv", index=False)








