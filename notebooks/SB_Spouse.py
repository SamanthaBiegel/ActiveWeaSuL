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
import pickle
import random
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

sys.path.append(os.path.abspath("../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -

# ## Load data

# Label matrices generated from https://github.com/snorkel-team/snorkel-tutorials/blob/master/spouse/spouse_demo.ipynb

L_train = np.load("../data/spouse/L_train_spouse.npy")
L_dev = np.load("../data/spouse/L_dev_spouse.npy")
y_dev = np.load("../data/spouse/Y_dev_spouse.npy")
L_test = np.load("../data/spouse/L_test_spouse.npy")
y_test = np.load("../data/spouse/Y_test_spouse.npy")

# Drop rows with only abstains

all_abstain = (L_dev == -1).sum(axis=1) == 9
L_dev = L_dev[~all_abstain]
y_dev = y_dev[~all_abstain]

all_abstain = (L_test == -1).sum(axis=1) == 9
L_test = L_test[~all_abstain]
y_test = y_test[~all_abstain]

# +
# L_train[L_train == -1] = 0
# L_dev[L_dev == -1] = 0
# L_test[L_test == -1] = 0
# -

# ## Initial label model fit

y_dev.mean()

# +
class_balance = np.array([0.93,0.07])

cliques = [[0],[1],[2],[3],[4],[5],[6],[7],[8]]
# -

# Note: we use the dev set as train set for now

# +
lm = LabelModel(y_true=y_dev,
                n_epochs=200,
                lr=1e-1)

# Fit and predict on train set
Y_probs = lm.fit(label_matrix=L_dev, cliques=cliques, class_balance=class_balance).predict()
# Predict on test set
Y_probs_test = lm._predict(L_test, lm.mu, 0.07)
# Analyze test set performance
lm._analyze(Y_probs_test, y_test)
# -

# ## Active learning pipeline

al_kwargs = {'active_learning': "probs",
             'y_true': y_dev,
             'n_epochs': 200
            }

# +
it = 30
# Choose strategy from ["maxkl", "margin", "nashaat"]
query_strategy = "margin"

al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            query_strategy=query_strategy)

Y_probs_al = al.refine_probabilities(label_matrix=L_dev,
                                     cliques=cliques,
                                     class_balance=class_balance,
                                     label_matrix_test=L_test,
                                     y_test=y_test)
# -

# ## Analyze results

# +
dev_metrics = pd.DataFrame.from_dict(al.metrics).drop("Labels", errors="ignore").T
dev_metrics = dev_metrics.stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Metric", 0: "Value"})

dev_metrics["Value"] = dev_metrics["Value"].apply(pd.to_numeric)
dev_metrics["Set"] = "dev"

test_metrics = pd.DataFrame.from_dict(al.test_metrics).drop("Labels", errors="ignore").T
test_metrics = test_metrics.stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Metric", 0: "Value"})

test_metrics["Value"] = test_metrics["Value"].apply(pd.to_numeric)
test_metrics["Set"] = "test"

joined_metrics = pd.concat([dev_metrics, test_metrics])

# +
colors = ["#368f8b", "#ec7357"]

sns.set(style="whitegrid")
ax = sns.relplot(data=joined_metrics, x="Number of labeled points", y="Value", col="Metric",
                 kind="line", hue="Set",legend=False, palette=sns.color_palette(colors))

show_handles = [ax.axes[0][0].lines[0], ax.axes[0][0].lines[1]]
show_labels = ["dev", "test"]
ax.axes[0][4].legend(handles=show_handles, labels=show_labels, loc="lower right")

ax.set_ylabels("")
ax.set_titles("{col_name}")
plt.show()
# -








