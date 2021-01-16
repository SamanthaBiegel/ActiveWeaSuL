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
from experiments import process_metric_dict, plot_metrics
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveWeaSuLPipeline
# -

# ## Load data

df = pd.read_csv("../data/synthetic_dataset_3.csv")
L = np.array(df[["wl1", "wl2", "wl3"]])

# ## Initial label model fit

# +
class_balance = np.array([0.5, 0.5])

cliques = [[0],[1,2]]

# +
lm = LabelModel(y_true=df.y.values,
                n_epochs=200,
                lr=1e-1)

# Fit and predict on train set
Y_probs = lm.fit(label_matrix=L,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# Predict on test set
Y_probs_test = lm._predict(L, lm.mu, 0.5)

# Analyze test set performance
lm._analyze(Y_probs_test, df.y.values)
# -

# ## Active learning pipeline

final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 100}

# +
it = 10
# Choose strategy from ["maxkl", "margin", "nashaat"]
query_strategy = "maxkl"

al = ActiveWeaSuLPipeline(it=it,
                          y_true=df.y.values,
                          final_model = DiscriminativeModel(df, **final_model_kwargs),
                          df=df,
                          n_epochs=200,
                          query_strategy=query_strategy)

Y_probs_al = al.run_active_weasul(label_matrix=L,
                                  cliques=cliques,
                                  class_balance=class_balance,
                                  label_matrix_test=L,
                                  y_test=df.y.values)
# -

# ## Analyze results

metric_df = process_metric_dict(al.metrics)

plot_metrics(metric_df)












