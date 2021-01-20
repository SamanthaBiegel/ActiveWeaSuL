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
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm

sys.path.append(os.path.abspath("../activelearning"))
from synthetic_data import SyntheticDataGenerator, SyntheticDataset
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline
from plot import plot_probs, plot_train_loss
# -

# ## Load data

df = pd.read_csv("../data/synthetic_dataset_3.csv")
L = np.array(df[["wl1", "wl2", "wl3"]])

# ## Initial label model fit

# +
class_balance = np.array([0.5, 0.5])

cliques = [[0],[1,2]]

# +
lm = LabelModel(n_epochs=200,
                lr=1e-1)

# Fit and predict on train set
Y_probs = lm.fit(label_matrix=L,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# Predict on test set
Y_probs_test = lm.predict(L, lm.mu, 0.5)

# Analyze test set performance
lm.analyze(df.y.values)
# -

final_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-3,
                          n_epochs=100)
dm = LogisticRegression(**final_model_kwargs)

train_dataset = SyntheticDataset(df=df, Y=Y_probs.detach())

# +
# train_set = torch.Tensor(df[["x1", "x2"]].values)
# train_tensor_set = torch.utils.data.TensorDataset(train_set, Y_probs.detach())
train_set.Y = Y_probs.detach()
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=256, shuffle=True)

train_preds = dm.fit(train_loader).predict()
dm.analyze(df.y.values, train_preds)

# +
train_set.Y = Y_probs_al.detach()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=256, shuffle=True)

train_preds_al = dm.fit(train_loader).predict()
# -

dm.analyze(df.y.values)

# ## Active learning pipeline

final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'n_epochs': 100}

# +
it = 10
# Choose strategy from ["maxkl", "margin", "nashaat"]
query_strategy = "maxkl"

al = ActiveWeaSuLPipeline(it = it,
                          final_model = LogisticRegression(**final_model_kwargs),
                          discr_model_frequency = 10,
                          batch_size = 256,
                          n_epochs = 200,
                          query_strategy=query_strategy)

Y_probs_al = al.run_active_weasul(label_matrix=L,
                                  y_train = df.y.values,
                                  train_dataset = train_dataset,
                                  cliques=cliques,
                                  class_balance=class_balance)
# -

train_dataset.Y = Y_probs_al.clone().detach()
dl_train = DataLoader(train_dataset, shuffle=True, batch_size=256)
dl_train_predict = DataLoader(train_dataset, shuffle=False, batch_size=256)

preds = al.final_model.fit(dl_train).predict()

al.final_model.analyze(df.y.values, preds)

# ## Analyze results

metric_df = process_metric_dict(al.metrics, query_strategy)

plot_metrics(metric_df)










