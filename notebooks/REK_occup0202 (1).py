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
#     display_name: Python 3
#     language: python
#     name: python3
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

sys.path.append(os.path.abspath("../activeweasul"))
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, CustomTensorDataset, set_seed
from plot import plot_probs, plot_train_loss
# -

# ## Load data

path_prefix = "../data/data/datasets/"


L_train = np.load("../data/data/L_train_300121.npy")
L1 = np.load("../data/data/L1_300121.npy")
L2 = np.load("../data/data/L2_300121.npy")
df_occup = pd.read_csv("../data/data/df_occup_300121.csv")
df1 = pd.read_csv("../data/data/df1_300121.csv")
df2 = pd.read_csv("../data/data/df2_300121.csv")

y_train = df_occup["Occupancy"]
y1 = df1["Occupancy"]
y2 = df2["Occupancy"]

# +
# L_train = pickle.load(open("../data/data/L_train_300121.pkl", "rb"))
# L1 = pickle.load(open(path_prefix + "../datasets/occupancy_data/L1.pkl", "rb"))
# L2 = pickle.load(open(path_prefix + "../datasets/occupancy_data/L2.pkl", "rb"))

# y_train = pickle.load(open(path_prefix + "../datasets/occupancy_data/y_train.pkl", "rb"))
# y1 = pickle.load(open(path_prefix + "../datasets/occupancy_data/y1.pkl", "rb"))
# y2 = pickle.load(open(path_prefix + "../datasets/occupancy_data/y2.pkl", "rb"))
# -

L_train.shape, L1.shape, L2.shape

# ## Initial label model fit

# Remove rows with all abstains
L_train.shape, L_train[~(L_train.sum(axis=1) == - L_train.shape[1])].shape

L = L_train[~(L_train.sum(axis=1) == - L_train.shape[1])]
y = y_train[~(L_train.sum(axis=1) == - L_train.shape[1])]

L.shape, y.shape

np.sum(y)/len(y)

L = L[:,:2]

# +
class_balance_1 = 0.21 # np.sum(y)/len(y)
class_balance = np.array([1 - class_balance_1, class_balance_1])

cliques = [[0], [1]]#, [2], [3], [4], [5]]#, [6], [7], [8]]
# -

L != -1

_, lambda_abstain_indicator_inverse, lambda_abstain_indicator_counts = np.unique(L != -1, axis=0, return_inverse=True, return_counts=True)

lambda_combs, lambda_index, lambda_inverse, lambda_counts = np.unique(L, axis=0, return_counts=True, return_inverse=True, return_index=True)


ma = lambda_counts.copy()
rows_not_abstain, cols_not_abstain = np.where(lambda_combs != -1)
for i, comb in enumerate(lambda_combs):
    nr_non_abstain = (comb != -1).sum()
    if nr_non_abstain < 2:
        if nr_non_abstain == 0:
            new_counts[i] = 0
        else:
            match_rows = np.where((lambda_combs[:, cols_not_abstain[rows_not_abstain == i]] == lambda_combs[i, cols_not_abstain[rows_not_abstain == i]]).all(axis=1))       
            new_counts[i] = lambda_counts[match_rows].sum()
            lambda_counts[np.where((lambda_combs[:, cols_not_abstain[rows_not_abstain == i]] != -1).all(axis=1))].sum()


P_lambda = torch.Tensor((new_counts)[lambda_inverse]/(3694 + 94 + 2339 + 83 + 1617))[:, None]

lambda_abstain_indicator_counts

lambda_counts

lambda_combs

(1617+164)/((lambda_counts[3:]).sum())

(83+138+2339)/((lambda_counts[3:]).sum())

(94 + 83 + 1617)/(3694 + 94 + 2339 + 83 + 1617)

P_lambda = torch.Tensor(lambda_counts[lambda_inverse]/lambda_abstain_indicator_counts[lambda_abstain_indicator_inverse])[:, None]

lm.P_lambda[lambda_index]

_, lambda_inverse, lambda_counts = np.unique(L, axis=0, return_inverse=True, return_counts=True)

# +
set_seed(243)
lm = LabelModel(n_epochs=200,
                lr=1e-1)

# Fit and predict on train set
Y_probs = lm.fit(label_matrix=L,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# Predict on test set
# Y_probs_test = lm.predict(L1, lm.mu, class_balance_1)

# Analyze test set performance
# lm.analyze(y)
# -

lm.P_lambda[lambda_index]

np.unique(lm.P_lambda)

np.unique(Y_probs.detach().numpy(), axis=0)



np.where(L != -1)

final_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-3,
                          n_epochs=100)
dm = LogisticRegression(**final_model_kwargs)

df = pd.DataFrame(L)
df["y"] = y
df.sample()

# ## Active learning pipeline

al_kwargs = {
     'n_epochs': 200
}

features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
train_dataset = CustomTensorDataset(
    torch.FloatTensor(df_occup[features].values), 
    torch.FloatTensor(df_occup["Occupancy"].values))
test_dataset = CustomTensorDataset(
    torch.FloatTensor(df1[features].values), 
    torch.FloatTensor(df1["Occupancy"].values))

it = 20
query_strategy = "maxkl" # Choose strategy from ["maxkl", "margin", "nashaat"]
final_model_kwargs = {'input_dim': 5, 'output_dim': 2, 'lr': 0.01, 'n_epochs': 100}
al = ActiveWeaSuLPipeline(
    it=it,
    **al_kwargs,
    penalty_strength=1e4,
    query_strategy=query_strategy,
    seed=11,
    starting_seed=120, 
    discr_model_frequency=6, 
    final_model=LogisticRegression(**final_model_kwargs)
)
Y_probs_al = al.run_active_weasul(
    label_matrix=L,
    cliques=cliques,
    class_balance=class_balance,
    label_matrix_test=L1,
    y_test=y1,
    y_train=y, 
    train_dataset=train_dataset, 
    test_dataset=test_dataset
)   

plot_metrics(process_metric_dict(al.metrics, "test"), filter_metrics=["F1"])

pd.DataFrame.from_dict(al.probs["bucket_labels_train"]).stack().reset_index()



plot_metrics(process_metric_dict(al.metrics, "test"), filter_metrics=["F1"])

al.probs["Discriminative_train"][18][:30]

y_train[:30]

al.label_model.analyze(y, al.label_model.predict_true(y))

metric_df = process_metric_dict(al.metrics, strategy_string=query_strategy)

plot_metrics(metric_df)

plot_metrics(metric_df, filter_metrics=["Accuracy"])

plot_metrics(metric_df, filter_metrics=["MCC"])

plot_metrics(metric_df, filter_metrics=["Precision"])

plot_metrics(metric_df, filter_metrics=["Recall"])

plot_metrics(metric_df, filter_metrics=["F1"])

plot_metrics(metric_df, filter_metrics=["F1", "Accuracy", "Precision", "Recall", "MCC"]) 


