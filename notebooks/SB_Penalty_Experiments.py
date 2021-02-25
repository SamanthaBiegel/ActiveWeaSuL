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
from scipy.stats import entropy
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath("../activeweasul"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from plot import plot_probs, plot_train_loss
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment, bucket_entropy_experiment, add_baseline, synthetic_al_experiment, add_optimal
# -

# ### Generate data

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])

p_z = 0.5

set_seed(932)
data = SyntheticDataGenerator(N, p_z, centroids)
df_train = data.sample_dataset().create_df()
y_train = df_train.y.values

# +
N = 3000

set_seed(466)
data = SyntheticDataGenerator(N, p_z, centroids)
df_test = data.sample_dataset().create_df()
y_test = df_test.y.values
# -

# ### Apply labeling functions

# +
df_train.loc[:, "wl1"] = (df_train.x2<0.4)*1
df_train.loc[:, "wl2"] = (df_train.x1<-0.3)*1
df_train.loc[:, "wl3"] = (df_train.x1<-1)*1

label_matrix = df_train[["wl1", "wl2", "wl3"]].values

# +
df_test.loc[:, "wl1"] = (df_test.x2<0.4)*1
df_test.loc[:, "wl2"] = (df_test.x1<-0.3)*1
df_test.loc[:, "wl3"] = (df_test.x1<-1)*1

label_matrix_test = df_test[["wl1", "wl2", "wl3"]].values
# -

# ### Fit label model

# +
final_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-1,
                          n_epochs=100)

class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]

# +
set_seed(243)

lm = LabelModel(n_epochs=200,
                lr=1e-1)

Y_probs = lm.fit(label_matrix=label_matrix,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# lm.active_learning = True
# lm.penalty_strength = 1e4

# # Fit and predict on train set
# Y_probs = lm.fit(label_matrix=label_matrix,
#                  cliques=cliques,
#                  class_balance=class_balance,
#                  ground_truth_labels=y_train).predict()

# Predict on test set
Y_probs_test = lm.predict(label_matrix_test, lm.mu, p_z)

# Analyze test set performance
lm.analyze(y_test, Y_probs_test)
# -

plt.plot(lm.losses)
plt.show()

# ### Fit discriminative model

# 32: 0.022041911
# 36: 2.9367499e-08
# 35: 1.3665967e-06
# 78: 0.00035945282

# +
it = 0
# Choose strategy from ["maxkl", "margin", "nashaat"]
query_strategy = "maxkl"

seed = 76

al = ActiveWeaSuLPipeline(it=it,
                          final_model = LogisticRegression(**final_model_kwargs),
                          n_epochs=200,
                          query_strategy=query_strategy,
                          discr_model_frequency=1,
                          penalty_strength=1,
                          batch_size=256,
                          randomness=0,
                          seed=seed,
                          starting_seed=78)

Y_probs_al = al.run_active_weasul(label_matrix=label_matrix,
                                  y_train=y_train,
                                  label_matrix_test=label_matrix_test,
                                  y_test=y_test,
                                  cliques=cliques,
                                  class_balance=class_balance,
                                  train_dataset=CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach()),
                                  test_dataset=CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach()))
# -

batch_size=256

starting_seed = 36
penalty_strength = 1
nr_trials = 6
al_it = 15

penalty_strengths = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6]

# #### Active WeaSuL

exp_kwargs = dict(nr_trials=nr_trials,
                  al_it=al_it,
                  label_matrix=label_matrix,
                  y_train=y_train,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=starting_seed,
                  penalty_strength=penalty_strength,
                  batch_size=batch_size,
                  final_model=LogisticRegression(**final_model_kwargs, early_stopping=True, warm_start=False),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach()),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach()),
                  label_matrix_test=label_matrix_test,
                  y_test=y_test)

# +
penalty_dict_36 = {}
exp_kwargs["starting_seed"] = 36

for i, penalty in enumerate(penalty_strengths):
    exp_kwargs["penalty_strength"] = penalty
    np.random.seed(284)
    exp_kwargs["seeds"]= np.random.randint(0,1000,10)
    penalty_dict_36[i], _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")
# -



# +
penalty_dict_32 = {}
exp_kwargs["starting_seed"] = 32

for i, penalty in enumerate(penalty_strengths):
    exp_kwargs["penalty_strength"] = penalty
    np.random.seed(284)
    exp_kwargs["seeds"]= np.random.randint(0,1000,10)
    penalty_dict_32[i], _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

# +
penalty_dict_35 = {}
exp_kwargs["starting_seed"] = 35

for i, penalty in enumerate(penalty_strengths):
    exp_kwargs["penalty_strength"] = penalty
    np.random.seed(284)
    exp_kwargs["seeds"]= np.random.randint(0,1000,10)
    penalty_dict_35[i], _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

# +
penalty_dict_78 = {}
exp_kwargs["starting_seed"] = 78

for i, penalty in enumerate(penalty_strengths):
    exp_kwargs["penalty_strength"] = penalty
    np.random.seed(284)
    exp_kwargs["seeds"]= np.random.randint(0,1000,10)
    penalty_dict_78[i], _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")
# -



