# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath("../activelearning"))
from data import sample_clusters
from final_model import fit_predict_fm
from label_model import get_properties, fit_predict_lm, get_overall_accuracy
from pipeline import AL_pipeline
from plot import plot_probs, plot_accuracies
# -


# # Create clusters

# +
N_1 = 5000
N_2 = 5000
centroid_1 = np.array([0.1, 1.3])
centroid_2 = np.array([-0.8, -0.5])

X, p, y = sample_clusters(N_1, N_2, centroid_1, centroid_2, std = 0.5, scaling_factor = 4)
# -

df = pd.DataFrame({'x1': X[:,0], 'x2': X[:,1], 'y': y})

plot_probs(X, p, soft_labels=False)

# # Create weak labels

df.loc[:, "wl1"] = (X[:,1]<0.4)*1
df.loc[:, "wl2"] = (X[:,0]<-0.3)*1
df.loc[:, "wl3"] = (X[:,0]<-1)*1

print("Accuracy wl1:", (df["y"] == df["wl1"]).sum()/len(y))
print("Accuracy wl2:", (df["y"] == df["wl2"]).sum()/len(y))
print("Accuracy wl3:", (df["y"] == df["wl3"]).sum()/len(y))

label_matrix = np.array(df[["wl1", "wl2", "wl3", "y"]])

# # Compare approaches

N_total, nr_wl, y_set, y_dim = get_properties(label_matrix)

# +
label_model_kwargs = dict(n_epochs=200,
                        cliques=[[1,2]],
                        class_balance=[0.5,0.5],
                        lr=1e-1)

al_kwargs = dict(it=10)

final_model_kwargs = dict(input_dim=2,
                      output_dim=2,
                      lr=1e-3,
                      batch_size=256,
                      n_epochs=200)
# -

accuracies = {}
n_runs = 5

# +
# label model without active learning

accuracies["no_LM"] = []
accuracies["no_final"] = []

for i in range(n_runs):
    L = label_matrix[:,:-1]
    
    _, Y_probs = fit_predict_lm(L, label_model_kwargs, al=False)
    _, probs = fit_predict_fm(df[["x1", "x2"]].values, Y_probs, **final_model_kwargs, soft_labels=True)

    accuracies["no_LM"].append(get_overall_accuracy(Y_probs, df["y"]))
    accuracies["no_final"].append(get_overall_accuracy(probs, df["y"]))

# +
# label model with active learning

wl_al = np.full_like(df["y"], -1)

accuracies["yes_LM"] = []
accuracies["yes_final"] = []

for i in range(n_runs):
    L = np.concatenate([label_matrix[:,:-1], wl_al.reshape(len(wl_al),1)], axis=1)
    
    Y_probs_al, _, queried = AL_pipeline(L, df["y"], label_model_kwargs, **al_kwargs)
    _, probs_al = fit_predict_fm(df[["x1", "x2"]].values, Y_probs_al, **final_model_kwargs, soft_labels=True)

    accuracies["yes_LM"].append(get_overall_accuracy(Y_probs_al, df["y"]))
    accuracies["yes_final"].append(get_overall_accuracy(probs_al, df["y"]))
# -

accuracies

# +
mean_accuracies = {keys: np.array(values).mean() for keys, values in accuracies.items()}

df_accuracies = pd.DataFrame.from_dict(mean_accuracies, orient="index", columns=["Mean accuracy"])
df_accuracies["Active learning"], df_accuracies["Labels"] = df_accuracies.index.str.split('_').str
df_accuracies.set_index(["Labels", "Active learning"]).sort_values(["Active learning"])
pd.pivot_table(df_accuracies, columns="Labels", index="Active learning")

# +
# final model ground truth labels
_, probs_up = fit_predict_fm(df[["x1", "x2"]].values, df["y"].values, **final_model_kwargs, soft_labels=False)

get_overall_accuracy(probs_up, df["y"])

# +
# final model without active learning
# _, probs = fit_predict_final(df[["x1", "x2"]].values, Y_probs, **final_model_kwargs, soft_labels=True)

# get_overall_accuracy(probs, df["y"])

# +
# final model with active learning
# _, probs_al = fit_predict_final(df[["x1", "x2"]].values, Y_probs_al, **final_model_kwargs, soft_labels=True)

# get_overall_accuracy(probs_al, df["y"])
# -

# Probabilistic labels without active learning
plot_probs(X, Y_probs)

# Probabilistic labels with active learning
plot_probs(X, Y_probs_al)

plot_accuracies(accuracies["yes_final"])

# # Train model on queried data points

queried = np.array(queried)

# +
_, probs_q = fit_predict_fm(df[["x1", "x2"]].values, df["y"].values, **final_model_kwargs, soft_labels=False, subset=queried)

get_overall_accuracy(probs_q, df["y"])
# -

plot_probs(X, probs_q, soft_labels=True, subset=queried)

# # Train model on random subset

random_idx = np.random.choice(range(N_total), al_kwargs["it"])

# +
_, probs_r = fit_predict_fm(df[["x1", "x2"]].values, df["y"].values, **final_model_kwargs, soft_labels=False, subset=random_idx)

get_overall_accuracy(probs_r, df["y"])
# -

plot_probs(X, probs_r, soft_labels=True, subset=random_idx)


