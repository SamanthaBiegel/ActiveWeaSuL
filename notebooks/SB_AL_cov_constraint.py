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
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
import dash
import dash_html_components as html
import dash_core_components as dcc

sys.path.append(os.path.abspath("../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_accuracies
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -

pd.options.display.expand_frame_repr = False 
np.set_printoptions(suppress=True, precision=16)

# # Create data

N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

data = SyntheticData(N, p_z, centroids)
df = data.sample_data_set().create_df()

df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1

label_matrix = np.array(df[["wl1", "wl2", "wl3", "y"]])

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 250}

class_balance = np.array([0.5,0.5])
cliques=[[0],[1,2],[3]]

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "cov",
             'final_model_kwargs': final_model_kwargs,
             'df': df
            }

# +
it = 1000
query_strategy = "margin"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            n_epochs=200,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
print("Accuracy:", al._accuracy(Y_probs_al, data.y))
# -

print(al.wl_idx)

al.plot_iterations()

al.plot_parameters()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=False)
probs_final_true = fm.fit(features=data.X, labels=data.y).predict()
fm.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=np.concatenate([(1 - al.prob_dict[100])[:, None], (al.prob_dict[100])[:, None]], axis=1)).predict()
fm.accuracy()

plot_probs(df, probs_final.detach().numpy(), soft_labels=True, subset=None)

al.mu

al.get_true_mu()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=np.concatenate([(1 - al.prob_dict[100])[:, None], (al.prob_dict[100])[:, None]], axis=1)).predict()
fm.accuracy()

plot_probs(df, probs_final.detach().numpy(), soft_labels=True, subset=None)





al.plot_iterations()

al.plot_iterations()

al.plot_parameters()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=np.concatenate([(1 - al.prob_dict[1000])[:, None], (al.prob_dict[1000])[:, None]], axis=1)).predict()
fm.accuracy()

plot_probs(df, probs_final.detach().numpy(), soft_labels=True, subset=None)



al.plot_parameters()

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried[:100])

conf_list = np.vectorize(al.confs.get)(al.unique_inverse[al.queried])

# +
input_df = pd.DataFrame.from_dict(al.unique_prob_dict)
input_df = input_df.stack().reset_index().rename(columns={"level_0": "WL Configuration", "level_1": "Active Learning Iteration", 0: "P(Y = 1|...)"})

input_df["WL Configuration"] = input_df["WL Configuration"].map(al.confs)
# -

fig = al.plot_probabilistic_labels()

for i, conf in enumerate(np.unique(conf_list)):
    x = np.array(range(it))[conf_list == conf]
    y = input_df[(input_df["WL Configuration"] == conf)].set_index("Active Learning Iteration").iloc[x]["P(Y = 1|...)"]

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=5, color="black"), showlegend=False))

fig.show()




