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
from scipy.spatial.distance import pdist, squareform
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

N = 1000
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
cliques=[[0],[1,2]]

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "update_params",
             'final_model_kwargs': final_model_kwargs,
             'df': df,
             'n_epochs': 200
            }

# +
it = 200
query_strategy = "margin"
alpha = 0.03

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            alpha=alpha,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
print("Accuracy:", al._accuracy(Y_probs_al, data.y))
# -

plot_probs(df, al.predict_true().detach().numpy())

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.accuracy()

al.z.grad

al.plot_iterations()

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

# +
L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            alpha=alpha,
                            query_strategy=query_strategy,
                            randomness=0.5)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
print("Accuracy:", al._accuracy(Y_probs_al, data.y))
# -

al.plot_parameters()

# # Margin + information density tradeoff

# +
L = label_matrix[:, :-1]
cliques=[[0],[1,2]]

lm = LabelModel(final_model_kwargs=final_model_kwargs,
                df=df,
                active_learning=False,
                add_cliques=True,
                add_prob_loss=False,
                n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
lm.accuracy()
# -

# $m(x_i)^{\beta} * d(x_i)^{1-\beta}$

plot_probs(df, Y_probs.detach().numpy())

# +
measures = pd.DataFrame()
abs_diff = 1 / torch.abs(Y_probs[:, 1] - Y_probs[:, 0])
I = torch.Tensor(1 / lm.N * squareform(1 / pdist(data.X, metric="euclidean")).sum(axis=1))

# Normalize
# abs_diff = abs_diff / abs_diff.max()
# I = I / I.max()

for beta in list(np.round(np.linspace(0,1,11), decimals=1)):
    measures[beta] = ((abs_diff)**beta * I**(1 - beta)).detach().numpy()
# -

measures = measures.stack().reset_index(1)

measures_df = df.merge(measures, left_index=True, right_index=True).rename(columns={"level_1": "beta", 0: "measure"})

fig = px.scatter(measures_df, x="x1", y="x2", color="measure", animation_frame="beta", color_continuous_scale=px.colors.diverging.Geyser)
fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      width=700, height=700, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
fig.show()

fig = px.scatter(measures_df, x="x1", y="x2", color="measure", animation_frame="beta", color_continuous_scale=px.colors.diverging.Geyser)
fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      width=700, height=700, xaxis_title="x1", yaxis_title="x2", template="plotly_white")
fig.show()

# +
it = 200
query_strategy = "margin_density"
beta = 0.2

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            beta=beta,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
print("Accuracy:", al._accuracy(Y_probs_al, data.y))
# -

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.accuracy()

# Margin + information density
al.plot_parameters()

plot_probs(df, al.prob_dict[5], add_labeled_points=al.queried[:50], soft_labels=False)






