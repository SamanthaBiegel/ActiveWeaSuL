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

# +
it = 50
active_learning = "probs"
query_strategy = "margin"
add_cliques=True
add_prob_loss=False

cliques=[[0],[1,2]]
L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            final_model_kwargs=final_model_kwargs,
                            df=df,
                            n_epochs=200,
                            active_learning=active_learning,
                            query_strategy=query_strategy,
                            add_cliques=add_cliques,
                            add_prob_loss=add_prob_loss,
                            randomness = 0.1)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.accuracy()
# -

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.accuracy()

# +
df["label"] = Y_probs_al.detach().numpy()[:,1]

fig = go.Figure(go.Scatter(x=df["x1"], y=df["x2"], mode="markers", hovertext=df["label"],hoverinfo="text", marker=dict(color=df["label"], colorscale=px.colors.diverging.Geyser, colorbar=dict(title="Labels"),cmid=0.5), showlegend=False))

fig.add_trace(go.Scatter(x=data.X[al.queried,0], y=data.X[al.queried,1], mode="markers", marker=dict(color='Black', size=5), showlegend=False))

fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                      width=700, height=700, xaxis_title="x1", yaxis_title="x2", template="plotly_white")

fig.show()
# -

plot_probs(df, probs_final_al.detach().numpy(), soft_labels=True, subset=None)

probs_df = pd.DataFrame.from_dict(al.prob_dict)
probs_df = probs_df.stack().reset_index().rename(columns={"level_0": "x", "level_1": "iteration", 0: "prob_y"})
probs_df = probs_df.merge(df, left_on = "x", right_index=True)

# +
fig = px.scatter(probs_df, x="x1", y="x2", color="prob_y", animation_frame="iteration", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0,-1]], color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)
fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                  width=1000, height=1000, xaxis_title="x1", yaxis_title="x2", template="plotly_white")

# fig.show()

# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])

# app.run_server(debug=True, use_reloader=False)
# -

unique_probs_df = pd.DataFrame.from_dict(al.unique_prob_dict)
unique_probs_df = unique_probs_df.stack().reset_index().rename(columns={"level_0": "Configuration", "level_1": "Iteration", 0: "P_Y_1"})

fig = px.line(unique_probs_df, x="Iteration", y="P_Y_1", color="Configuration")
fig.show()

mu_df = pd.DataFrame.from_dict(al.mu_dict)
mu_df = mu_df.iloc[[0,1,6,7,8,9],:].stack().reset_index().rename(columns={"level_0": "Parameter", "level_1": "Iteration", 0: "Probability"})

fig = px.line(mu_df, x="Iteration", y="Probability", color="Parameter")
fig.show()


