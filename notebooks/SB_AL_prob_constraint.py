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
from plot import plot_probs
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

plot_probs(df, probs=data.y, soft_labels=False)

df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1


# +
def random_LF(y, fp, fn, abstain):
    ab = np.random.uniform()
    
    if ab < abstain:
        return -1
    
    threshold = np.random.uniform()
    
    if y == 1 and threshold < fn:
        return 0
        
    elif y == 0 and threshold < fp:
        return 1
        
    
    
    return y

# df.loc[:, "wl1"] = [random_LF(y, fp=0.1, fn=0.2, abstain=0) for y in df["y"]]
# df.loc[:, "wl2"] = [random_LF(y, fp=0.05, fn=0.4, abstain=0) for y in df["y"]]
# df.loc[:, "wl3"] = [random_LF(y, fp=0.6, fn=0.8, abstain=0) for y in df["y"]]
# -

label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])

_, inv_idx = np.unique(label_matrix[:, :-1], axis=0, return_inverse=True)

plot_probs(df, probs=inv_idx, soft_labels=False)

# +
# psi_y, wl_idx_y = lm._get_psi(label_matrix, [[0],[1],[2],[3]], 4)

# +
# pd.DataFrame(np.linalg.pinv(np.cov(psi_y.T))).style.apply(color, axis=None)

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 250}

class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]
# cliques=[[0],[1],[2]]

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'df': df,
             'n_epochs': 200
            }
# -

# # Margin strategy

# +
L = label_matrix[:, :-1]

lm = LabelModel(df=df,
                active_learning=False,
                add_cliques=True,
                add_prob_loss=False,
                n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
lm.analyze()
lm.print_metrics()

# +
it = 1
query_strategy = "test"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
#                             final_model = DiscriminativeModel(df, **final_model_kwargs),
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.label_model.print_metrics()

# +
it = 10
query_strategy = "margin"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            final_model = DiscriminativeModel(df, **final_model_kwargs),
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.label_model.print_metrics()
# -

al.plot_animation()

al.plot_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=lm.predict_true().detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=Y_probs.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

al.mu

lm.mu

lm.get_true_mu()

lm.P_lambda

lm.nr_wl

plot_probs(df, lm.predict_true().detach().numpy())

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, Y_probs.detach().numpy())

plot_probs(df, probs_final_al.detach().numpy())

plot_probs(df, probs_final_al.detach().numpy())

plot_probs(df, probs_final.detach().numpy())

al.plot_iterations()

al.plot_parameters()

al.mu_dict

al.mu_dict[1][6] - al.mu_dict[0][6]

al.mu_dict[1][7] - al.mu_dict[0][7] + (al.mu_dict[1][9] - al.mu_dict[0][9])

al.mu_dict[1][9] - al.mu_dict[0][9]

fig = al.plot_probabilistic_labels()

conf_list = np.vectorize(al.confs.get)(al.unique_inverse[al.queried])

# +
input_df = pd.DataFrame.from_dict(al.unique_prob_dict)
input_df = input_df.stack().reset_index().rename(columns={"level_0": "WL Configuration", "level_1": "Active Learning Iteration", 0: "P(Y = 1|...)"})

input_df["WL Configuration"] = input_df["WL Configuration"].map(al.confs)
# -

for i, conf in enumerate(np.unique(conf_list)):
    x = np.array(range(it))[conf_list == conf]
    y = input_df[(input_df["WL Configuration"] == conf)].set_index("Active Learning Iteration").iloc[x]["P(Y = 1|...)"]

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10, color="black"), showlegend=False, hoverinfo="none"))

fig.show()

al.plot_parameters()

plot_probs(df, probs_final_al.detach().numpy(), soft_labels=True, subset=None)

probs_df = pd.DataFrame.from_dict(al.final_prob_dict)
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

# # Margin + information density

# +
it = 20
query_strategy = "margin_density"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
print("Accuracy:", al._accuracy(Y_probs_al, data.y))
# -

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.accuracy()

al.plot_parameters()

al.plot_iterations()

al.plot_iterations()

al.plot_parameters()

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

prob_label_df = pd.DataFrame.from_dict(al.prob_dict)
prob_label_df = prob_label_df.stack().reset_index().rename(columns={"level_0": "x", "level_1": "iteration", 0: "prob_y"})
prob_label_df = prob_label_df.merge(df, left_on = "x", right_index=True)

# +
fig = px.scatter(prob_label_df, x="x1", y="x2", color="prob_y", animation_frame="iteration", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0,-1]], color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)
fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                  width=1000, height=1000, xaxis_title="x1", yaxis_title="x2", template="plotly_white")

# fig2.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)
# -

# # Class balance

p_z = 0.01
centroids = np.array([[0.1, 1.3], [-1, -1]])
data = SyntheticData(N, p_z, centroids)
df = data.sample_data_set().create_df()

# +
df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1

label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])
# -

plot_probs(df, probs=data.y, soft_labels=False)

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 250}

class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]
# cliques=[[0],[1],[2]]

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'final_model_kwargs': final_model_kwargs,
             'df': df,
             'n_epochs': 200
            }

# +
L = label_matrix[:, :-1]

lm = LabelModel(final_model_kwargs=final_model_kwargs,
                df=df,
                active_learning=False,
                add_cliques=True,
                add_prob_loss=False,
                n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
lm.analyze()
lm.print_metrics()

# +
it = 10
query_strategy = "margin"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
#                             beta=1,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.analyze()
al.print_metrics()
# -

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=Y_probs.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm_al = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm_al.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm_al.analyze()
fm_al.print_metrics()

# +

df.iloc[al.queried]
# -



diff_prob_labels = al.prob_dict[1] - al.prob_dict[1-1]
df.iloc[np.where(al.unique_inverse == al.unique_inverse[np.argmax(diff_prob_labels)])[0]]

plot_probs(df, lm.predict_true().numpy())

plot_probs(df, Y_probs.detach().numpy())

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, probs_final.detach().numpy())

plot_probs(df, probs_final_al.detach().numpy())

al.plot_parameters()

al.plot_iterations()

al.get_true_mu().numpy()

al.mu


