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

# +
N = 10000


# centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
# centroids = {0: np.array([[1.1, 2.3], [-2, 1.5], [3, 3]]), 1: np.array([[-0.8, -0.5], [1.5, -0.5], [-3, -1.5]])}
# centroids = {0: np.array([[0, 0], [4, 0], [2, -2]]), 1: np.array([[2, 2], [-2, 2], [0, 4]])}

centroids = {0: np.array([[-1, 0], [0, -1]]), 1: np.array([[0,1], [1, 0]])}

p_z = 0.5

# p_z_ = {0: np.array([0.5, 0.25, 0.25]), 1: np.array([0.5, 0.25, 0.25])}

p_z_ = {0: np.array([0.25, 0.75]), 1: np.array([0.75, 0.25])}

# p_z_ = {0: np.array([1/3, 1/3, 1/3]), 1: np.array([1/3, 1/3, 1/3])}
# -

data = SyntheticData(N, p_z, np.array([[0.1, 1.3], [-0.8, -0.5]]))
data.sample_data_set()

cluster = np.zeros((N))

cluster[data.y == 0] = np.random.choice(a=np.arange(len(p_z_[0])), size=(len(data.y[data.y == 0])), p=p_z_[0])
cluster[data.y == 1] = np.random.choice(a=np.arange(len(p_z_[1])), size=(len(data.y[data.y == 1])), p=p_z_[1])

# +
X = np.zeros((N, 2))

for i in range(0, 2):
    for j in range(len(p_z_[i])):
#         scale = np.random.uniform(0.2, 0.5)
        scale = [[0.2, 0.3], [0.3, 0.2]][i][j]
        X[np.logical_and((data.y == i), (cluster == j)), :] = np.random.normal(loc=centroids[i][j, :], scale=np.array([scale, scale]), size=(np.logical_and((data.y == i), (cluster == j)).sum(), 2))
        
# -

df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': data.y})

plot_probs(df, probs=data.y, soft_labels=False)

# +
# df.loc[:, "wl1"] = (df["x2"]<0.4)*1
# df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
# df.loc[:, "wl3"] = (df["x1"]<-1)*1

df.loc[:, "wl1"] = (df["x2"]>0.5)*1
df.loc[:, "wl2"] = (df["x1"]>0.5)*1
# df.loc[:, "wl3"] = (df["x1"]>-0.5)*1
# df.loc[:, "wl4"] = (df["x2"]>-0.5)*1

# df.loc[:, "wl1"] = (df["x2"]>3)*1
# df.loc[:, "wl2"] = (df["x1"]<-1)*1
# df.loc[:, "wl3"] = (df["x1"]<1)*1

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
df.loc[:, "wl3"] = [random_LF(y, fp=0.2, fn=0.3, abstain=0) for y in df["y"]]
# -

label_matrix = np.array(df[["wl1", "wl2", "wl3", "y"]])

_, inv_idx = np.unique(label_matrix[:, :-1], axis=0, return_inverse=True)

plot_probs(df, probs=inv_idx, soft_labels=False)

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 250}

class_balance = np.array([0.5,0.5])
# cliques=[[0],[1,2]]
cliques=[[0],[1],[2]]

al_kwargs = {'add_prob_loss': False,
             'add_cliques': False,
             'active_learning': "probs",
             'final_model_kwargs': final_model_kwargs,
             'df': df,
             'n_epochs': 200
            }

# +
it = 20
query_strategy = "margin"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=1)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
print("Accuracy:", al._accuracy(Y_probs_al, data.y))

# +
L = label_matrix[:, :-1]

lm = LabelModel(final_model_kwargs=final_model_kwargs,
                df=df,
                active_learning=False,
                add_cliques=False,
                add_prob_loss=False,
                n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
lm.accuracy()
# -

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm.fit(features=X, labels=Y_probs_al.detach().numpy()).predict()
fm.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=X, labels=Y_probs.detach().numpy()).predict()
fm.accuracy()

lm.mu

lm.get_true_mu()

psi_y, wl_idx_y = lm._get_psi(label_matrix, [[0],[1],[2],[3]], 4)


def color(df_opt):
    c1 = 'background-color: red'
    c2 = 'background-color: green'
    df1 = pd.DataFrame(c1, index=df_opt.index, columns=df_opt.columns)
    idx = np.where(lm.mask)
    for i in range(len(idx[0])):
        df1.loc[(idx[0][i], idx[1][i])] = c2
    return df1


pd.DataFrame(np.linalg.pinv(np.cov(psi_y.T))).style.apply(color, axis=None)

pd.DataFrame(np.linalg.pinv(np.cov(lm.psi.T))).style.apply(color, axis=None)

pd.DataFrame(np.linalg.pinv(np.cov(L_y.T))).style.apply(color, axis=None)

pd.DataFrame(np.linalg.pinv(np.cov(L_y.T))).style.apply(color, axis=None)

plot_probs(df, lm.predict_true().detach().numpy())

plot_probs(df, Y_probs.detach().numpy())

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, Y_probs_al.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, probs_final.detach().numpy())

plot_probs(df, probs_final_al.detach().numpy())







