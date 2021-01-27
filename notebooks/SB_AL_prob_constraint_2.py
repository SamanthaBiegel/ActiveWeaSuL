# -*- coding: utf-8 -*-
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
from synthetic_data import SyntheticDataGenerator, SyntheticDataset
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed
from plot import plot_probs, plot_train_loss
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment
# -

pd.options.display.expand_frame_repr = False 
np.set_printoptions(suppress=True, precision=16)

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

data = SyntheticDataGenerator(N, p_z, centroids)
df = data.sample_dataset().create_df()

df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1
# df.loc[:, "wl4"] = (df["x2"]<-0.5)*1
# df.loc[:, "wl5"] = (df["x1"]<0)*1

label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])
# -

df_big=df

df = pd.read_csv("../data/synthetic_small.csv")

df["u"] = "?"

# +
# df.to_csv("../data/synthetic_small.csv", index=False)
# -

df_big = pd.read_csv("../data/synthetic_dataset_3.csv")
label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])

_, inv_idx = np.unique(label_matrix[:, :-1], axis=0, return_inverse=True)

# +
class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]

L = df[["wl1", "wl2", "wl3"]].values

lm = LabelModel(n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
# lm.analyze()
# lm.print_metrics()
# -

np.unique(lm.predict_true().detach().numpy(), axis=0)

_, unique_idx, unique_inverse = np.unique(Y_probs.clone().detach().numpy()[:, 1], return_index=True, return_inverse=True)
confs = {range(len(unique_idx))[i]: "-".join([str(e) for e in row]) for i, row in enumerate(L[unique_idx, :])}
conf_list = np.vectorize(confs.get)(unique_inverse)

L = df[["wl1", "wl2", "wl3"]].values
psi,_ = lm._get_psi(L, cliques, 3)
true_test = lm.predict(L, psi, lm.get_true_mu()[:, 1][:, None], df["y"].mean())



sampled_idx = random.sample(list(df[unique_inverse == 1].index), 5)

df["u"] = 0.5

df.loc[sampled_idx, "u"] = df.loc[sampled_idx, "y"]

df.iloc[sampled_idx]

# +
# colors = ["#2b4162", "#ec7357", "#368f8b", "#e9c46a", "#721817", "#fa9f42", "#0b6e4f", "#96bdc6",  "#c09891", "#5d576b", "#c6dabf"]
# colors = ["#368F8B", "#3EA39E", "#43B1AC", "#F1937E", "#EF846C", "#EC7357"]
colors = ["#368F8B", "#43B1AC", "#5CC1BC", "#F5B2A3", "#F1937E", "#EC7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=6), rc={'figure.figsize':(15,15)})

fig, ax = plt.subplots()

scatter = plt.scatter(x=df.x1, y=df.x2, c=df.u, s=(700), edgecolor="black", cmap=cmap)
# plt.legend(labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
# g = sns.scatterplot(x=df.x1, y=df.x2, hue=Y_probs.detach().numpy()[:,1], s=(700), edgecolor="black",
#                     palette=sns.color_palette("mycolormap", n_colors=6))
plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
plt.clim(0,1)

handles, labels = scatter.legend_elements()

legend1 = ax.legend(handles=[handles[i] for i in [0,2]], labels=[0, 1],
                    loc="upper left", title="", fontsize="large")
ax.add_artist(legend1)


plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])
# plt.colorbar(sm)

# plt.savefig("plots/sample-examples.png")

# +
# colors = ["#2b4162", "#ec7357", "#368f8b", "#e9c46a", "#721817", "#fa9f42", "#0b6e4f", "#96bdc6",  "#c09891", "#5d576b", "#c6dabf"]
# colors = ["#368F8B", "#3EA39E", "#43B1AC", "#F1937E", "#EF846C", "#EC7357"]
colors = ["#368F8B", "#43B1AC", "#5CC1BC", "#F5B2A3", "#F1937E", "#EC7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=6), rc={'figure.figsize':(15,15)})

plt.scatter(x=df.x1, y=df.x2, c=Y_probs.detach().numpy()[:,1], s=(700), edgecolor="black", cmap=cmap)
labels=["0-0-0", "0-1-0", "1-0-0", "0-1-1", "1-1-0", "1-1-1"]
# plt.legend(labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
# g = sns.scatterplot(x=df.x1, y=df.x2, hue=Y_probs.detach().numpy()[:,1], s=(700), edgecolor="black",
#                     palette=sns.color_palette("mycolormap", n_colors=6))
plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
plt.clim(0,1)

# g.set(yticks=[], xticks=[])
# handles,labels = g.axes.get_legend_handles_labels()
# handles,labels = plt.axes.get_legend_handles_labels()


plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])
# cbar = plt.colorbar(sm)
# cbar.ax.set_title(r"P(Y=1|λ)", fontsize=30, y=1.02)
# cbar.ax.tick_params(labelsize=20)
# plt.savefig("plots/configurations_colorbar.png")

# plt.show()
# plt.savefig("plots/configurations.png")
# fig = g.get_figure()
# fig.savefig("plots/configurations.png")
# -

np.unique(Y_probs.detach().numpy(), axis=0)

# +
# colors = ["#2b4162", "#ec7357", "#368f8b", "#e9c46a", "#721817", "#fa9f42", "#0b6e4f", "#96bdc6",  "#c09891", "#5d576b", "#c6dabf"]
# colors = ["#368F8B", "#3EA39E", "#43B1AC", "#F1937E", "#EF846C", "#EC7357"]
colors = ["#368F8B", "#43B1AC", "#5CC1BC", "#F5B2A3", "#F1937E", "#EC7357"]

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=6), rc={'figure.figsize':(15,15)})

g = sns.scatterplot(x=df_big.x1, y=df_big.x2, hue=lm.predict_true().detach().numpy()[:,1], s=(700), edgecolor="black",
                    palette=sns.color_palette("mycolormap", n_colors=6))
plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
g.set(yticks=[], xticks=[])
handles,labels = g.axes.get_legend_handles_labels()
labels=["0-0-0", "0-1-0", "1-0-0", "0-1-1", "1-1-0", "1-1-1"]
plt.legend(handles=handles, labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)

plt.show()
fig = g.get_figure()
fig.savefig("plots/configurations.png")
# +
colors = ["#368f8b", "#ec7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=200), rc={'figure.figsize':(17,15)})
# sns.set(style="white", palette=sns.color_palette("Set2"), rc={'figure.figsize':(15,15)})

plt.scatter(x=df.x1, y=df.x2, c=probs_final.detach().numpy()[:,1], s=(700), edgecolor="black", cmap=cmap)
plt.clim(0,1)

plt.plot([-2.5, 1.7],[y1, y2], linewidth=5, color="black")
# p.set(yticks=[], xticks=[])
# handles,labels = g.axes.get_legend_handles_labels()
# plt.legend(handles=handles, labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
# plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])
cbar = plt.colorbar(sm)
cbar.ax.set_title(r"P(Y=1|X)", fontsize=30, y=1.02)
cbar.ax.tick_params(labelsize=20)
plt.savefig("plots/decboundary_colorbar.png")


# plt.savefig("plots/decboundary.png")

# +
colors = ["#368f8b", "#ec7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=200), rc={'figure.figsize':(17,15)})
# sns.set(style="white", palette=sns.color_palette("Set2"), rc={'figure.figsize':(15,15)})

plt.scatter(x=df.x1, y=df.x2, c=probs_final_al.detach().numpy()[:,1], s=(700), edgecolor="black", cmap=cmap)
plt.clim(0,1)

plt.plot([-2.5, 1.7],[y1_al, y2_al], linewidth=5, color="black")
# plt.plot([-0.263722, -0.577264],[0.320930, 0.572447], linewidth=5, color="black")
# p.set(yticks=[], xticks=[])
# handles,labels = g.axes.get_legend_handles_labels()
# plt.legend(handles=handles, labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
# plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])
cbar = plt.colorbar(sm)
cbar.ax.set_title(r"P(Y=1|X)", fontsize=30, y=1.02)
cbar.ax.tick_params(labelsize=20)
plt.savefig("plots/decboundary_al_colorbar.png")


# plt.savefig("plots/decboundary.png")

# +
colors = ["#368f8b", "#ec7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=200), rc={'figure.figsize':(15,15)})
# sns.set(style="white", palette=sns.color_palette("Set2"), rc={'figure.figsize':(15,15)})

plt.scatter(x=df.x1, y=df.x2, c=probs_final.detach().numpy()[:,1], s=(700), edgecolor="black", cmap=cmap)
plt.clim(0,1)

plt.plot([-2.5, 1.7],[y1, y2], linewidth=5, color="black")

# p.set(yticks=[], xticks=[])
# handles,labels = g.axes.get_legend_handles_labels()
# plt.legend(handles=handles, labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

# plt.colorbar(sm)

plt.savefig("plots/decboundary.png")

# +
colors = ["#368f8b", "#ec7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

sns.set(style="white", palette=sns.color_palette("mycolormap", n_colors=200), rc={'figure.figsize':(15,15)})
# sns.set(style="white", palette=sns.color_palette("Set2"), rc={'figure.figsize':(15,15)})

plt.scatter(x=df.x1, y=df.x2, c=probs_final_al.detach().numpy()[:,1], s=(700), edgecolor="black", cmap=cmap)
plt.clim(0,1)

plt.plot([-2.5, 1.7],[y1_al, y2_al], linewidth=5, color="black")

# p.set(yticks=[], xticks=[])
# handles,labels = g.axes.get_legend_handles_labels()
# plt.legend(handles=handles, labels=labels, loc="lower right", prop={'size': 30}, title=r'$\bf{wl1-wl2-wl3}$')
plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

# plt.colorbar(sm)

plt.savefig("plots/decboundary_al.png")

# +
colors = ["#368f8b", "#ec7357"]

norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap="mycolormap", norm=norm)
sm.set_array([])

fig, ax = plt.subplots(figsize=(17,15))

scatter = ax.scatter(x=df.x1, y=df.x2, c=true_test.detach().numpy()[:,1], s=(700), edgecolor="black", cmap=cmap)
labels=["0-0-0", "0-1-0", "0-1-1", "1-0-0", "1-1-0", "1-1-1"]
# plt.legend()
handles, _ = scatter.legend_elements()

legend1 = ax.legend(handles=handles, labels=labels,
                    loc="upper left", title="", fontsize="large")
ax.add_artist(legend1)

plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
# scatter.clim(0,1)

plt.xlim(-2.2,1.6)
plt.ylim(-2.1,3.1)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(sm)


plt.savefig("plots/trueproblabels.png")
# -

df[["x1","x2"]].values.shape



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
# +
# _, inv_idx = np.unique(label_matrix[:, :-1], axis=0, return_inverse=True)

# +
# plot_probs(df, probs=inv_idx, soft_labels=False)

# +
# psi_y, wl_idx_y = lm._get_psi(label_matrix, [[0],[1],[2],[3]], 4)

# +
# pd.DataFrame(np.linalg.pinv(np.cov(psi_y.T))).style.apply(color, axis=None)

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 100}


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
L = df_big[["wl1", "wl2", "wl3"]].values

lm = LabelModel(df=df_big,
                active_learning=False,
                add_cliques=True,
                add_prob_loss=False,
                n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
lm.analyze()
lm.print_metrics()
# -

Y_probs=lm._predict(L, psi, lm.mu, df.y.mean())

probs_final_big = fm._predict(torch.Tensor(df_big[["x1","x2"]].values))
decbound_points = df_big.iloc[np.argsort(np.abs(probs_final_big[:,1]-0.5))[:2]][["x1", "x2"]].values
a = (decbound_points[1,1] - decbound_points[0,1])/(decbound_points[1,0] - decbound_points[0,0])
b = decbound_points[0,1] - decbound_points[0,0]*a
y1 = a*-2.5 + b
y2 = a*1.7 + b

probs_final_big = fm_al._predict(torch.Tensor(df_big[["x1","x2"]].values))
decbound_points = df_big.iloc[np.argsort(np.abs(probs_final_big[:,1]-0.5))[:2]][["x1", "x2"]].values
a = (decbound_points[1,1] - decbound_points[0,1])/(decbound_points[1,0] - decbound_points[0,0])
b = decbound_points[0,1] - decbound_points[0,0]*a
y1_al = a*-2.5 + b
y2_al = a*1.7 + b

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=df_big[["x1","x2"]].values, labels=Y_probs.detach().numpy())._predict(torch.Tensor(df[["x1","x2"]].values))
# fm.analyze()
# fm.print_metrics()

fm_al = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm_al.fit(features=df_big[["x1","x2"]].values, labels=Y_probs_al.detach().numpy())._predict(torch.Tensor(df[["x1","x2"]].values))

# +
true_probs, uniqueidx = np.unique(lm.predict_true()[:, 1], return_index=True)
true_probs_counts = np.array(lm.predict_true_counts()[uniqueidx, 1])

fig = go.Figure()
fig.add_trace(go.Scatter(x=true_probs_counts, y=true_probs, mode='markers', showlegend=False, marker_color=np.array(px.colors.qualitative.Pastel)[0]))
fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100), line=dict(dash="longdash", color=np.array(px.colors.qualitative.Pastel)[1]), showlegend=False))

fig.update_yaxes(range=[0, 1], title_text="True from Junction Tree ")
fig.update_xaxes(range=[0, 1], title_text="True from P(Y, lambda)")
fig.update_layout(template="plotly_white", width=1000, height=500)
fig.show()

# +
true_probs_list=[]
true_probs_counts_list=[]
P_lambda_list=[]

N = 800
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

for i in range(10):

    data = SyntheticData(N, p_z, centroids)
    df = data.sample_data_set().create_df()
    
    wl_1 = np.random.uniform(-1, 2)
    wl_2 = np.random.uniform(-1.5, 1)
    wl_3 = np.random.uniform(-1.5, 1)

    df.loc[:, "wl1"] = (df["x2"]<wl_1)*1
    df.loc[:, "wl2"] = (df["x1"]<wl_2)*1
    df.loc[:, "wl3"] = (df["x1"]<wl_3)*1

    label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])
    L = label_matrix[:, :-1]

    lm = LabelModel(df=df,
                    active_learning=False,
                    add_cliques=True,
                    add_prob_loss=False,
                    n_epochs=200,
                    lr=1e-1)

    Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
    true_probs, uniqueidx = np.unique(lm.predict_true()[:, 1], return_index=True)
    true_probs_counts = np.array(lm.predict_true_counts()[uniqueidx, 1])
    true_probs_list.append(list(true_probs.clip(0,1)))
    true_probs_counts_list.append(list(true_probs_counts))
    P_lambda_list.append(list(lm.P_lambda[uniqueidx].squeeze()))

# fig = go.Figure(go.Scatter(x=lm.P_lambda[uniqueidx].squeeze(), y=np.abs(np.array(true_probs - true_probs_counts)), mode="markers"))
# fig.update_layout(template="plotly_white", xaxis_title="P(lambda)", title_text="Deviation true and junction tree posteriors")
# fig.show()

# +
df_small_set = pd.DataFrame({"True from Junction Tree": [item for sublist in true_probs_list for item in sublist], "True from P(Y, lambda)": [item for sublist in true_probs_counts_list for item in sublist], "P_lambda": [item.item() for sublist in P_lambda_list for item in sublist]})
df_small_set["Dataset size"] = "N=800"

# df_large_set = pd.DataFrame({"True from Junction Tree": [item for sublist in true_probs_list for item in sublist], "True from P(Y, lambda)": [item for sublist in true_probs_counts_list for item in sublist], "P_lambda": [item for sublist in P_lambda_list for item in sublist]})
# df_large_set["Dataset size"] = "N=1000000"

line_05 = pd.DataFrame({"x": np.linspace(0,1,100), "y": np.linspace(0,1,100)})
# -

df_both_sets = pd.concat([df_small_set, df_large_set])

# +
# df_both_sets.to_csv("results/posteriors_comparison.csv", index=False)

# +
sns.set_context("paper")
# colors = ["#086788",  "#ef7b45",  "#e3b505", "#000000", "#000000", "#d88c9a"]
# colors = ["#000000","#e3b505", "#ef7b45", "#086788"]
colors = ["#368f8b", "#ec7357"]

sns.set(style="whitegrid", palette="Set2")

# sns.relplot(data=df_both_sets, y="True from Junction Tree", x="True from P(Y, lambda)", col="Dataset size")


fig, axes = plt.subplots(1,2, figsize=(15,8), sharey=True)

sns.scatterplot(data=df_small_set, y="True from Junction Tree", x="True from P(Y, lambda)", ax=axes[0], hue="Dataset size", size="P_lambda", legend=False, palette=sns.color_palette([colors[1]]))
sns.lineplot(data=line_05, x="x", y="y", ax=axes[0], legend=False, palette=sns.color_palette([colors[0]]))
axes[0].lines[0].set_linestyle("--")

sns.scatterplot(data=df_large_set, y="True from Junction Tree", x="True from P(Y, lambda)", hue="Dataset size", size="P_lambda", legend=False, palette=sns.color_palette([colors[1]]), ax=axes[1])
sns.lineplot(data=line_05, x="x", y="y", ax=axes[1], legend=False, palette=sns.color_palette([colors[0]]))
axes[1].lines[0].set_linestyle("--")

axes[0].set_title("N=800")
axes[1].set_title("N=1000000")

axes[0].set_xlabel("Posterior from counts")
axes[1].set_xlabel("Posterior from counts")
axes[0].set_ylabel("Posterior from junction tree")
# axes[0].set_ylabel("")

plt.tight_layout()

# plt.savefig("plots/posteriors.png")
plt.show()

# +
diff_probs = np.abs(np.array([item for sublist in true_probs_list for item in sublist]) - np.array([item for sublist in true_probs_counts_list for item in sublist]))

fig = go.Figure(go.Scatter(x=[item for sublist in P_lambda_list for item in sublist], y=diff_probs, mode="markers"))
fig.update_layout(template="plotly_white", xaxis_title="P(lambda)", title_text="Deviation true and junction tree posteriors")
fig.show()

# +
diff_probs = np.abs(np.array([item for sublist in true_probs_list for item in sublist]) - np.array([item for sublist in true_probs_counts_list for item in sublist]))

fig = go.Figure(go.Scatter(x=[item for sublist in P_lambda_list for item in sublist], y=diff_probs, mode="markers"))
fig.update_layout(template="plotly_white", xaxis_title="P(lambda)", title_text="Deviation true and junction tree posteriors")
fig.show()
# -

al_kwargs["df"] = df_big

df_big[["wl1", "wl2", "wl3"]].values

al_kwargs["df"] = df_big

# +
it = 30
query_strategy = "relative_entropy"
# L = label_matrix[:, :-1]
al_kwargs["active_learning"] = "probs"

al = ActiveLearningPipeline(it=it,
#                             final_model = DiscriminativeModel(df, **final_model_kwargs),
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=df_big[["wl1", "wl2", "wl3"]].values, cliques=cliques, class_balance=class_balance, label_matrix_test=df[["wl1", "wl2", "wl3"]].values, y_test=df["y"].values)
al.label_model.print_metrics()
# -

al.plot_metrics(al.metrics)

al.label_model.predict()

plot_probs(df, Y_probs_al.detach().numpy())

entropy_df = pd.DataFrame.from_dict(al.bucket_AL_values).stack().reset_index().rename(columns={"level_0": "WL bucket", "level_1": "Active Learning Iteration", 0: "KL divergence"})

entropy_df["WL bucket"] = entropy_df["WL bucket"].map(al.confs)

sns.set(rc={'figure.figsize':(15,10)})
sns.set_theme(style="whitegrid")
divergence_plot = sns.lineplot(data=entropy_df, x="Active Learning Iteration", y="KL divergence", hue="WL bucket", palette="Set2")
# fig = divergence_plot.get_figure()
# fig.savefig("plots/divergence_plot.png")

al.plot_iterations()


def active_learning_experiment(nr_al_it, nr_runs, al_approach, query_strategy, randomness):
    al_metrics = {}
    al_metrics["lm_metrics"] = {}
    al_metrics["fm_metrics"] = {}
    al_kwargs["active_learning"] = al_approach

    for i in tqdm(range(nr_runs)):
        L = label_matrix[:, :-1]

        al = ActiveLearningPipeline(it=nr_al_it,
                                    final_model = DiscriminativeModel(df, **final_model_kwargs),
                                    **al_kwargs,
                                    query_strategy=query_strategy,
                                    randomness=randomness)

        Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
#         al.label_model.print_metrics()
        al_metrics["lm_metrics"][i] = al.metrics
        al_metrics["fm_metrics"][i] = al.final_metrics
#         al.plot_metrics()
        
    return al_metrics, Y_probs_al, al


runs=10
it=30

al_metrics_re, Y_probs_al_re, al_re = active_learning_experiment(it, runs, "probs", "relative_entropy", 0)

al_metrics_marg, Y_probs_al_marg, al_marg = active_learning_experiment(it, runs, "probs", "margin", 0)

al_metrics_random, Y_probs_al_random, al_random = active_learning_experiment(it, runs, "probs", "relative_entropy", 1)


# +
# al_metrics_hybrid, Y_probs_al_hybrid, al_hybrid = active_learning_experiment(it, runs, None, "hybrid", 0)
# -

def create_metric_df(al_metrics, nr_runs, metric_string, strategy_string):
    joined_metrics = pd.DataFrame()
    for i in range(nr_runs):
        int_df = pd.DataFrame.from_dict(al_metrics[metric_string][i]).drop("Labels").T
        int_df = int_df.stack().reset_index().rename(columns={"level_0": "Active Learning Iteration", "level_1": "Metric", 0: "Value"})
        int_df["Run"] = str(i)

        joined_metrics = pd.concat([joined_metrics, int_df])

    joined_metrics["Value"] = joined_metrics["Value"].apply(pd.to_numeric)
    joined_metrics["Strategy"] = strategy_string
    # joined_metrics = joined_metrics[joined_metrics["Run"] != "7"]
    
    return joined_metrics


# +
metric_df_re_lm = create_metric_df(al_metrics_re, runs, "lm_metrics", "Relative Entropy")
metric_df_re_fm = create_metric_df(al_metrics_re, runs, "fm_metrics", "Relative Entropy")

# metric_df_marg_lm = create_metric_df(al_metrics_marg, runs, "lm_metrics", "Margin")
# metric_df_marg_fm = create_metric_df(al_metrics_marg, runs, "fm_metrics", "Margin")

# metric_df_random_lm = create_metric_df(al_metrics_random, runs, "lm_metrics", "Random")
# metric_df_random_fm = create_metric_df(al_metrics_random, runs, "fm_metrics", "Random")

# # metric_df_marg_hybrid_lm = create_metric_df(al_metrics_hybrid, runs, "lm_metrics", "Hybrid")
# # metric_df_marg_hybrid_fm = create_metric_df(al_metrics_hybrid, runs, "fm_metrics", "Hybrid")

# # joined_df_lm = pd.concat([metric_df_re_lm, metric_df_random_lm, metric_df_marg_hybrid_lm])
# # joined_df_fm = pd.concat([metric_df_re_fm, metric_df_random_fm, metric_df_marg_hybrid_fm])

# joined_df_lm = pd.concat([metric_df_re_lm, metric_df_marg_lm, metric_df_random_lm])
# joined_df_fm = pd.concat([metric_df_re_fm, metric_df_marg_fm, metric_df_random_fm])
# -

sns.set_theme(style="white")

ax = sns.relplot(data=metric_df_re_lm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, hue="Metric",legend=False)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=metric_df_re_fm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, hue="Metric",legend=False)
(ax.set_titles("{col_name}"))

joined_df_lm.to_csv("results/lm_re_random_2.csv")
joined_df_fm.to_csv("results/fm_re_random_2.csv")

# +
# joined_df_fm = pd.read_csv("results/fm_re_random.csv", index_col=0)
# -

ax = sns.relplot(data=joined_df_lm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci="sd", estimator="mean", hue="Strategy", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_df_lm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, estimator="mean", hue="Strategy", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

colors = ["#086788",  "#e3b505","#ef7b45",  "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))

ax = sns.relplot(data=joined_df_fm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=1000, hue="Strategy", legend=True)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_df_fm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci="sd", hue="Strategy", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_df_fm[joined_df_fm["Strategy"] == "Relative Entropy"], x="Active Learning Iteration", y="Value", estimator=None, hue="Run", col="Metric", kind="line", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))


joined_df_lm["Model"] = "Generative"
joined_df_fm["Model"] = "Discriminative"
joined_df_models = pd.concat([joined_df_lm, joined_df_fm])
joined_df_models = joined_df_models[joined_df_models["Metric"] == "Accuracy"].rename(columns={"Value": "Accuracy"})

joined_df_models

sns.set_context("paper")
colors = ["#086788",  "#ef7b45", "#e3b505", "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
ax = sns.relplot(data=joined_df_models, x="Active Learning Iteration", y="Accuracy", col="Model", hue="Strategy", ci=68, n_boot=1000, kind="line", facet_kws={"despine": False})
# plt.grid(figure=ax.axes[0], alpha=0.2)
# plt.grid(figure=ax.fig, alpha=0.2)
(ax.set_titles("{col_name}"))

joined_df_lm[joined_df_lm["Strategy"] == "Random"]

al.plot_iterations()

pd.DataFrame.from_dict(al_metrics["probs"])

al_metrics["probs"]

# +
it = 20
query_strategy = "margin"
L = label_matrix[:, :-1]

al = ActiveLearningPipeline(it=it,
#                             final_model = DiscriminativeModel(df, **final_model_kwargs),
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al_mar = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.label_model.print_metrics()
# -

al.plot_iterations()

al.plot_metrics()

al.plot_animation()

data_test = SyntheticData(5000, p_z, centroids)
df_test = data_test.sample_data_set().create_df()

df_test.loc[:, "wl1"] = (df_test["x2"]<0.4)*1
df_test.loc[:, "wl2"] = (df_test["x1"]<-0.3)*1
df_test.loc[:, "wl3"] = (df_test["x1"]<-1)*1
df_test.loc[:, "wl4"] = (df_test["x2"]<-0.5)*1
df_test.loc[:, "wl5"] = (df_test["x1"]<0)*1

label_matrix_test = np.array(df_test[["wl1", "wl2", "wl3","y"]])
L_test = label_matrix_test[:, :-1]

# +
psi_test, _ = al.label_model._get_psi(L_test, cliques, al.label_model.nr_wl)

probs_test = al.label_model._predict(L_test, psi_test, al.label_model.mu, torch.tensor(al.label_model.E_S))
# -

al.label_model._analyze(probs_test, df_test['y'].values)

probs_test_lm = lm._predict(L_test, psi_test, lm.mu, torch.tensor(lm.E_S))
lm._analyze(probs_test_lm, df_test['y'].values)

lm.mu

al.label_model.mu

al.label_model.wl_idx

plot_probs(df_test, probs_test.detach().numpy())

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=lm.predict_true().detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

plot_train_loss(fm.losses)

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=Y_probs.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al_mar = fm.fit(features=data.X, labels=Y_probs_al_mar.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al_re = fm.fit(features=data.X, labels=Y_probs_al_re.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

plot_probs(df, lm.predict_true().detach().numpy())

plot_probs(df, Y_probs_al_mar.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, Y_probs_al_re.detach().numpy(), add_labeled_points=al.queried)

plot_probs(df, Y_probs.detach().numpy())

plot_probs(df, probs_final.detach().numpy())

plot_probs(df, probs_final_al_re.detach().numpy())

plot_probs(df, probs_final_al_mar.detach().numpy())

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



lm





ax = sns.relplot(data=cov_df, x="Active Learning Iteration", y="Value", hue="Strategy", col="Metric", row="Model", kind="line", ci=None, legend=False)
(ax.set_titles("{col_name}"))






