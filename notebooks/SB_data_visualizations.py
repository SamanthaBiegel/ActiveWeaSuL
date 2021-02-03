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
import matplotlib.colors as clr
import matplotlib.cm
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
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel, CustomTensorDataset
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed
from plot import plot_probs, plot_train_loss
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

set_seed(932)
data = SyntheticDataGenerator(N, p_z, centroids)
df = data.sample_dataset().create_df()

df.loc[:, "wl1"] = (df.x2<0.4)*1
df.loc[:, "wl2"] = (df.x1<-0.3)*1
df.loc[:, "wl3"] = (df.x1<-1)*1

label_matrix = df[["wl1", "wl2", "wl3"]].values
# -

cmap = clr.LinearSegmentedColormap.from_list('', ['#368f8b',"#BBBBBB",'#ec7357'], N=200)
matplotlib.cm.register_cmap("mycolormap", cmap)

bp1 = centroids.sum(axis=0)/2
diff = centroids[1,:]-centroids[0,:]
slope = diff[1]/diff[0]
perp_slope = -1/slope
b = bp1[1] - perp_slope*bp1[0]
coef = [b, perp_slope, -1]
x_dec = np.linspace(centroids[0,0]-4, centroids[1,0]+4, 1000)
y_dec = (- coef[0] - coef[1]*x_dec)/coef[2]

# +
sns.set(style="white",rc={'figure.figsize':(15,15)})

plt.scatter(x=df.x1, y=df.x2, c=df.y, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot(x_dec, y_dec, color="black")
plt.scatter(0.1, 1.3, s=200, color="black")
plt.scatter(-0.8, -0.5, s=200, color="black")

plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.savefig("../plots/truesyntdata.png")

# +
plt.scatter(x=df.x1, y=df.x2, c=df.wl1, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("../plots/truesyntdata_wl1.png")

# +
plt.scatter(x=df.x1, y=df.x2, c=df.wl2, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("../plots/truesyntdata_wl2.png")

# +
plt.scatter(x=df.x1, y=df.x2, c=df.wl3, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("../plots/truesyntdata_wl3.png")
# -
# ## Small example dataset

# +
N = 200
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

set_seed(246)
data = SyntheticDataGenerator(N, p_z, centroids)
df_example = data.sample_dataset().create_df()

df_example.loc[:, "wl1"] = (df_example.x2<0.4)*1
df_example.loc[:, "wl2"] = (df_example.x1<-0.3)*1
df_example.loc[:, "wl3"] = (df_example.x1<-1)*1

label_matrix_example = df_example[["wl1", "wl2", "wl3"]].values

# +
colors = ["#368f8b", "#ec7357"]

sns.set(style="white", palette=sns.color_palette(colors), rc={'figure.figsize':(15,15)})

sns.set_context("paper")

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c=df_example.y, s=500, edgecolor="black", cmap=cmap)
plt.plot(x_dec, y_dec, color="black")
plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.savefig("../plots/truelabels.png")

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c="#BBBBBB", s=500, edgecolor="black", cmap=cmap)
plt.legend(labels="?", loc="lower right", prop={'size': 30})
plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.savefig("../plots/missinglabels.png")

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c=df_example.wl1, s=500, edgecolor="black", cmap=cmap)

plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")

plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)

plt.xticks([], [])
plt.yticks([], [])

plt.savefig("../plots/wl1.png")

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c=df_example.wl2, s=500, edgecolor="black", cmap=cmap)

plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")

plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)

plt.xticks([], [])
plt.yticks([], [])

plt.savefig("../plots/wl2.png")

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c=df_example.wl3, s=500, edgecolor="black", cmap=cmap)

plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.savefig("../plots/wl3.png")
# +
class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]

L = df[["wl1", "wl2", "wl3"]].values

lm = LabelModel(n_epochs=200,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()


Y_probs_example = lm.predict(label_matrix_example, lm.mu, df_example.y.mean())

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c=Y_probs_example[:,1].detach().numpy(), s=500, edgecolor="black", cmap=cmap)

plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")

plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.show()

# +
set_seed(243)

train_dataset = CustomTensorDataset(X=torch.Tensor(df[["x1", "x2"]].values), Y=Y_probs.detach())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

test_dataset = CustomTensorDataset(X=torch.Tensor(df_example[["x1", "x2"]].values), Y=Y_probs_example.detach())
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

dm = LogisticRegression(input_dim=2, output_dim=2, lr=1e-3, n_epochs=3, soft_labels=True)

train_preds = dm.fit(train_loader).predict()
test_preds = dm.predict(test_dataloader)
# -

decbound_points = df.iloc[np.argsort(np.abs(train_preds[:,1]-0.5))[:2]][["x1", "x2"]].values
a = (decbound_points[1,1] - decbound_points[0,1])/(decbound_points[1,0] - decbound_points[0,0])
b = decbound_points[0,1] - decbound_points[0,0]*a
y1 = a*-2.5 + b
y2 = a*1.7 + b

# +
plt.scatter(x=df_example.x1, y=df_example.x2, c=test_preds[:,1].detach().numpy(), s=500, edgecolor="black", cmap=cmap)

plt.plot([-2.5, 1.7],[y1, y2], linewidth=5, color="black")

plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.show()


# +
plt.scatter(x=df.x1, y=df.x2, c=train_preds[:,1].detach().numpy(), s=500, edgecolor="black", cmap=cmap)

plt.plot([-2.5, 1.7],[y1, y2], linewidth=5, color="black")

plt.xlim(-2.7,2.2)
plt.ylim(-2,2.9)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks([], [])
plt.yticks([], [])

plt.show()
# -










