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
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -

N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

df = pd.read_csv("../data/synthetic_dataset_3.csv")
label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])

# +

cmap = clr.LinearSegmentedColormap.from_list('', ['#368f8b',"#BBBBBB",'#ec7357'], N=200)
matplotlib.cm.register_cmap("mycolormap", cmap)
# -

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

plt.savefig("plots/truesyntdata.png")

# +
plt.scatter(x=df.x1, y=df.x2, c=df.wl1, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot([-5, 5],[0.4, 0.4], linewidth=.5, color="black")
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("plots/truesyntdata_wl1.png")

# +
plt.scatter(x=df.x1, y=df.x2, c=df.wl2, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot([-0.3, -0.3], [-5, 5], linewidth=.5, color="black")
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("plots/truesyntdata_wl2.png")

# +
plt.scatter(x=df.x1, y=df.x2, c=df.wl3, s=75, edgecolor="black", linewidth=0.5, cmap=cmap)
plt.plot([-1, -1], [-5, 5], linewidth=.5, color="black")
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.xlabel("x1", fontsize=30)
plt.ylabel("x2", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.savefig("plots/truesyntdata_wl3.png")
# -










