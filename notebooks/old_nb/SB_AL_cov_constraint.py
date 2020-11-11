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

sys.path.append(os.path.abspath("../../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs
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

df = pd.read_csv("../../data/synthetic_dataset_3.csv")
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
             'df': df,
             "lr": 1e-1
            }

# +
al_metrics = {}
al_metrics["lm_metrics"] = {}
al_metrics["fm_metrics"] = {}

for j in tqdm(range(10)):
    it = 100
    query_strategy = "relative_entropy"

    L = label_matrix[:, :-1]

    al = ActiveLearningPipeline(it=it,
                                final_model = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True),
                                **al_kwargs,
                                n_epochs=200,
                                query_strategy=query_strategy,
                                randomness=0)

    Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
    al_metrics["lm_metrics"][j] = al.metrics
    al_metrics["fm_metrics"][j] = al.final_metrics
    print("Accuracy:", al.label_model._accuracy(Y_probs_al, df["y"].values))
    al.plot_metrics()


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
metric_df_cov_lm = create_metric_df(al_metrics, 10, "lm_metrics", "Cov Constraint")
metric_df_cov_fm = create_metric_df(al_metrics, 10, "fm_metrics", "Cov Constraint")

metric_df_cov_lm["Model"] = "Generative"
metric_df_cov_fm["Model"] = "Discriminative"
joined_df_models = pd.concat([metric_df_cov_lm, metric_df_cov_fm])
joined_df_models = joined_df_models[joined_df_models["Metric"] == "Accuracy"].rename(columns={"Value": "Accuracy"})

# +
sns.set_context("paper")
colors = ["#086788",  "#ef7b45",  "#e3b505", "#000000", "#000000", "#d88c9a"]
colors = ["#000000","#e3b505", "#ef7b45", "#086788"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,2, figsize=(15,8), sharey=True)

(
    sns.lineplot(data=joined_df_models[joined_df_models["Model"] == "Generative"], x="Active Learning Iteration", y="Accuracy",
                ci=68, n_boot=1000, estimator="mean",
                ax=axes[0])
)
# handles, labels = axes[0].get_legend_handles_labels()
# show_handles = [handles[4], handles[3], handles[2]]
# show_labels = [labels[4], labels[3], labels[2]]
# axes[0].legend(handles=show_handles, labels=show_labels, loc="lower right")
axes[0].title.set_text("Generative")

(
    sns.lineplot(data=joined_df_models[joined_df_models["Model"] == "Discriminative"], x="Active Learning Iteration", y="Accuracy",
                ci=68, n_boot=1000, estimator="mean",
                ax=axes[1])
)
# handles, labels = axes[1].get_legend_handles_labels()
# show_handles = [handles[4], handles[3], handles[2]]
# show_labels = [labels[4], labels[3], labels[2]]
# axes[1].legend(handles=show_handles, labels=show_labels, loc="lower right")
axes[1].title.set_text("Discriminative")

plt.tight_layout()

# plt.savefig("plots/strategies.png")
plt.show()
# -

ax = sns.relplot(data=joined_df_models[joined_df_models["Model"] == "Discriminative"], x="Active Learning Iteration", y="Accuracy", estimator=None, hue="Run", kind="line", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

# +
from scipy.stats import entropy

lm_posteriors = al.unique_prob_dict[100]
lm_posteriors = np.concatenate([1-lm_posteriors[:, None], lm_posteriors[:, None]], axis=1).clip(1e-5, 1-1e-5)

rel_entropy = np.zeros(len(lm_posteriors))
sample_posteriors = np.zeros(lm_posteriors.shape)

for i in range(len(lm_posteriors)):
    bucket_items = al.ground_truth_labels[np.where(al.unique_inverse == i)[0]]
    bucket_gt = bucket_items[bucket_items != -1]
    bucket_gt = np.array(list(bucket_gt) + [np.round(al.unique_prob_dict[0][i])])
    # if bucket_gt.size == 0:
    #     eps = 1e-2
    #     sample_posteriors[i, 1] = np.argmax(lm_posteriors[i, :]).clip(eps, 1-eps)

    # else:
    eps = 1e-2/(len(bucket_gt))
    sample_posteriors[i, 1] = bucket_gt.mean().clip(eps, 1-eps)

    sample_posteriors[i, 0] = 1 - sample_posteriors[i, 1]

    rel_entropy[i] = entropy(lm_posteriors[i, :], sample_posteriors[i, :])#/len(bucket_gt)
# -

rel_entropy

lm_posteriors

sample_posteriors

al.label_model.predict_true_counts()[al.unique_idx, 1]

df["y"][al.unique_inverse == 2].mean()

al.final_metrics

from plot import plot_train_loss
plot_train_loss(al.label_model.losses)

al.label_model.print_metrics()

al.label_model.mu[al.label_model.max_clique_idx]

al.label_model.get_true_mu()[al.label_model.max_clique_idx]

al.plot_iterations()

al.plot_parameters()

al.plot_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=False)
probs_final_true = fm.fit(features=data.X, labels=data.y).predict()
fm.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=df[["x1", "x2"]].values, labels=Y_probs_al.detach().numpy()).predict()
fm.analyze()
fm.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=df[["x1", "x2"]].values, labels=np.concatenate([(1 - al.prob_dict[100])[:, None], (al.prob_dict[100])[:, None]], axis=1)).predict()
fm.analyze()
fm.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_first = fm.fit(features=df[["x1", "x2"]].values, labels=np.concatenate([(1 - al.prob_dict[0])[:, None], (al.prob_dict[0])[:, None]], axis=1)).predict()
fm._accuracy(probs_first, df["y"].values)

probs_final

al.final_prob_dict[0]

al.final_probs

plot_probs(df, probs_final.detach().numpy(), soft_labels=True, subset=None)





al.plot_iterations()

al.plot_iterations()

al.plot_parameters()

for i in range(100):
    if (i == 0) | ((i+1) % 10 == 0):
        fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
        probs_final = fm.fit(features=df[["x1", "x2"]].values, labels=np.concatenate([(1 - al.prob_dict[i])[:, None], (al.prob_dict[i])[:, None]], axis=1)).predict()
        al_metrics["fm_metrics"][i]["Accuracy"] = fm._accuracy(probs_final, df["y"].values)

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




