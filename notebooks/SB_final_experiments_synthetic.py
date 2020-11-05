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
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -

pd.options.display.expand_frame_repr = False 
np.set_printoptions(suppress=True, precision=16)

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

data = SyntheticData(N, p_z, centroids)
df = data.sample_data_set().create_df()

df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1

label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 100}

class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'df': df,
             'n_epochs': 200
            }


# -

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
                                    penalty_strength=1,
                                    query_strategy=query_strategy,
                                    randomness=randomness)

        Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
        al.label_model.print_metrics()
        al_metrics["lm_metrics"][i] = al.metrics
        al_metrics["fm_metrics"][i] = al.final_metrics
        al.plot_metrics()
        
    return al_metrics, Y_probs_al, al


runs=10
it=30

al_metrics_re, Y_probs_al_re, al_re = active_learning_experiment(it, runs, "probs", "relative_entropy", 0)

al_metrics_marg, Y_probs_al_marg, al_marg = active_learning_experiment(it, runs, "probs", "margin", 0)

al_metrics_random, Y_probs_al_random, al_random = active_learning_experiment(it, runs, "probs", "relative_entropy", 1)


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
# metric_df_re_lm_2 = create_metric_df(al_metrics_re_2, runs, "lm_metrics", "Relative Entropy")
# metric_df_re_fm_2 = create_metric_df(al_metrics_re_2, runs, "fm_metrics", "Relative Entropy")

metric_df_re_lm = create_metric_df(al_metrics_re, runs, "lm_metrics", "Relative Entropy")
# metric_df_re_fm = create_metric_df(al_metrics_re, runs, "fm_metrics", "Relative Entropy")

metric_df_marg_lm = create_metric_df(al_metrics_marg, runs, "lm_metrics", "Margin")
# metric_df_marg_fm = create_metric_df(al_metrics_marg, runs, "fm_metrics", "Margin")

# metric_df_marg_lm_2 = create_metric_df(al_metrics_marg_2, runs, "lm_metrics", "Margin")
# metric_df_marg_fm_2 = create_metric_df(al_metrics_marg_2, runs, "fm_metrics", "Margin")

metric_df_random_lm = create_metric_df(al_metrics_random, runs, "lm_metrics", "Random")
metric_df_random_fm = create_metric_df(al_metrics_random, runs, "fm_metrics", "Random")

# # metric_df_marg_hybrid_lm = create_metric_df(al_metrics_hybrid, runs, "lm_metrics", "Hybrid")
# # metric_df_marg_hybrid_fm = create_metric_df(al_metrics_hybrid, runs, "fm_metrics", "Hybrid")

# # joined_df_lm = pd.concat([metric_df_re_lm, metric_df_random_lm, metric_df_marg_hybrid_lm])
# # joined_df_fm = pd.concat([metric_df_re_fm, metric_df_random_fm, metric_df_marg_hybrid_fm])

joined_df_lm = pd.concat([metric_df_re_lm, metric_df_marg_lm, metric_df_random_lm])
joined_df_fm = pd.concat([metric_df_re_fm, metric_df_marg_fm, metric_df_random_fm])

# joined_df_lm = pd.concat([metric_df_re_lm, metric_df_re_lm_2, metric_df_marg_lm, metric_df_marg_lm_2])
# joined_df_fm = pd.concat([metric_df_re_fm, metric_df_re_fm_2, metric_df_marg_fm, metric_df_marg_fm_2])
# -

sns.set_theme(style="white")
colors = ["#086788",  "#e3b505","#ef7b45",  "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))

ax = sns.relplot(data=metric_df_re_fm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=1000, hue="Metric",legend=False)
(ax.set_titles("{col_name}"))

joined_df_lm = pd.read_csv("results/lm_re_random_2.csv")
joined_df_fm = pd.read_csv("results/fm_re_random_2.csv")

joined_df_lm = pd.concat([joined_df_lm, metric_df_marg_lm])
joined_df_fm = pd.concat([joined_df_fm, metric_df_marg_fm])

joined_df_lm["Model"] = "Generative"
joined_df_fm["Model"] = "Discriminative"
joined_df_models = pd.concat([joined_df_lm, joined_df_fm])
joined_df_models = joined_df_models[joined_df_models["Metric"] == "Accuracy"].rename(columns={"Value": "Accuracy"})

sns.set_context("paper")
colors = ["#086788",  "#ef7b45", "#e3b505", "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
ax = sns.relplot(data=joined_df_models, x="Active Learning Iteration", y="Accuracy", col="Model", hue="Strategy", ci=50, n_boot=1000, estimator=np.median, kind="line", facet_kws={"despine": False})
# plt.grid(figure=ax.axes[0], alpha=0.2)
# plt.grid(figure=ax.fig, alpha=0.2)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_df_fm[joined_df_fm["Strategy"] == "Relative Entropy"], x="Active Learning Iteration", y="Value", estimator=None, hue="Run", col="Metric", kind="line", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))






