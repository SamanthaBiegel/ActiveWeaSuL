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

# data = SyntheticData(N, p_z, centroids)
# df = data.sample_data_set().create_df()

# df.loc[:, "wl1"] = (df["x2"]<0.4)*1
# df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
# df.loc[:, "wl3"] = (df["x1"]<-1)*1

# label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])

# +
# df.to_csv("../data/synthetic_dataset_3.csv", index=False)
# -

df = pd.read_csv("../data/synthetic_dataset_3.csv")
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

# +
L = label_matrix[:, :-1]
al_kwargs["active_learning"] = "probs"    

al = ActiveLearningPipeline(it=10,
                            final_model = DiscriminativeModel(df, **final_model_kwargs),
                            **al_kwargs,
                            query_strategy="margin",
                            randomness=1)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.label_model.print_metrics()
# -

al.final_probs

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=df[["x1", "x2"]].values, labels=Y_probs_al.detach().numpy()).predict()
fm.analyze()
fm.accuracy()

probs_final

al.final_prob_dict[10]

al.label_model.cov_AL

cov_OS_al = al.label_model.cov_OS[list(itertools.chain.from_iterable([al.label_model.wl_idx[clique] for clique in ["3", "0_3", "1_3", "2_3", "1_2_3"]]))]


al

al.label_model.mu[al.label_model.max_clique_idx]

lm.mu[lm.max_clique_idx]

Y_probs_al

al.label_model.predict_true()

al.plot_iterations()

al.plot_metrics()

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
# -

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=df[["x1", "x2"]].values, labels=Y_probs.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=df[["x1", "x2"]].values, labels=Y_probs_al.detach().numpy()).predict()
fm.analyze()
fm.print_metrics()


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

# +
# al_metrics_test, Y_probs_al_test, al_test = active_learning_experiment(it, 10, "probs", "relative_entropy", 0)
# -

al.label_model.wl_idx

al_metrics_cov, Y_probs_al_cov, al_cov = active_learning_experiment(it, runs, "cov", "relative_entropy", 0)

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
# metric_df_re_lm_2 = create_metric_df(al_metrics_re_2, runs, "lm_metrics", "Relative Entropy")
# metric_df_re_fm_2 = create_metric_df(al_metrics_re_2, runs, "fm_metrics", "Relative Entropy")

metric_df_re_lm = create_metric_df(al_metrics_re, runs, "lm_metrics", "Relative Entropy")
metric_df_re_fm = create_metric_df(al_metrics_re, runs, "fm_metrics", "Relative Entropy")

metric_df_marg_lm = create_metric_df(al_metrics_marg, runs, "lm_metrics", "Margin")
metric_df_marg_fm = create_metric_df(al_metrics_marg, runs, "fm_metrics", "Margin")

# metric_df_marg_lm_2 = create_metric_df(al_metrics_marg_2, runs, "lm_metrics", "Margin")
# metric_df_marg_fm_2 = create_metric_df(al_metrics_marg_2, runs, "fm_metrics", "Margin")

metric_df_random_lm = create_metric_df(al_metrics_random, runs, "lm_metrics", "Random")
metric_df_random_fm = create_metric_df(al_metrics_random, runs, "fm_metrics", "Random")

# metric_df_marg_hybrid_lm = create_metric_df(al_metrics_hybrid, runs, "lm_metrics", "Hybrid")
# metric_df_marg_hybrid_fm = create_metric_df(al_metrics_hybrid, runs, "fm_metrics", "Hybrid")

# joined_df_lm = pd.concat([metric_df_re_lm, metric_df_random_lm, metric_df_marg_hybrid_lm])
# joined_df_fm = pd.concat([metric_df_re_fm, metric_df_random_fm, metric_df_marg_hybrid_fm])

# joined_df_lm = pd.concat([metric_df_re_lm, metric_df_marg_lm, metric_df_random_lm, metric_df_marg_hybrid_lm])
# joined_df_fm = pd.concat([metric_df_re_fm, metric_df_marg_fm, metric_df_random_fm, metric_df_marg_hybrid_fm])

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

# +
# joined_df_lm = pd.read_csv("results/lm_re_random.csv")
# joined_df_fm = pd.read_csv("results/fm_re_random.csv")

# +
# joined_df_lm = pd.concat([joined_df_lm, metric_df_marg_lm])
# joined_df_fm = pd.concat([joined_df_fm, metric_df_marg_fm])
# -

joined_df_lm["Model"] = "Generative"
joined_df_fm["Model"] = "Discriminative"
joined_df_models = pd.concat([joined_df_lm, joined_df_fm])
joined_df_models = joined_df_models[joined_df_models["Metric"] == "Accuracy"].rename(columns={"Value": "Accuracy"})
joined_df_models["Dash"] = "y"

# +
optimal_fm = pd.DataFrame({"Active Learning Iteration": list(range(it+1)), "Accuracy": np.repeat(0.9759, it+1)})
optimal_fm["Model"] = "Discriminative"
optimal_fm["Run"] = 0
optimal_fm["Strategy"] = "DM*"
optimal_fm["Metric"] = "Accuracy"
optimal_fm["Dash"] = "n"

optimal_lm = pd.DataFrame({"Active Learning Iteration": list(range(it+1)), "Accuracy": np.repeat(0.9648, it+1)})
optimal_lm["Model"] = "Generative"
optimal_lm["Run"] = 0
optimal_lm["Strategy"] = "GM*"
optimal_lm["Metric"] = "Accuracy"
optimal_lm["Dash"] = "n"

joined_df_models = pd.concat([joined_df_models, optimal_lm, optimal_fm])

# +
# joined_df_models.to_csv("results/re_marg_random_joined.csv", index=False)
# -

pd.read_csv("results/re_marg_random_joined.csv")

# +
sns.set_context("paper")
colors = ["#086788",  "#ef7b45",  "#e3b505", "#000000", "#000000", "#d88c9a"]
colors = ["#000000","#e3b505", "#ef7b45", "#086788"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,2, figsize=(15,8), sharey=True)

(
    sns.lineplot(data=joined_df_models[joined_df_models["Model"] == "Generative"], x="Active Learning Iteration", y="Accuracy",
                hue="Strategy", ci=68, n_boot=10000, estimator="mean", style="Dash", sizes=(1, 2), hue_order=["GM*", "Random", "Margin", "Relative Entropy"],
                size="Dash", ax=axes[0])
)
handles, labels = axes[0].get_legend_handles_labels()
show_handles = [handles[4], handles[3], handles[2]]
show_labels = [labels[4], labels[3], labels[2]]
axes[0].legend(handles=show_handles, labels=show_labels, loc="lower right")
axes[0].title.set_text("Generative")

(
    sns.lineplot(data=joined_df_models[joined_df_models["Model"] == "Discriminative"], x="Active Learning Iteration", y="Accuracy",
                hue="Strategy", ci=68, n_boot=10000, estimator="mean", style="Dash", sizes=(1, 2), hue_order=["DM*", "Random", "Margin", "Relative Entropy"],
                size="Dash", ax=axes[1])
)
handles, labels = axes[1].get_legend_handles_labels()
show_handles = [handles[4], handles[3], handles[2]]
show_labels = [labels[4], labels[3], labels[2]]
axes[1].legend(handles=show_handles, labels=show_labels, loc="lower right")
axes[1].title.set_text("Discriminative")

plt.tight_layout()

# plt.savefig("plots/strategies.png")
plt.show()
# -

sns.set_context("paper")
# colors = ["#086788",  "#ef7b45", "#e3b505", "#739e82", "#d88c9a"]
colors = ["#086788",  "#ef7b45", "#000000", "#000000", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
# fig, ax = plt.subplots()
g = (
    sns.relplot(data=joined_df_models, x="Active Learning Iteration", y="Accuracy", col="Model",
                hue="Strategy", ci=68, n_boot=1000, estimator="mean", kind="line", style="Dash", sizes=(1, 2),
                size="Dash", facet_kws={"despine": False})
)
handles,labels = g.axes[0][0].get_legend_handles_labels()
plt.legend(handles=handles[1:3], labels=labels[1:3], loc="lower right")
# plt.grid(figure=ax.axes[0], alpha=0.2)
# plt.grid(figure=ax.fig, alpha=0.2)
g.set_titles("{col_name}")
plt.show()




# +
sns.set_context("paper")
colors = ["#086788",  "#ef7b45", "#e3b505", "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
ax = sns.relplot(data=joined_df_models, x="Active Learning Iteration", y="Accuracy", col="Model", hue="Strategy", ci=50, n_boot=1000, estimator=np.median, kind="line", facet_kws={"despine": False})
# plt.grid(figure=ax.axes[0], alpha=0.2)
# plt.grid(figure=ax.fig, alpha=0.2)
ax.set_titles("{col_name}")




# +
# joined_df_models.to_csv("results/strategies.csv", index=False)
# -



ax = sns.relplot(data=joined_df_fm[joined_df_fm["Strategy"] == "Margin"], x="Active Learning Iteration", y="Value", estimator=None, hue="Run", col="Metric", kind="line", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=al_re.label_model.predict_true().detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=False)
probs_final = fm.fit(features=data.X, labels=df["y"].values).predict()
fm.analyze()
fm.print_metrics()

al_re.label_model._accuracy(al_re.label_model.predict_true(), df["y"].values)



optimal_fm

joined_df_models

for j in range(50):
    N = 10000
    centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
    p_z = 0.5

    data = SyntheticData(N, p_z, centroids)
    df = data.sample_data_set().create_df()

    df.loc[:, "wl1"] = (df["x2"]<0.4)*1
    df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
    df.loc[:, "wl3"] = (df["x1"]<-0.8)*1

    label_matrix = np.array(df[["wl1", "wl2", "wl3","y"]])
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
    print(np.unique(lm.predict_true(), axis=0))


