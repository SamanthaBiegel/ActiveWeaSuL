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
from scipy.stats import entropy
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

# ### Bucket entropy over time

# +
it = 30
query_strategy = "relative_entropy"
L = label_matrix[:, :-1]

al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)

# entropy_sampled_buckets = []

# for j in range(it):
#     bucket_list = al.unique_inverse[al.queried[:j+1]]
#     entropy_sampled_buckets.append(entropy([len(np.where(bucket_list == j)[0])/len(bucket_list) for j in range(6)]))


# -

def bucket_entropy_experiment(strategy, randomness):

    entropies = {}
    for i in tqdm(range(10), desc="Repetitions"):
        it = 30
        query_strategy = strategy
        L = label_matrix[:, :-1]

        al = ActiveLearningPipeline(it=it,
                                    **al_kwargs,
                                    query_strategy=query_strategy,
                                    randomness=randomness)

        Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)

        entropy_sampled_buckets = []

        for j in range(it):
            bucket_list = al.unique_inverse[al.queried[:j+1]]
            entropy_sampled_buckets.append(entropy([len(np.where(bucket_list == j)[0])/len(bucket_list) for j in range(6)]))

        entropies[i] = entropy_sampled_buckets
        
    return entropies


entropies_marg = bucket_entropy_experiment("margin", 0)
entropies_re = bucket_entropy_experiment("relative_entropy", 0)
entropies_random = bucket_entropy_experiment("margin", 1)

# +
entropies_marg_df = pd.DataFrame.from_dict(entropies_marg).stack().reset_index().rename(columns={"level_0": "Number of points acquired", "level_1": "Run", 0: "Entropy"})
entropies_marg_df["Strategy"] = "Margin"

entropies_re_df = pd.DataFrame.from_dict(entropies_re).stack().reset_index().rename(columns={"level_0": "Number of points acquired", "level_1": "Run", 0: "Entropy"})
entropies_re_df["Strategy"] = "Relative Entropy"

entropies_random_df = pd.DataFrame.from_dict(entropies_random).stack().reset_index().rename(columns={"level_0": "Number of points acquired", "level_1": "Run", 0: "Entropy"})
entropies_random_df["Strategy"] = "Random"

entropies_joined = pd.concat([entropies_re_df, entropies_marg_df, entropies_random_df])

entropies_joined["Number of points acquired"] = entropies_joined["Number of points acquired"].apply(lambda x: x+1)
# -

entropies_joined = entropies_joined.rename(columns={"Number of points acquired": "Number of labeled points"})

# +
# entropies_joined.to_csv("results/bucket_entropies.csv", index=False)

# +
# entropies_joined = pd.read_csv("results/bucket_entropies.csv")

# +
colors = ["#d88c9a",  "#086788",  "#e3b505","#ef7b45",  "#739e82"]

sns.set(rc={'figure.figsize':(15, 10)}, style="whitegrid", palette=sns.color_palette(colors))
ax = sns.lineplot(data=entropies_joined, x="Number of labeled points", y="Entropy", hue="Strategy", ci=68, n_boot=10000, hue_order=["Random", "Relative Entropy", "Margin"])
handles,labels = ax.axes.get_legend_handles_labels()
labels = ["Random", "MaxKL", "Margin"]
plt.legend(handles=handles, labels=labels, loc="lower right")
plt.show()
# fig = ax.get_figure()
# fig.savefig("plots/entropies.png")
# -

entropy_df = pd.DataFrame.from_dict(al.bucket_AL_values).stack().reset_index().rename(columns={"level_0": "WL bucket", "level_1": "Active Learning Iteration", 0: "KL divergence"})

entropy_df["WL bucket"] = entropy_df["WL bucket"].map(al.confs)

sns.set(rc={'figure.figsize':(15,10)})
sns.set_theme(style="whitegrid")
divergence_plot = sns.lineplot(data=entropy_df, x="Active Learning Iteration", y="KL divergence", hue="WL bucket", palette="Set2")
# fig = divergence_plot.get_figure()
# fig.savefig("plots/divergence_plot.png")

def active_learning_experiment(nr_al_it, nr_runs, al_approach, query_strategy, randomness, penalty_strength):
    al_metrics = {}
    al_metrics["lm_metrics"] = {}
    al_metrics["fm_metrics"] = {}
    al_kwargs["active_learning"] = al_approach

    for i in range(nr_runs):
        L = label_matrix[:, :-1]

        al = ActiveLearningPipeline(it=nr_al_it,
                                    final_model = DiscriminativeModel(df, **final_model_kwargs),
                                    **al_kwargs,
                                    penalty_strength=penalty_strength,
                                    query_strategy=query_strategy,
                                    randomness=randomness)

        Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance, label_matrix_test=L, y_test=df["y"].values)
        al.label_model.print_metrics()
        al.final_model.print_metrics()
        al_metrics["lm_metrics"][i] = al.metrics
        al_metrics["fm_metrics"][i] = al.final_metrics
        
    return al_metrics


runs=10
it=30

# +
test_strengths = [1, 1e2, 1e3, 1e4, 1e5]
strength_metrics = {}

for test, strength in enumerate(test_strengths):

    strength_metrics[test] = active_learning_experiment(it, runs, "probs", "relative_entropy", 0, strength)


# -

def create_metric_df(al_metrics, nr_runs, metric_string, strategy_string):
    joined_metrics = pd.DataFrame()
    for i in range(nr_runs):
        int_df = pd.DataFrame.from_dict(al_metrics[metric_string][i]).drop("Labels").T
        int_df = int_df.stack().reset_index().rename(columns={"level_0": "Active Learning Iteration", "level_1": "Metric", 0: "Value"})
        int_df["Run"] = str(i)

        joined_metrics = pd.concat([joined_metrics, int_df])

    joined_metrics["Value"] = joined_metrics["Value"].apply(pd.to_numeric)
    joined_metrics["Lambda"] = strategy_string
    # joined_metrics = joined_metrics[joined_metrics["Run"] != "7"]
    
    return joined_metrics


for key, value in strength_metrics.items():
    if key == 0:
        joined_lambdas = create_metric_df(value, 10, "lm_metrics", test_strengths[key])
    else:
        joined_lambdas = pd.concat([joined_lambdas, create_metric_df(value, 10, "lm_metrics", test_strengths[key])])


for key, value in strength_metrics.items():
    if key == 0:
        joined_lambdas_final = create_metric_df(value, 10, "fm_metrics", test_strengths[key])
    else:
        joined_lambdas_final = pd.concat([joined_lambdas, create_metric_df(value, 10, "fm_metrics", test_strengths[key])])

joined_lambdas = joined_lambdas[joined_lambdas["Metric"] == "Accuracy"]
joined_lambdas["Model"] = "Generative"

joined_lambdas_final = joined_lambdas_final[joined_lambdas_final["Metric"] == "Accuracy"]
joined_lambdas_final["Model"] = "Discriminative"

joined_lambdas_models = pd.concat([joined_lambdas, joined_lambdas_final])

joined_lambdas_models

colors = ["#086788",  "#e3b505","#ef7b45",  "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors), rc={'figure.figsize':(15,10)})


joined_lambdas_filter = joined_lambdas[joined_lambdas["Lambda"] != 1e2]

ax = sns.relplot(data=joined_lambdas_models, x="Active Learning Iteration", y="Value", col="Model", ci=68, n_boot=1000, estimator="mean", hue="Lambda", kind="line", palette=sns.color_palette(colors), legend=True)
# (ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_lambdas, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, estimator="mean", hue="Lambda", palette="deep", legend=True)
(ax.set_titles("{col_name}"))

al_metrics_random, Y_probs_al_random, al_random = active_learning_experiment(it, runs, "probs", "relative_entropy", 1)

al_metrics_hybrid, Y_probs_al_hybrid, al_hybrid = active_learning_experiment(it, runs, None, "hybrid", 0)


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
# metric_df_re_fm = create_metric_df(al_metrics_re, runs, "fm_metrics", "Relative Entropy")
metric_df_re_lm = metric_df_re_lm[metric_df_re_lm["Run"] != "7"]
# metric_df_re_fm = metric_df_re_fm[metric_df_re_fm["Run"] != "7"]

metric_df_random_lm = create_metric_df(al_metrics_random, runs, "lm_metrics", "Random")
metric_df_random_fm = create_metric_df(al_metrics_random, runs, "fm_metrics", "Random")

metric_df_marg_hybrid = create_metric_df(al_metrics_re, runs, "lm_metrics", "Relative Entropy")
# metric_df_re_fm = create_metric_df(al_metrics_re, runs, "fm_metrics", "Relative Entropy")

joined_df_lm = pd.concat([metric_df_re_lm, metric_df_random_lm])
# joined_df_fm = pd.concat([metric_df_re_fm, metric_df_random_fm])
# -

sns.set_theme(style="white")

ax = sns.relplot(data=metric_df_re_lm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, hue="Metric",legend=False)
(ax.set_titles("{col_name}"))

# +
# joined_df_fm.to_csv("results/fm_re_random.csv")
# -

ax = sns.relplot(data=joined_df_lm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, estimator="mean", hue="Strategy", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_df_fm, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=10000, hue="Strategy", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))

ax = sns.relplot(data=joined_df_lm[joined_df_lm["Strategy"] == "Relative Entropy"], x="Active Learning Iteration", y="Value", estimator=None, hue="Run", col="Metric", kind="line", palette="deep", style="Strategy", legend=True)
(ax.set_titles("{col_name}"))


joined_df_lm = pd.read_csv("results/lm_re_random.csv")
joined_df_fm = pd.read_csv("results/fm_re_random.csv")

joined_df_lm["Model"] = "Generative"
joined_df_fm["Model"] = "Discriminative"
joined_df_models = pd.concat([joined_df_lm, joined_df_fm])
joined_df_models = joined_df_models[joined_df_models["Metric"] == "Accuracy"].rename(columns={"Value": "Accuracy"})

sns.set_context("paper")
colors = ["#086788",  "#ef7b45", "#e3b505", "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
ax = sns.relplot(data=joined_df_models, x="Active Learning Iteration", y="Accuracy", col="Model", hue="Strategy", ci=68, n_boot=10000, estimator="mean", kind="line", facet_kws={"despine": False})
# plt.grid(figure=ax.axes[0], alpha=0.2)
# plt.grid(figure=ax.fig, alpha=0.2)
(ax.set_titles("{col_name}"))

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

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=lm.predict_true().detach().numpy()).predict()
fm.analyze()
fm.print_metrics()

plot_train_loss(fm.losses)

df = pd.read_csv("../data/synthetic_dataset_3.csv")

random.seed(None)
df_1 = df.iloc[[random.choice(range(len(df)))]]

df_1[["x1", "x2"]].values.squeeze()

fm = DiscriminativeModel(df_1, **final_model_kwargs, soft_labels=False)
probs_final = fm._predict(torch.Tensor(df[["x1", "x2"]].values))
fm._analyze(probs_final, df["y"].values)

fm = DiscriminativeModel(df_1, **final_model_kwargs, soft_labels=False)
probs_final = fm.fit(features=df_1[["x1", "x2"]].values[None, :], labels=np.array(df_1["y"])[None])._predict(torch.Tensor(df[["x1", "x2"]].values))
fm._analyze(probs_final, df["y"].values)["Accuracy"]

torch.argmin(torch.abs(probs_final[:, 1] - probs_final[:, 0])).item()

probs_final[6977]

final_model_kwargs

final_model_kwargs["batch_size"] = 1

# +

accuracy_dict = {}
for j in tqdm(range(10)):
    accuracies = []
    queried = []
    fm = DiscriminativeModel(df_1, **final_model_kwargs, soft_labels=False)
    probs = fm._predict(torch.Tensor(df[["x1", "x2"]].values))
    accuracies.append(fm._analyze(probs, df["y"].values)["Accuracy"])

    for i in range(30):    
        queried.append(torch.argmin(torch.abs(probs[:, 1] - probs[:, 0])).item())

        if i==0:
            X = df.iloc[queried][["x1", "x2"]].values.squeeze()[None, :]
            y = np.array(df.iloc[queried]["y"]).squeeze()[None]
        else:
            X = df.iloc[queried][["x1", "x2"]].values
            y = np.array(df.iloc[queried]["y"])
    #     print(X.shape)
    #     print(y.shape)

        fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=False)
        probs = fm.fit(features=X, labels=y)._predict(torch.Tensor(df[["x1", "x2"]].values))
        accuracies.append(fm._analyze(probs, df["y"].values)["Accuracy"])
        
    accuracy_dict[j] = accuracies


# -

accuracy_df = pd.DataFrame.from_dict(accuracy_dict)

accuracy_df = accuracy_df.stack().reset_index().rename(columns={"level_0": "Active Learning Iteration", "level_1": "Run", 0: "Accuracy"})

accuracy_df

# +
# accuracy_df.to_csv("results/accuracy_only_AL.csv")
# -

sns.lineplot(data=accuracy_df, x="Active Learning Iteration", y="Accuracy", ci=68)

plot_probs(df, probs, add_labeled_points=queried)

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=False)
probs_final_al_mar = fm.fit(features=data.X, labels=data.y).predict()
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

# +
p_z = 0.5
# centroids = np.array([[0.1, 1.3], [-1, -1]])
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])

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
             'df': df,
             'n_epochs': 200
            }

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
it = 10
query_strategy = "relative_entropy"

L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
#                             beta=1,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
al.label_model.analyze()
al.label_model.print_metrics()
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


