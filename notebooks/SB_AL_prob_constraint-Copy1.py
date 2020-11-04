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

N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

data = SyntheticData(N, p_z, centroids)
df = data.sample_data_set().create_df()

plot_probs(df, probs=data.y, soft_labels=False)

df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1
df.loc[:, "wl4"] = (df["x2"]<-0.5)*1
df.loc[:, "wl5"] = (df["x1"]<0)*1


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
                      'n_epochs': 100}

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
lambda_combs, lambda_index, lambda_counts = np.unique(lm.label_matrix, axis=0, return_counts=True, return_inverse=True)
new_counts = lambda_counts.copy()
rows_not_abstain, cols_not_abstain = np.where(lambda_combs != -1)
for i, comb in enumerate(lambda_combs):
    nr_non_abstain = (comb != -1).sum()
    if nr_non_abstain < lm.nr_wl:
        if nr_non_abstain == 0:
            new_counts[i] = 0
        else:
            match_rows = np.where((lambda_combs[:, cols_not_abstain[rows_not_abstain == i]] == lambda_combs[i, cols_not_abstain[rows_not_abstain == i]]).all(axis=1))       
            new_counts[i] = lambda_counts[match_rows].sum()

P_lambda = torch.Tensor((new_counts/lm.N)[lambda_index][:, None])

# +
lambda_combs, lambda_index, lambda_counts = np.unique(np.concatenate([lm.label_matrix,df.y.values[:,None]],axis=1), axis=0, return_counts=True, return_inverse=True)

P_Y_lambda = np.zeros((lm.N, 2))

P_Y_lambda[df.y.values == 0, 0] = ((lambda_counts/lm.N)[lambda_index]/P_lambda.squeeze())[df.y.values == 0]
P_Y_lambda[df.y.values == 0, 1] = 1 - P_Y_lambda[df.y.values == 0, 0]

P_Y_lambda[df.y.values == 1, 1] = ((lambda_counts/lm.N)[lambda_index]/P_lambda.squeeze())[df.y.values == 1]
P_Y_lambda[df.y.values == 1, 0] = 1 - P_Y_lambda[df.y.values == 1, 1]                                                               

# +
true_probs = lm.predict_true()[:, 1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=P_Y_lambda[:, 1], y=true_probs, mode='markers', showlegend=False, marker_color=np.array(px.colors.qualitative.Pastel)[0]))
fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100), line=dict(dash="longdash", color=np.array(px.colors.qualitative.Pastel)[1]), showlegend=False))

fig.update_yaxes(range=[0, 1], title_text="True from Junction Tree ")
fig.update_xaxes(range=[0, 1], title_text="True from P(Y, lambda)")
fig.update_layout(template="plotly_white", width=1000, height=500)
fig.show()
# -

fig = go.Figure(go.Scatter(x=P_lambda.squeeze(), y=np.array(true_probs)-P_Y_lambda[:,1], mode="markers"))
fig.update_layout(template="plotly_white", xaxis_title="P(lambda)", title_text="Deviation true and junction tree posteriors")
fig.show()

final_model_kwargs

al_kwargs

entropies_random = {}
for i in range(10):
    it = 30
    query_strategy = "margin"
    L = label_matrix[:, :-1]
    al_kwargs["active_learning"] = "probs"

    al = ActiveLearningPipeline(it=it,
    #                             final_model = DiscriminativeModel(df, **final_model_kwargs),
                                **al_kwargs,
                                query_strategy=query_strategy,
                                randomness=1)

    Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
    al.label_model.print_metrics()
    
    entropy_sampled_buckets = []

    for j in range(it):
        bucket_list = al.unique_inverse[al.queried[:j+1]]
        entropy_sampled_buckets.append(entropy([len(np.where(bucket_list == j)[0])/len(bucket_list) for j in range(6)]))
        
    entropies_random[i] = entropy_sampled_buckets

entropies_marg[0]

entropies_marg_df = pd.DataFrame.from_dict(entropies_marg).stack().reset_index().rename(columns={"level_0": "Number of points acquired", "level_1": "Run", 0: "Entropy"})
entropies_marg_df["Strategy"] = "Margin"

entropies_re_df = pd.DataFrame.from_dict(entropies_re).stack().reset_index().rename(columns={"level_0": "Number of points acquired", "level_1": "Run", 0: "Entropy"})
entropies_re_df["Strategy"] = "Relative Entropy"

entropies_random_df = pd.DataFrame.from_dict(entropies_random).stack().reset_index().rename(columns={"level_0": "Number of points acquired", "level_1": "Run", 0: "Entropy"})
entropies_random_df["Strategy"] = "Random"

entropies_joined = pd.concat([entropies_re_df, entropies_marg_df, entropies_random_df])

entropies_joined.to_csv("results/bucket_entropies.csv", index=False)

entropies_joined["Number of points acquired"] = entropies_joined["Number of points acquired"].apply(lambda x: x+1)

# +
# sns.set_theme(style="white")
colors = ["#d88c9a",  "#086788",  "#e3b505","#ef7b45",  "#739e82"]

sns.set(rc={'figure.figsize':(15, 10)}, style="whitegrid", palette=sns.color_palette(colors))
ax = sns.lineplot(data=entropies_joined, x="Number of points acquired", y="Entropy", hue="Strategy", ci=68, n_boot=10000, hue_order=["Random", "Relative Entropy", "Margin"])
plt.gca().legend().set_title('')
fig = ax.get_figure()
fig.savefig("plots/entropies.png")
# -

entropies_joined

al.plot_metrics()

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

    for i in range(nr_runs):
        L = label_matrix[:, :-1]

        al = ActiveLearningPipeline(it=nr_al_it,
                                    final_model = DiscriminativeModel(df, **final_model_kwargs),
                                    **al_kwargs,
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

metric_df_re_fm = metric_df_re_fm[metric_df_re_fm["Run"] != "7"]
metric_df_re_fm

joined_df_lm[joined_df_lm["Run"] == "7"]

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


