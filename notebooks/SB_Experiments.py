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
import os
import pandas as pd
import pickle
import random
from scipy.stats import entropy
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath("../activelearning"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from plot import plot_probs, plot_train_loss
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment, bucket_entropy_experiment, add_baseline, synthetic_al_experiment
# -

# ### Generate data

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

set_seed(932)
data = SyntheticDataGenerator(N, p_z, centroids)
df_train = data.sample_dataset().create_df()
y_train = df_train.y.values

# +
N = 3000

set_seed(466)
data = SyntheticDataGenerator(N, p_z, centroids)
df_test = data.sample_dataset().create_df()
y_test = df_test.y.values
# -

# ### Apply labeling functions

# +
df_train.loc[:, "wl1"] = (df_train.x2<0.4)*1
df_train.loc[:, "wl2"] = (df_train.x1<-0.3)*1
df_train.loc[:, "wl3"] = (df_train.x1<-1)*1

label_matrix = df_train[["wl1", "wl2", "wl3"]].values

# +
df_test.loc[:, "wl1"] = (df_test.x2<0.4)*1
df_test.loc[:, "wl2"] = (df_test.x1<-0.3)*1
df_test.loc[:, "wl3"] = (df_test.x1<-1)*1

label_matrix_test = df_test[["wl1", "wl2", "wl3"]].values
# -

# ### Fit label model

# +
final_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-1,
                          n_epochs=100)

class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]

# +
set_seed(243)

lm = LabelModel(n_epochs=200,
                lr=1e-1)

# Fit and predict on train set
Y_probs = lm.fit(label_matrix=label_matrix,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# Predict on test set
Y_probs_test = lm.predict(label_matrix_test, lm.mu, p_z)

# Analyze test set performance
lm.analyze(y_test, Y_probs_test)
# -

# ### Fit discriminative model

# +
batch_size = 256

set_seed(27)

train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=lm.predict_true(y_train).detach())
test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

dm = LogisticRegression(**final_model_kwargs)

dm.reset()
train_preds = dm.fit(train_loader).predict()
test_preds = dm.predict(test_dataloader)

# +
# plot_train_loss(dm.average_losses)

# +
# # it = 30
# # Choose strategy from ["maxkl", "margin", "nashaat"]
# query_strategy = "maxkl"

# seed = 631

# al = ActiveWeaSuLPipeline(it=1,
# #                           final_model = LogisticRegression(**final_model_kwargs),
#                           n_epochs=200,
#                           query_strategy=query_strategy,
#                           discr_model_frequency=5,
#                           penalty_strength=1,
#                           batch_size=256,
#                           randomness=0,
#                           seed=seed,
#                           starting_seed=243)

# Y_probs_al = al.run_active_weasul(label_matrix=label_matrix,
#                                   y_train=y_train,
#                                   cliques=cliques,
#                                   class_balance=class_balance,
#                                   train_dataset=CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach()))

# +
# plot_metrics(process_metric_dict(al.metrics, "MaxKL", remove_test=True))

# +
# plot_it=5
# plot_probs(df, al.probs["Generative_train"][plot_it], soft_labels=False, add_labeled_points=al.queried[:plot_it])
# -

# ### Figure 1

starting_seed = 243
penalty_strength = 1
nr_trials = 10
al_it = 30

# #### Active WeaSuL

exp_kwargs = dict(nr_trials=nr_trials,
                  al_it=al_it,
                  label_matrix=label_matrix,
                  y_train=y_train,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=starting_seed,
                  penalty_strength=penalty_strength,
                  batch_size=batch_size,
                  final_model=LogisticRegression(**final_model_kwargs, hide_progress_bar=True),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach()),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach()),
                  label_matrix_test=label_matrix_test,
                  y_test=y_test)

np.random.seed(284)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl, queried_maxkl, probs_maxkl, entropies_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

# #### Nashaat et al.

np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat, queried_nashaat, probs_nashaat, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)

# Nashaat 1000 iterations
np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
exp_kwargs["al_it"] = 1000
exp_kwargs["discr_model_frequency"] = 20
metrics_nashaat_1000, _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)
exp_kwargs["al_it"] = al_it
exp_kwargs["discr_model_frequency"] = 1

# #### Active learning

# +
set_seed(76)

predict_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1","x2"]].values), Y=torch.Tensor(y_train))
test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1","x2"]].values), Y=torch.Tensor(y_test))

al_exp_kwargs = dict(
    nr_trials=nr_trials,
    al_it=al_it,
    batch_size=batch_size,
    seeds = np.random.randint(0,1000,10),
    features = df_train.loc[:, ["x1", "x2"]],
    y_train = y_train,
    y_test = y_test,
    train_dataset = CustomTensorDataset(X=df_train.loc[[0],["x1", "x2"]], Y=y_train[0]),
    predict_dataloader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=df_test.loc[:, ["x1", "x2"]].values
)
# -

al_accuracies = synthetic_al_experiment(**al_exp_kwargs)

# #### Process results

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(al_accuracies, "Active learning by itself")])
metric_dfs = metric_dfs.reset_index(level=0).rename(columns={"level_0": "Run"})
metric_dfs["Dash"] = "n"

# +
joined_df = add_baseline(metric_dfs, al_it)

optimal_generative_test = lm.analyze(y_test, lm.predict_true(y_train, y_test, label_matrix_test))
optimal_discriminative_test = dm.analyze(y_test, test_preds)
joined_df = add_optimal(joined_df, al_it, optimal_generative_test, optimal_discriminative_test)
# -

joined_df = joined_df[joined_df["Metric"] == "Accuracy"]
joined_df = joined_df[joined_df["Set"] == "test"]

nashaat_df = process_exp_dict(metrics_nashaat_1000, "Nashaat et al.")
nashaat_df = nashaat_df[nashaat_df["Metric"] == "Accuracy"]
nashaat_df = nashaat_df[nashaat_df["Set"] == "test"]

font_size=25
legend_size=25
tick_size=20
n_boot=10000

# +
colors = ["#000000", "#2b4162", "#368f8b", "#ec7357", "#e9c46a"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,3, figsize=(22.5,8), sharey=True)

plt.tight_layout()

sns.lineplot(data=joined_df[joined_df["Model"] == "Generative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", style="Dash", legend=False,
            hue_order=["Upper bound","Active WeaSuL", "Nashaat et al.", "Weak supervision by itself"], ax=axes[0])

# handles, labels = axes[0].get_legend_handles_labels()
# leg = axes[0].legend(handles=handles[1:], labels=labels[1:5], loc="lower right", title="Method", fontsize=legend_size, title_fontsize=legend_size)
# leg._legend_box.align = "left"
# leg_lines = leg.get_lines()
# leg_lines[0].set_linestyle("--")
axes[0].set_title("Generative model (30 iterations)", size=font_size)

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", style="Dash",
            hue_order=["Upper bound","Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"], ax=axes[1])
axes[1].legend([],[])
axes[1].set_title("Discriminative model (30 iterations)", fontsize=font_size)

ax = sns.lineplot(data=nashaat_df[nashaat_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value", hue="Approach", ci=68, n_boot=n_boot, palette=sns.color_palette([colors[2]]))
# handles,labels = ax.axes.get_legend_handles_labels()
# leg = plt.legend(handles=handles, labels=labels, loc="lower right", title="Method", fontsize=legend_size, title_fontsize=legend_size)
# leg._legend_box.align = "left"
handles, labels = axes[1].get_legend_handles_labels()
leg = axes[2].legend(handles=handles[1:], labels=labels[1:6], loc="lower right", title="Method", fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"
leg_lines = leg.get_lines()
leg_lines[0].set_linestyle("--")
axes[2].set_title("Discriminative model (1000 iterations)", fontsize=font_size)

axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
axes[2].tick_params(axis='both', which='major', labelsize=tick_size)

axes[0].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[1].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[2].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("Accuracy", fontsize=font_size)

plt.ylim(0.48, 0.98)

plt.tight_layout()

plt.savefig("../plots/performance_baselines.png")
# plt.show()
# -

# ### Figure 2AB

# #### Other sampling strategies

np.random.seed(568)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_margin, queried_margin, probs_margin, entropies_margin = active_weasul_experiment(**exp_kwargs, query_strategy="margin")

np.random.seed(568)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_random, queried_random, probs_random, entropies_random = active_weasul_experiment(**exp_kwargs, query_strategy="margin", randomness=1)

# #### Process results

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "MaxKL"),
                        process_exp_dict(metrics_margin, "Margin"),
                        process_exp_dict(metrics_random, "Random")])

# +
metric_dfs = metric_dfs[metric_dfs.Set != "train"]
metric_dfs = metric_dfs[metric_dfs["Metric"].isin(["Accuracy"])]

lines = list(metric_dfs.Approach.unique())

colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"][:len(lines)]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,2, figsize=(15,8), sharey=True)

sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Generative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean",
            hue_order=["MaxKL", "Random", "Margin"], ax=axes[0])

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)
axes[0].set_title("Generative model", fontsize=font_size)

sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean",
            hue_order=["MaxKL",  "Random", "Margin"], ax=axes[1])

handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)
axes[1].set_title("Discriminative model", fontsize=font_size)

axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)

axes[0].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[1].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("Accuracy", fontsize=font_size)

plt.ylim(0.85,0.978)

plt.tight_layout()

plt.savefig("../plots/sampling_strategies.png")
# -

# ### Figure 2C

# #### Process entropies

# +
maxkl_entropies_df = pd.DataFrame.from_dict(entropies_maxkl).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
maxkl_entropies_df["Approach"] = "MaxKL"

margin_entropies_df = pd.DataFrame.from_dict(entropies_margin).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
margin_entropies_df["Approach"] = "Margin"

random_entropies_df = pd.DataFrame.from_dict(entropies_random).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
random_entropies_df["Approach"] = "Random"
# -

entropies_df = pd.concat([maxkl_entropies_df, margin_entropies_df, random_entropies_df])
entropies_df["Number of labeled points"] = entropies_df["Number of labeled points"].apply(lambda x: x+1)

# +
colors = ["#368f8b","#2b4162", "#ec7357"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

plt.subplots(1,1,figsize=(8,8))
ax = sns.lineplot(data=entropies_df, x="Number of labeled points", y="Entropy", hue="Approach", ci=68, n_boot=n_boot, hue_order=["Random", "MaxKL", "Margin"])
handles,labels = ax.axes.get_legend_handles_labels()
plt.legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.set_xlabel("Number of active learning iterations", fontsize=font_size)
ax.set_ylabel("Diversity (entropy)", fontsize=font_size)
ax.set_title("Diversity of sampled buckets", fontsize=font_size)
plt.ylim(-0.05,1.8)

plt.tight_layout()
plt.savefig("../plots/entropies.png")
# -






