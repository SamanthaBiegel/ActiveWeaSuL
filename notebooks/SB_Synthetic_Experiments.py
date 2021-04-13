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
import seaborn as sns
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("../activeweasul"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, add_baseline, synthetic_al_experiment
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

# ### Experiments

starting_seed = 36
penalty_strength = 1
nr_trials = 10
al_it = 30

# +
final_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-1,
                          n_epochs=100)

batch_size = 256
class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1,2]]
# -

# Plot settings
font_size=25
legend_size=25
tick_size=20
n_boot=10000
linewidth=4

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
                  final_model=LogisticRegression(**final_model_kwargs, early_stopping=True, patience=5, warm_start=False),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=y_train),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=y_test),
                  label_matrix_test=label_matrix_test,
                  y_test=y_test)

np.random.seed(284)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl_2, queried_maxkl_2, probs_maxkl_2, entropies_maxkl_2 = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

# #### Nashaat et al.

np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat, queried_nashaat, probs_nashaat, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat")

# #### Extend active learning iterations

exp_kwargs["al_it"] = 1000
exp_kwargs["discr_model_frequency"] = 50

# Active WeaSuL 1000 iterations
np.random.seed(284)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl_1000, _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

# Nashaat 1000 iterations
np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat_1000, _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat")

exp_kwargs["al_it"] = al_it
exp_kwargs["discr_model_frequency"] = 1

# #### Active learning

# +
set_seed(76)

train_dataset = CustomTensorDataset(X=df_train.loc[[0],["x1", "x2"]], Y=y_train[0])
predict_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1","x2"]].values), Y=torch.Tensor(y_train))
test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1","x2"]].values), Y=torch.Tensor(y_test))

al_exp_kwargs = dict(
    nr_trials=nr_trials,
    al_it=al_it,
    batch_size=batch_size,
    seeds = np.random.randint(0,1000,nr_trials),
    features = df_train.loc[:, ["x1","x2"]],
    y_train = y_train,
    y_test = y_test,
    train_dataset = train_dataset,
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=df_test.loc[:, ["x1", "x2"]].values
)
# -

al_accuracies = synthetic_al_experiment(**al_exp_kwargs)

from performance import PerformanceMixin

PerformanceMixin().analyze(y_test, torch.Tensor(np.full_like(y_test, 0)), soft_labels)

TN, FN, FP, TP = (((predicted_labels == i) & (y == j)).sum() for i in [0,1] for j in [0,1])

# #### Process results

# +
metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(al_accuracies, "Active learning by itself")]).reset_index(level=0).rename(columns={"level_0": "Run"})

joined_df = add_baseline(metric_dfs, al_it)

joined_df = joined_df[joined_df["Metric"] == "Accuracy"]
joined_df = joined_df[joined_df["Set"] == "test"]

# +
metric_dfs_1000 = pd.concat([process_exp_dict(metrics_nashaat_1000, "Nashaat et al."),
                             process_exp_dict(metrics_maxkl_1000, "Active WeaSuL")]).reset_index(level=0).rename(columns={"level_0": "Run"})

metric_dfs_1000 = metric_dfs_1000[metric_dfs_1000["Metric"] == "Accuracy"]
metric_dfs_1000 = metric_dfs_1000[metric_dfs_1000["Set"] == "test"]

# +
# joined_df.to_csv("../results/figure_3AB.csv", index=False)

# +
# metric_dfs_1000.to_csv("../results/figure_3C.csv", index=False)
# -

agg_df = joined_df.groupby(["Model", "Approach", "Number of labeled points"]).mean().reset_index().round(2)

agg_df[agg_df["Number of labeled points"] == 0]

agg_df[(agg_df["Approach"] == "Active learning by itself") & (agg_df["Model"] == "Discriminative")]

agg_df[(agg_df["Approach"] == "Active learning by itself") & (agg_df["Model"] == "Discriminative")]

# +
colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,3, figsize=(22.5,8), sharey=True)

plt.tight_layout()

sns.lineplot(data=joined_df[joined_df["Model"] == "Generative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", legend=False, linewidth=linewidth,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself"], ax=axes[0])

axes[0].set_title("Generative model (30 iterations)", size=font_size)

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"], ax=axes[1])
axes[1].get_legend().remove()
axes[1].set_title("Discriminative model (30 iterations)", fontsize=font_size)

ax = sns.lineplot(data=metric_dfs_1000[metric_dfs_1000["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
                  hue="Approach", ci=68, n_boot=n_boot, hue_order=["Active WeaSuL", "Nashaat et al."],
                  palette=sns.color_palette(colors[:2]), linewidth=linewidth)

handles, labels = axes[1].get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
leg = axes[2].legend(handles=handles, labels=labels, loc="lower right", title="Method", fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"
axes[2].set_title("Discriminative model (1000 iterations)", fontsize=font_size)

axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
axes[2].tick_params(axis='both', which='major', labelsize=tick_size)

axes[0].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[1].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[2].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("Accuracy", fontsize=font_size)

plt.ylim(0.5, 1)

plt.tight_layout()

# plt.savefig("../plots/performance_baselines_5.png")
plt.show()
# -




