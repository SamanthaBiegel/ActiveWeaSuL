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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath("../activeweasul"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from plot import plot_probs, plot_train_loss
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment, bucket_entropy_experiment, add_baseline, synthetic_al_experiment, add_optimal
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

Y_probs = lm.fit(label_matrix=label_matrix,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# lm.active_learning = True
# lm.penalty_strength = 1e4

# # Fit and predict on train set
# Y_probs = lm.fit(label_matrix=label_matrix,
#                  cliques=cliques,
#                  class_balance=class_balance,
#                  ground_truth_labels=y_train).predict()

# Predict on test set
Y_probs_test = lm.predict(label_matrix_test, lm.mu, p_z)

# Analyze test set performance
lm.analyze(y_test, Y_probs_test)
# -

plt.plot(lm.losses)
plt.show()

lm.analyze(y_train)

torch.norm((lm.cov_O_inverse + lm.z.detach() @ lm.z.detach().T)[torch.BoolTensor(lm.mask)]) ** 2

# +
# mu > z
true_mu = lm.get_true_mu(y_train)[:, 1]
cov_OS = true_mu[:, None] - torch.Tensor(lm.E_O[:, None] @ lm.E_S[None, None])
c = torch.inverse((lm.cov_S - cov_OS.T @ lm.cov_O_inverse @ cov_OS))
z = torch.sqrt(c) * lm.cov_O_inverse @ cov_OS

torch.norm((lm.cov_O_inverse + z @ z.T)[torch.BoolTensor(lm.mask)]) ** 2
# -

# z > mu
c = 1 / lm.cov_S * (1 + torch.mm(torch.mm(z.T, lm.cov_O), z))
cov_OS = torch.mm(lm.cov_O, z / torch.sqrt(c))
lm_mu = cov_OS + torch.Tensor(lm.E_O[:, None] @ lm.E_S[None, None])

# ### Fit discriminative model

# +
batch_size = 256

dm = LogisticRegression(**final_model_kwargs)

dm.reset()

indices_shuffle = np.random.permutation(len(label_matrix))
split_nr = int(np.ceil(0.9*len(label_matrix)))
train_idx, val_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]

train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach())
test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach())
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

dl_train = DataLoader(CustomTensorDataset(*train_dataset[train_idx]), shuffle=True, batch_size=batch_size)
dl_val = DataLoader(CustomTensorDataset(*train_dataset[val_idx]), shuffle=True, batch_size=batch_size)

train_preds = dm.fit(dl_train, dl_val).predict()
test_preds = dm.predict(test_dataloader)
# -

dm.analyze(y_test, test_preds)

# +
it = 0
# Choose strategy from ["maxkl", "margin", "nashaat"]
query_strategy = "maxkl"

seed = 76

al = ActiveWeaSuLPipeline(it=it,
                          final_model = LogisticRegression(**final_model_kwargs),
                          n_epochs=200,
                          query_strategy=query_strategy,
                          discr_model_frequency=1,
                          penalty_strength=1,
                          batch_size=256,
                          randomness=0,
                          seed=seed,
                          starting_seed=243)

Y_probs_al = al.run_active_weasul(label_matrix=label_matrix,
                                  y_train=y_train,
                                  label_matrix_test=label_matrix_test,
                                  y_test=y_test,
                                  cliques=cliques,
                                  class_balance=class_balance,
                                  train_dataset=CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach()),
                                  test_dataset=CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach()))
# -

al.color_cov(np.array(y_train))

lm.E_O

cov_OS = torch.Tensor([6.236878, -6.236878, 0.561111, -0.207885, 0, -0.353226])
cov_OS[:,None] + torch.Tensor(lm.E_O[[0,1,6,7,8,9], None] @ lm.E_S[None, None])

lm.cov_OS

cov_OS = torch.Tensor([6.236878, -6.236878, 1.544744, -0.346475, 0, -353226])
cov_OS[:,None] + torch.Tensor(lm.E_O[[0,1,6,7,8,9], None] @ lm.E_S[None, None])

al.color_cov(np.array(y_train))



plt.plot([loss for it_loss in al.lm_losses for loss in it_loss])
plt.show()

# +
# plot_it=5
# plot_probs(df, al.probs["Generative_train"][plot_it], soft_labels=False, add_labeled_points=al.queried[:plot_it])
# -

# ### Figure 1

starting_seed = 36
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
                  final_model=LogisticRegression(**final_model_kwargs, early_stopping=True),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=Y_probs.detach()),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=Y_probs_test.detach()),
                  label_matrix_test=label_matrix_test,
                  y_test=y_test)

np.random.seed(284)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl, queried_maxkl, probs_maxkl, entropies_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

margin_df = process_exp_dict(metrics_maxkl, "margin").reset_index(level=0)
margin_df = margin_df[margin_df["Metric"] == "Accuracy"]
margin_df = margin_df[margin_df["Set"] == "test"]

sns.relplot(data=margin_df, x="Number of labeled points", y="Value", col="Model", kind="line")

# #### Nashaat et al.

np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
exp_kwargs["final_model"]=LogisticRegression(**final_model_kwargs, early_stopping=True, warm_start=False)
metrics_nashaat, queried_nashaat, probs_nashaat, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)
exp_kwargs["final_model"]=LogisticRegression(**final_model_kwargs, early_stopping=True, warm_start=True)

# Active WeaSuL 1000 iterations
np.random.seed(284)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
exp_kwargs["al_it"] = 1000
exp_kwargs["discr_model_frequency"] = 20
metrics_maxkl_1000, _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl", randomness=0)
exp_kwargs["al_it"] = al_it
exp_kwargs["discr_model_frequency"] = 1

# Nashaat 1000 iterations
np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
exp_kwargs["al_it"] = 1000
exp_kwargs["discr_model_frequency"] = 20
exp_kwargs["final_model"]=LogisticRegression(**final_model_kwargs, early_stopping=True, warm_start=False)
metrics_nashaat_1000, _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)
exp_kwargs["al_it"] = al_it
exp_kwargs["discr_model_frequency"] = 1
exp_kwargs["final_model"]=LogisticRegression(**final_model_kwargs, early_stopping=True, warm_start=True)

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
    features = df_train.loc[:, ["x1","x2"]],
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

# optimal_generative_test = lm.analyze(y_test, lm.predict_true(y_train, y_test, label_matrix_test))
# optimal_discriminative_test = dm.analyze(y_test, test_preds)
# joined_df = add_optimal(joined_df, al_it, optimal_generative_test, optimal_discriminative_test)

# +
# joined_df.to_csv("../results/figure_3AB.csv", index=False)
# -

joined_df = joined_df[joined_df["Metric"] == "Accuracy"]
joined_df = joined_df[joined_df["Set"] == "test"]

# +
metric_dfs_1000 = pd.concat([process_exp_dict(metrics_nashaat_1000, "Nashaat et al."),
                             process_exp_dict(metrics_maxkl_1000, "Active WeaSuL")])

metric_dfs_1000 = metric_dfs_1000.reset_index(level=0).rename(columns={"level_0": "Run"})

# +
# metric_dfs_1000.to_csv("../results/figure_3C.csv", index=False)
# -

metric_dfs_1000 = metric_dfs_1000[metric_dfs_1000["Metric"] == "Accuracy"]
metric_dfs_1000 = metric_dfs_1000[metric_dfs_1000["Set"] == "test"]

font_size=25
legend_size=25
tick_size=20
n_boot=100
linewidth=4

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

# plt.savefig("../plots/performance_baselines_4.png")
plt.show()
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
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth,
            hue_order=["MaxKL", "Random", "Margin"], ax=axes[0])

handles, labels = axes[0].get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
axes[0].legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)
axes[0].set_title("Generative model", fontsize=font_size)

sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth, legend=False,
            hue_order=["MaxKL", "Random", "Margin"], ax=axes[1])

axes[1].set_title("Discriminative model", fontsize=font_size)

axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)

axes[0].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[1].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("Accuracy", fontsize=font_size)

plt.ylim(0.5,1)

plt.tight_layout()

# plt.savefig("../plots/sampling_stratfegies_2.png")
plt.show()
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
ax = sns.lineplot(data=entropies_df, x="Number of labeled points", y="Entropy", hue="Approach", ci=68,
                  linewidth=linewidth, n_boot=n_boot, hue_order=["Random", "MaxKL", "Margin"], legend=False)
# handles,labels = ax.axes.get_legend_handles_labels()
# [ha.set_linewidth(linewidth) for ha in handles]
# plt.legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.set_xlabel("Number of active learning iterations", fontsize=font_size)
ax.set_ylabel("Diversity (entropy)", fontsize=font_size)
ax.set_title("Diversity of sampled buckets", fontsize=font_size)
plt.ylim(-0.05,1.8)

plt.tight_layout()
# plt.savefig("../plots/entropies.png")
plt.show()
# -






