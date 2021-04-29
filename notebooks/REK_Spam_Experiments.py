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
import pickle
import random
from scipy.stats import entropy
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath("../activeweasul"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from experiments import process_metric_dict, active_weasul_experiment, process_exp_dict, active_learning_experiment, add_weak_supervision_baseline, synthetic_al_experiment
# -

# ### Load data

# +
path_prefix = "../data/spam/"

L_train = pickle.load(open(path_prefix + "L_train.pkl", "rb"))
L1 = pickle.load(open(path_prefix + "L_test.pkl", "rb"))

df_occup = pickle.load(open(path_prefix + "X_train.pkl", "rb"))
df1 = pickle.load(open(path_prefix + "X_test.pkl", "rb"))

y_train = pickle.load(open(path_prefix + "Y_train.pkl", "rb"))
y1 = pickle.load(open(path_prefix + "Y_test.pkl", "rb"))

# +
indices_keep = L_train.sum(axis=1) != -7
L_train = L_train[indices_keep]
y_train = y_train[indices_keep]

df_occup = pd.DataFrame.sparse.from_spmatrix(df_occup)
df1 = pd.DataFrame.sparse.from_spmatrix(df1).reset_index()
df_occup = df_occup.iloc[indices_keep].reset_index()
# -

np.unique(y_train, return_counts=True)

L_train.shape, L1.shape, y_train.shape

label_matrix = L_train
label_matrix_test = L1
y_test = y1

682 / (480 + 682)

# ### Fit label model

# +
p_z = 0.58

class_balance = np.array([1 - p_z, p_z])
cliques = [[0],[1],[2],[3],[4],[5],[6]]

# +
set_seed(34)

lm = LabelModel(n_epochs=200,
                lr=1e-2)

# Fit and predict on train set
Y_probs = lm.fit(label_matrix=label_matrix,
                 cliques=cliques,
                 class_balance=class_balance).predict()

# Predict on test set
Y_probs_test = lm.predict(label_matrix_test, lm.mu, p_z)

# Analyze test set performance
lm.analyze(y_test, Y_probs_test)
# -

lm.loss_func()

# ### Fit discriminative model

# +
batch_size = 256

set_seed(34)

features = df_occup.columns

train_dataset = CustomTensorDataset(
    X=torch.Tensor(df_occup[features].values), 
    Y=Y_probs.detach())
test_dataset = CustomTensorDataset(
    X=torch.Tensor(df1[features].values), 
    Y=Y_probs_test.detach())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

final_model_kwargs = dict(input_dim=df_occup.shape[1],
                          output_dim=2,
                          lr=1e-2,
                          n_epochs=100)

dm = LogisticRegression(**final_model_kwargs)

dm.reset()
train_preds = dm.fit(train_loader).predict()
test_preds = dm.predict(test_dataloader)
# -

# ### Figure 1

starting_seed = 34
penalty_strength = 1e6
nr_trials = 5
al_it = 100

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
                  final_model=LogisticRegression(**final_model_kwargs),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_occup[features].values), Y=Y_probs.detach()),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df1[features].values), Y=Y_probs_test.detach()),
                  label_matrix_test=label_matrix_test,
                  y_test=y_test)

# +
# np.random.seed(284)
# exp_kwargs["seeds"]= np.random.randint(0, 1000, 10)
# metrics_maxkl, queried_maxkl, probs_maxkl, entropies_maxkl = active_weasul_experiment(
#     **exp_kwargs, 
#     query_strategy="maxkl"
# )
# -

# #### Nashaat et al.

np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat, queried_nashaat, probs_nashaat, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)

pickle.dump(metrics_nashaat, open("metrics_nashaat_spam.pkl", "wb"))
pickle.dump(queried_nashaat, open("queried_nashaat_spam.pkl", "wb"))
pickle.dump(probs_nashaat, open("probs_nashaat_spam.pkl", "wb"))

# +
# metrics_nashaat = pickle.load(open("metrics_nashaat_spam.pkl", "rb"))
# queried_nashaat = pickle.load(open("queried_nashaat_spam.pkl", "rb"))
# probs_nashaat = pickle.load(open("probs_nashaat_spam.pkl", "rb"))
# -

# Nashaat 1000 iterations
np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
exp_kwargs["al_it"] = 1000
exp_kwargs["discr_model_frequency"] = 20
metrics_nashaat_1000, _, _, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)
exp_kwargs["al_it"] = al_it
exp_kwargs["discr_model_frequency"] = 1

import pickle
np.random.seed(25)
metrics_nashaat_1000 = pickle.load(open("metrics_nashaat_1000.pkl", "rb"))

# #### Active learning

# +
set_seed(76)

predict_dataset = CustomTensorDataset(X=torch.Tensor(df_occup[features].values), Y=torch.Tensor(y_train))
test_dataset = CustomTensorDataset(X=torch.Tensor(df1[features].values), Y=torch.Tensor(y_test))

al_exp_kwargs = dict(
    nr_trials=10,
    al_it=al_it,
    batch_size=batch_size,
    seeds = np.random.randint(0,1000,10),
    features = df_occup[features],
    list_features = features,
    y_train = y_train,
    y_test = y_test,
    train_dataset = CustomTensorDataset(X=df_occup.loc[[0],features], Y=y_train[0]),
    predict_dataloader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=df1[features].values
)
# -

from performance import PerformanceMixin
from sklearn.linear_model import LogisticRegression

PerformanceMixin().analyze(y_test, torch.full_like(y_test, 0))


def nonsynthetic_al_experiment(nr_trials, al_it, features, list_features, y_train, y_test, batch_size, seeds, train_dataset, predict_dataloader, test_dataloader, test_features):

    metric_dict = {}

    for j in tqdm(range(nr_trials), desc="Trials"):
        metric_dict[j] = {}
        metric_dict[j]["Discriminative_train"] = {}
        metric_dict[j]["Discriminative_test"] = {}
        queried = []
        
        set_seed(seeds[j])
        
        model = LogisticRegression(solver='liblinear')            

        for i in range(len(queried), al_it + 1):

            if (len(queried) < 2) or (len(np.unique(y_train[queried])) < 2):
                queried.append(random.sample(range(len(y_train)), 1)[0])
                metric_dict[j]["Discriminative_train"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
                metric_dict[j]["Discriminative_test"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0}
            else:      
                Y = y_train[queried]
                df_1 = features.iloc[queried]
                train_preds = model.fit(df_1[list_features].values, Y).predict_proba(features.values)
                    
                queried.append(np.argmin(np.abs(train_preds[:, 1] - train_preds[:, 0])).item())

                test_preds = model.predict_proba(test_features)

                metric_dict[j]["Discriminative_train"][i] = PerformanceMixin().analyze(y=y_train, preds=torch.Tensor(train_preds))
                metric_dict[j]["Discriminative_test"][i] = PerformanceMixin().analyze(y=y_test, preds=torch.Tensor(test_preds))

    return metric_dict


al_accuracies = nonsynthetic_al_experiment(**al_exp_kwargs)

pickle.dump(al_accuracies, open("al_accuracies_spam.pkl", "wb"))

# al_accuracies = pickle.load(open("spam_al_accuracies.pkl", "rb"))

# #### Process results

metrics_maxkl = pickle.load(open("../results/aw_spam_ss34_metrics_maxkl.pkl", "rb"))

metrics_nashaat_3 = pickle.load(open("../results/ns_spam_ss34_metrics_nashaat.pkl", "rb"))

al_accuracies = pickle.load(open("../results/al_accuracies_spam.pkl", "rb"))

metrics_nashaat_2 = pickle.load(open("../results/metrics_nashaat_spam.pkl", "rb"))

al_df = process_exp_dict(al_accuracies, "Active learning by itself")

al_df = al_df[al_df["Metric"] == "F1"]
al_df = al_df[al_df["Set"] == "test"]

al_df.loc[al_df["Value"] == 0.5, "Value"] = 0



metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
#                         process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(metrics_nashaat_3, "Nashaat et al."),
                        al_df])
metric_dfs = metric_dfs.reset_index(level=0).rename(columns={"level_0": "Run"})
metric_dfs["Dash"] = "n"

# +
joined_df = add_baseline(metric_dfs, al_it)

# optimal_generative_test = lm.analyze(y_test, lm.predict_true(y_train, y_test, label_matrix_test))
# optimal_discriminative_test = dm.analyze(y_test, test_preds)
# joined_df = add_optimal(joined_df, al_it, optimal_generative_test, optimal_discriminative_test)
# -

joined_df = joined_df[joined_df["Metric"] == "F1"]
joined_df = joined_df[joined_df["Set"] == "test"]


font_size=25
legend_size=25
tick_size=20
n_boot=10000
linewidth=4

# +
colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,1, figsize=(15,8), sharey=True)

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"], ax=axes)
axes.get_legend().remove()
axes.set_title("Discriminative model", fontsize=font_size)

handles, labels = axes.get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
leg = axes.legend(handles=handles, labels=labels, loc="lower right", title="Method", fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"

axes.tick_params(axis='both', which='major', labelsize=tick_size)

axes.set_xlabel("Number of active learning iterations", fontsize=font_size)
axes.set_ylabel("F1", fontsize=font_size)

plt.tight_layout()

plt.savefig("../plots/spam_performance_baselines.png")
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
metric_dfs = metric_dfs[metric_dfs["Metric"].isin(["F1"])]

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
axes[0].set_ylabel("F1", fontsize=font_size)

#plt.ylim(0.85,0.978)

plt.tight_layout()

plt.savefig("../plots/sampling_strategies.png")
# -

pickle.dump(metric_dfs, open("../outputs/metric_dfs.pkl", "wb"))

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

pickle.dump(entropies_df, open("../outputs/entropies_df.pkl", "wb"))

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
#plt.ylim(-0.05,1.8)

plt.tight_layout()
plt.savefig("../plots/entropies.png")
# -











































