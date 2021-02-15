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

sys.path.append(os.path.abspath("../activelearning"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from plot import plot_probs, plot_train_loss
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment, bucket_entropy_experiment, add_baseline, synthetic_al_experiment
# -

# ### Load data

# +
path_prefix = "../data/occupancy/"

L_train = np.load(path_prefix + "L_train_100221.npy")
L1 = np.load(path_prefix + "L1_100221.npy")
L2 = np.load(path_prefix + "L2_100221.npy")
df_occup = pd.read_csv(path_prefix + "df_occup_100221.csv")
df1 = pd.read_csv(path_prefix + "df1_100221.csv")
df2 = pd.read_csv(path_prefix + "df2_100221.csv")

y_train = df_occup["Occupancy"].values
y1 = df1["Occupancy"].values
y2 = df2["Occupancy"].values
# -

L_train.shape, L1.shape, L2.shape

label_matrix = L_train
label_matrix_test = L1
y_test = y1

# ### Fit label model

# +
p_z = 0.21

class_balance = np.array([1 - p_z, p_z])
cliques=[[0],[1],[2],[3],[4]]

# +
set_seed(243)

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

# ### Fit discriminative model

# +
batch_size = 256

set_seed(27)

features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

train_dataset = CustomTensorDataset(
    X=torch.Tensor(df_occup[features].values), 
    Y=lm.predict_true(y_train).detach())
test_dataset = CustomTensorDataset(
    X=torch.Tensor(df1[features].values), 
    Y=Y_probs_test.detach())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

final_model_kwargs = dict(input_dim=5,
                          output_dim=2,
                          lr=1e-2,
                          n_epochs=100)

dm = LogisticRegression(**final_model_kwargs)

dm.reset()
train_preds = dm.fit(train_loader).predict()
test_preds = dm.predict(test_dataloader)
# -

# ### Figure 1

starting_seed = 243
penalty_strength = 1e4
nr_trials = 10
al_it = 50

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
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_occup[features].values), Y=Y_probs.detach()),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df1[features].values), Y=Y_probs_test.detach()),
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

# import pickle
# np.random.seed(25)
# metrics_nashaat_1000 = pickle.load(open("metrics_nashaat_1000.pkl", "rb"))

# #### Active learning

# +
set_seed(76)

predict_dataset = CustomTensorDataset(X=torch.Tensor(df_occup[features].values), Y=torch.Tensor(y_train))
test_dataset = CustomTensorDataset(X=torch.Tensor(df1[features].values), Y=torch.Tensor(y_test))

al_exp_kwargs = dict(
    nr_trials=nr_trials,
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


def nonsynthetic_al_experiment(nr_trials, al_it, features, list_features, y_train, y_test, batch_size, seeds, train_dataset, predict_dataloader, test_dataloader, test_features):

    metric_dict = {}

    for j in tqdm(range(nr_trials), desc="Trials"):
        metric_dict[j] = {}
        metric_dict[j]["Discriminative_train"] = {}
        metric_dict[j]["Discriminative_test"] = {}
        queried = []
        
        model = LogisticRegression(solver='liblinear')

        set_seed(seeds[j])            

        for i in range(len(queried), al_it + 1):

            if (len(queried) < 2) or (len(np.unique(y_train[queried])) < 2):
                queried.append(random.sample(range(len(y_train)), 1)[0])
                metric_dict[j]["Discriminative_train"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
                metric_dict[j]["Discriminative_test"][i] = {"MCC": 0, "Precision": 0.5, "Recall": 0.5, "Accuracy": 0.5, "F1": 0.5}
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

# #### Process results

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(al_accuracies, "Active learning by itself")])
metric_dfs = metric_dfs.reset_index(level=0).rename(columns={"level_0": "Run"})
metric_dfs["Dash"] = "n"


def add_optimal(metric_dfs, al_it, optimal_generative_test, optimal_discriminative_test):

    optimal_lm = pd.DataFrame(optimal_generative_test, index=range(al_it+1)).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Metric", 0: "Value"})
    optimal_lm["Run"] = 0
    optimal_lm["Model"] = "Generative"
    optimal_lm["Approach"] = "Upper bound"
    optimal_lm["Dash"] = "y"
    optimal_lm["Set"] = "test"

    optimal_dm = pd.DataFrame(optimal_discriminative_test, index=range(al_it+1)).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Metric", 0: "Value"})
    optimal_dm["Run"] = 0
    optimal_dm["Model"] = "Discriminative"
    optimal_dm["Approach"] = "Upper bound"
    optimal_dm["Dash"] = "y"
    optimal_dm["Set"] = "test"

    return pd.concat([metric_dfs, optimal_lm, optimal_dm])


# +
joined_df = add_baseline(metric_dfs, al_it)

optimal_generative_test = lm.analyze(y_test, lm.predict_true(y_train, y_test, label_matrix_test))
optimal_discriminative_test = dm.analyze(y_test, test_preds)
joined_df = add_optimal(joined_df, al_it, optimal_generative_test, optimal_discriminative_test)
# -

joined_df = joined_df[joined_df["Metric"] == "MCC"]
joined_df = joined_df[joined_df["Set"] == "test"]

nashaat_df = process_exp_dict(metrics_nashaat_1000, "Nashaat et al.")
nashaat_df = nashaat_df[nashaat_df["Metric"] == "MCC"]
nashaat_df = nashaat_df[nashaat_df["Set"] == "test"]

font_size=25
legend_size=25
tick_size=20
n_boot=10000

pickle.dump(joined_df, open("../outputs/joined_df.pkl", "wb"))
pickle.dump(nashaat_df, open("../outputs/nashaat_df.pkl", "wb"))

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
axes[0].set_title("Generative model (20 iterations)", size=font_size)

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", style="Dash",
            hue_order=["Upper bound","Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"], ax=axes[1])
axes[1].legend([],[])
axes[1].set_title("Discriminative model (20 iterations)", fontsize=font_size)

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
axes[0].set_ylabel("MCC", fontsize=font_size)

plt.ylim(0.0, 1.00)

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
metric_dfs = metric_dfs[metric_dfs["Metric"].isin(["MCC"])]

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
axes[0].set_ylabel("MCC", fontsize=font_size)

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















