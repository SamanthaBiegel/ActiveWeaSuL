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
import random
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
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment
# -

# ### Create data

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
p_z = 0.5

set_seed(932)
data = SyntheticDataGenerator(N, p_z, centroids)
df = data.sample_dataset().create_df()
# -

N = 3000
set_seed(466)
data = SyntheticDataGenerator(N, p_z, centroids)
df_test = data.sample_dataset().create_df()

# ### Define labeling functions

# +
df.loc[:, "wl1"] = (df.x2<0.4)*1
df.loc[:, "wl2"] = (df.x1<-0.3)*1
df.loc[:, "wl3"] = (df.x1<-1)*1

label_matrix = df[["wl1", "wl2", "wl3"]].values

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

class_balance = np.array([0.5, 0.5])
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
Y_probs_test = lm.predict(label_matrix_test, lm.mu, 0.5)

# Analyze test set performance
lm.analyze(df_test.y.values, Y_probs_test)

# +
# set_seed(243)

# final_model_kwargs["lr"] = 1e-3
# # final_model_kwargs["n_epochs"] = 100

# train_dataset = SyntheticDataset(df=df, Y=Y_probs.detach())

# test_dataset = SyntheticDataset(df=df_test, Y=df_test.y.values)
# test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# dm = LogisticRegression(**final_model_kwargs)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

# dm.reset()
# train_preds = dm.fit(train_loader).predict()
# test_preds = dm.predict(test_dataloader)
# dm.analyze(df_test.y.values, test_preds)

# +
# plot_train_loss(dm.average_losses)

# +
# it = 30
# # Choose strategy from ["maxkl", "margin", "nashaat"]
# query_strategy = "maxkl"

# seed = 631

# al = ActiveWeaSuLPipeline(it=it,
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
#                                   y_train=df.y.values,
#                                   cliques=cliques,
#                                   class_balance=class_balance,
#                                   train_dataset = SyntheticDataset(df=df, Y=Y_probs.detach()))

# +
# plot_metrics(process_metric_dict(al.metrics, "MaxKL", remove_test=True))

# +
# plot_it=5
# plot_probs(df, al.probs["Generative_train"][plot_it], soft_labels=False, add_labeled_points=al.queried[:plot_it])
# -

# ### Run Active WeaSuL

exp_kwargs = dict(nr_trials=1,
                  al_it=1000,
                  label_matrix=label_matrix,
                  y_train=df.y.values,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=243,
                  penalty_strength=1,
                  batch_size=256,
                  final_model=LogisticRegression(**final_model_kwargs, hide_progress_bar=True),
                  discr_model_frequency=10,
                  train_dataset = SyntheticDataset(df=df, Y=Y_probs.detach()),
                  test_dataset = SyntheticDataset(df=df_test, Y=Y_probs_test.detach()),
                  label_matrix_test=label_matrix_test,
                  y_test=df_test.y.values)

np.random.seed(284)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl, queried_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

np.random.seed(568)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_margin, queried_margin = active_weasul_experiment(**exp_kwargs, query_strategy="margin")

np.random.seed(568)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_random, queried_random = active_weasul_experiment(**exp_kwargs, query_strategy="margin", randomness=1)

np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat, queried_nashaat = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat", randomness=0)

plot_metrics(process_exp_dict(metrics_nashaat, "Nashaat et al."))
# plt.savefig("paper_plots/nashaat.png")

# +
def active_learning_experiment(nr_trials, al_it, model, features, y_train, y_test, batch_size, seeds, train_dataset, predict_dataloader, test_dataloader, test_features):

    accuracy_dict = {}

    for j in tqdm(range(nr_trials), desc="Trials"):
        accuracies = []
        queried = []
        
        model = LogisticRegression()

        set_seed(seeds[j])
        
        while (len(queried) < 2) or (len(np.unique(y_train[queried])) < 2):
            queried.append(random.sample(range(len(y_train)), 1)[0])
            accuracies.append(0.5)

#         queried = list(random.sample(range(len(y_train)), 10))
        for i in range(len(queried), al_it + 1):
        
            Y = y_train[queried]
            df_1 = features.iloc[queried]
#             train_dataset.update(df_1, Y)
#             train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            train_preds = model.fit(df_1.loc[:, ["x1", "x2"]].values, Y).predict_proba(features.values)
#             if i == 3:
#                 df = features.copy()
#                 df["y"] = y_train
#                 plot_probs(df, train_preds, add_labeled_points=queried).show()
                
            queried.append(np.argmin(np.abs(train_preds[:, 1] - train_preds[:, 0])).item())

            test_preds = model.predict_proba(test_features)
            
            

            accuracies.append(model.score(test_features, y_test))

        accuracy_dict[j] = accuracies

    return accuracy_dict
# -



# +
from sklearn.linear_model import LogisticRegression

final_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-1,
                          n_epochs=100)

set_seed(76)

predict_dataset = SyntheticDataset(df=df, Y=df.y.values)
test_dataset = SyntheticDataset(df=df_test, Y=df_test.y.values)

batch_size=256

al_exp_kwargs = dict(
    nr_trials=10,
    al_it=30,
    model=LogisticRegression(),
#     model=LogisticRegression(**final_model_kwargs, soft_labels=False, hide_progress_bar=True),
    batch_size=batch_size,
    seeds = np.random.randint(0,1000,10),
    features = df.loc[:, ["x1", "x2"]],
    y_train = df.y.values,
    y_test = df_test.y.values,
    train_dataset = SyntheticDataset(df=df.iloc[[0]], Y=df.y.values[0]),
    predict_dataloader = torch.utils.data.DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=df_test.loc[:, ["x1", "x2"]].values
)
# -

al_accuracies = active_learning_experiment(**al_exp_kwargs)

# +
accuracy_df = pd.DataFrame.from_dict(al_accuracies)
accuracy_df = accuracy_df.stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Accuracy"})

accuracy_df["Metric"] = "Accuracy"
accuracy_df["Approach"] = "Active Learning"
accuracy_df["Model"] = "Discriminative"
accuracy_df["Set"] = "test"
accuracy_df["Dash"] = "n"
# -

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "MaxKL"),
                        process_exp_dict(metrics_margin, "Margin"),
                        process_exp_dict(metrics_random, "Random")])

fig = plot_metrics(metric_dfs)
# plt.savefig("paper_plots/strategies.png")

plot_metrics(accuracy_df)
# plt.savefig("paper_plots/active_learning.png")

accuracy_df

joined_df = pd.read_csv("results/combined_baselines.csv")
joined_df["Set"] = "test"

joined_df = joined_df[joined_df["Approach"] != "Active Learning"]

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al.")])
metric_dfs = metric_dfs.reset_index(level=0).rename(columns={"level_0": "Run"})
metric_dfs["Dash"] = "n"

# +
WS_baseline = metric_dfs[(metric_dfs["Number of labeled points"] == 0)
                         & (metric_dfs["Strategy"] == "Active WeaSuL")
                         & (metric_dfs["Metric"] == "Accuracy")
                         & (metric_dfs["Set"] == "test")].groupby("Model").mean()["Value"]

baseline_fm = pd.DataFrame({"Number of labeled points": list(range(31)), "Value": np.repeat(WS_baseline["Discriminative"], 31)})
baseline_fm["Model"] = "Discriminative"
baseline_fm["Run"] = 0
baseline_fm["Strategy"] = "Weak Supervision"
baseline_fm["Metric"] = "Accuracy"
baseline_fm["Dash"] = "n"
baseline_fm["Set"] = "test"

baseline_lm = pd.DataFrame({"Number of labeled points": list(range(31)), "Value": np.repeat(WS_baseline["Generative"], 31)})
baseline_lm["Model"] = "Generative"
baseline_lm["Run"] = 0
baseline_lm["Strategy"] = "Weak Supervision"
baseline_lm["Metric"] = "Accuracy"
baseline_lm["Dash"] = "n"
baseline_lm["Set"] = "test"

joined_df = pd.concat([metric_dfs, accuracy_df, baseline_lm, baseline_fm])
# -

joined_df = pd.concat([joined_df, accuracy_df])

joined_df

joined_df = joined_df[joined_df["Metric"] == "Accuracy"]
joined_df = joined_df[joined_df["Set"] == "test"]

# +
colors = ["#000000", "#2b4162", "#368f8b", "#ec7357", "#e9c46a"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,2, figsize=(16,8), sharey=True)

sns.lineplot(data=joined_df[joined_df["Model"] == "Generative"], x="Number of labeled points", y="Accuracy",
            hue="Approach", ci=68, n_boot=100, estimator="mean", style="Dash",
            hue_order=["*","Active WeaSuL", "Nashaat et al.", "Weak Supervision"], ax=axes[0])

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles[2:], labels=labels[2:5], loc="lower right")
axes[0].title.set_text("Generative")

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Accuracy",
            hue="Approach", ci=68, n_boot=100, estimator="mean", style="Dash",
            hue_order=["*","Active WeaSuL", "Nashaat et al.", "Weak Supervision", "Active Learning"], ax=axes[1])

handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles[2:], labels=labels[2:6], loc="lower right")
axes[1].title.set_text("Discriminative")

plt.ylabel("Accuracy")

plt.tight_layout()

# plt.savefig("paper_plots/performance_baselines-2.png")
# plt.show()
# -






