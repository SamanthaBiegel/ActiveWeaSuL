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
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

sys.path.append(os.path.abspath("../activeweasul"))
from synthetic_data import SyntheticDataGenerator
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment, add_baseline
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
from lf_utils import apply_lfs, analyze_lfs
from visualrelation import VisualRelationDataset, VisualRelationClassifier

# -

pretrained_model = models.resnet18(pretrained=True)
path_prefix = "../data/VRD/"

# ### Load data

# +
balance=False
semantic_predicates = [
        "carry",
        "cover",
        "fly",
        "look",
        "lying on",
        "park on",
        "sit on",
        "stand on",
        "ride",
    ]

classify = ["sit on"]
df_train, df_test = load_vr_data(classify=classify, include_predicates=semantic_predicates, path_prefix=path_prefix, drop_duplicates=True, balance=balance, validation=False)

y_train = df_train.y.values
y_test = df_test.y.values

print("Train Relationships: ", len(df_train))
print("Test Relationships: ", len(df_test))

# +
dataset_train = VisualRelationDataset(image_dir=path_prefix + "/images/train_images", df=df_train, Y=y_train)

dl_train = DataLoader(dataset_train, shuffle=False, batch_size=256)

final_model = VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix="../data/", soft_labels=False)

feature_tensor_train = torch.Tensor([])

for batch_features, batch_labels in tqdm(dl_train):
    feature_tensor_train = torch.cat((feature_tensor_train, final_model.extract_concat_features(batch_features).to("cpu")))

# +
dataset_test = VisualRelationDataset(image_dir=path_prefix + "images/test_images", 
                      df=df_test,
                      Y=y_test)

dl_test = DataLoader(dataset_test, shuffle=False, batch_size=256)

feature_tensor_test = torch.Tensor([])

for batch_features, batch_labels in dl_test:
    feature_tensor_test = torch.cat((feature_tensor_test, final_model.extract_concat_features(batch_features).to("cpu")))
# -

# ### Apply labeling functions

SITON = 1
OTHER = 0


# +
def lf_siton_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bench", "chair", "floor", "horse", "grass", "table"]:
            return SITON
    return OTHER

YMIN = 0
YMAX = 1
XMIN = 2
XMAX = 3

def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) >= 100:
        return OTHER
    return SITON

def area(bbox):
    return (bbox[YMAX] - bbox[YMIN]) * (bbox[XMAX] - bbox[XMIN])

def lf_area(x):
    if area(x.subject_bbox) / area(x.object_bbox) < 0.8:
        return SITON
    return OTHER


# +
lfs = [lf_siton_object, lf_dist, lf_area]

label_matrix = apply_lfs(df_train, lfs)
label_matrix_test = apply_lfs(df_test, lfs)
# -

analyze_lfs(label_matrix_test, df_test["y"], lfs)


# ### Experiments

starting_seed = 36
penalty_strength = 1
nr_trials = 10
al_it = 250

# +
final_model_kwargs = dict(lr=1e-3,
                          n_epochs=100)

batch_size = 20

class_balance = np.array([1-df_train.y.mean(), df_train.y.mean()])

cliques=[[0],[1,2]]
# -

# Plotting settings
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
                  final_model=VisualRelationClassifier(pretrained_model, **final_model_kwargs, data_path_prefix="../data/", patience=5, warm_start=False, early_stopping=True),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train)),
                  test_dataset = CustomTensorDataset(feature_tensor_test, torch.Tensor(y_test)),
                  label_matrix_test=label_matrix_test,
                  y_test=y_test)

np.random.seed(50)
exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_maxkl, queried_maxkl, probs_maxkl, entropies_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

# #### Nashaat et al.

np.random.seed(25)
exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_nashaat, queried_nashaat, probs_nashaat, _ = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat")

# #### Active learning

# +
set_seed(76)

train_dataset = CustomTensorDataset(feature_tensor_train[0,:], torch.Tensor(y_train[0]))
predict_dataset = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train))
test_dataset = CustomTensorDataset(feature_tensor_test, torch.Tensor(y_test))

final_model_kwargs["n_epochs"] = 2

al_exp_kwargs = dict(
    nr_trials=nr_trials,
    al_it=al_it,
    batch_size=batch_size,
    seeds = np.random.randint(0,1000,nr_trials),
    model = VisualRelationClassifier(pretrained_model, **final_model_kwargs, early_stopping=False, data_path_prefix="../data/", soft_labels=False),
    features = feature_tensor_train,
    y_train = y_train,
    y_test = y_test,
    train_dataset = train_dataset,
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=feature_tensor_test
)

final_model_kwargs["n_epochs"] = 100
# -

metrics_activelearning = active_learning_experiment(**al_exp_kwargs)

# #### Process results

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(metrics_activelearning, "Active learning by itself")]).reset_index(level=0).rename(columns={"level_0": "Run"})
joined_df = add_baseline(metric_dfs, al_it)
joined_df = joined_df[joined_df["Metric"] == "F1"]
joined_df = joined_df[joined_df["Set"] == "test"]

# +
colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,1, figsize=(15,8), sharey=True)

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"])
axes.set_title("Discriminative model", fontsize=font_size)

handles, labels = axes.get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
leg = axes.legend(handles=handles, labels=labels[0:6], loc="lower right", title="Method",
                     fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"

axes.tick_params(axis='both', which='major', labelsize=tick_size)

axes.set_xlabel("Number of active learning iterations", fontsize=font_size)
axes.set_ylabel("F1", fontsize=font_size)

plt.tight_layout()

plt.show()
# -

# ## Figure 5A

# #### Other sampling strategies

exp_kwargs["al_it"] = 50

np.random.seed(70)
exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_margin, queried_margin, probs_margin, entropies_margin = active_weasul_experiment(**exp_kwargs, query_strategy="margin")

np.random.seed(60)
exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_discr, queried_discr, probs_discr, entropies_discr = active_weasul_experiment(**exp_kwargs, query_strategy="discriminative")

np.random.seed(70)
exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_random, queried_random, probs_random, entropies_random = active_weasul_experiment(**exp_kwargs, query_strategy="margin", randomness=1)

# #### Process results

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "MaxKL"),
                        process_exp_dict(metrics_margin, "Margin"),
                        process_exp_dict(metrics_random, "Random")])

# +
metric_dfs = metric_dfs[metric_dfs.Set == "test"]
metric_dfs = metric_dfs[metric_dfs["Metric"].isin(["F1"])]
metric_dfs = metric_dfs[metric_dfs["Number of labeled points"] < 51]

lines = list(metric_dfs.Approach.unique())

colors = ["#2b4162", "#CC7178", "#598B2C", "#e9c46a"][:len(lines)]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,2, figsize=(15,8), sharey=True)

sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Generative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth,
            hue_order=["MaxKL",  "Margin", "Random"], ax=axes[0])

handles, labels = axes[0].get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
axes[0].legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)
axes[0].set_title("Generative model", fontsize=font_size)

sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth, legend=False,
            hue_order=["MaxKL", "Margin", "Random"], ax=axes[1])

axes[1].set_title("Discriminative model", fontsize=font_size)

axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)

axes[0].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[1].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("F1", fontsize=font_size)

plt.tight_layout()

plt.show()
# -

# ### Figure 5B

# #### Process entropies

# +
maxkl_entropies_df = pd.DataFrame.from_dict(entropies_maxkl).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
maxkl_entropies_df["Approach"] = "MaxKL"

margin_gen_entropies_df = pd.DataFrame.from_dict(entropies_margin).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
margin_gen_entropies_df["Approach"] = "Margin"

random_entropies_df = pd.DataFrame.from_dict(entropies_random).stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Entropy"})
random_entropies_df["Approach"] = "Random"
# -

entropies_df = pd.concat([maxkl_entropies_df, margin_gen_entropies_df, random_entropies_df])
entropies_df["Number of labeled points"] = entropies_df["Number of labeled points"].apply(lambda x: x+1)
entropies_df = entropies_df[entropies_df["Number of labeled points"] < 51]

# +
lines = list(entropies_df.Approach.unique())

# colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"][:len(lines)]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

plt.subplots(1,1,figsize=(8,8))
ax = sns.lineplot(data=entropies_df, x="Number of labeled points", y="Entropy", hue="Approach", ci=68, n_boot=n_boot,
                  legend=False,linewidth=linewidth, hue_order=["MaxKL", "Margin","Random"])
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.set_xlabel("Number of active learning iterations", fontsize=font_size)
ax.set_ylabel("Diversity (entropy)", fontsize=font_size)
ax.set_title("Diversity of sampled buckets", fontsize=font_size)

plt.tight_layout()

plt.show()
# -






