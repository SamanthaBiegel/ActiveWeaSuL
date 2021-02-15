# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
DAP = True
    
if DAP:
# #     ! pip install -r ../requirements.txt
# #     ! aws s3 cp s3://user/gc03ye/uploads/glove /tmp/data/word_embeddings --recursive
# #     ! aws s3 cp s3://user/gc03ye/uploads/resnet_old.pth /tmp/models/resnet_old.pth
# #     ! aws s3 cp s3://user/gc03ye/uploads /tmp/data/visual_genome/VG_100K --recursive --exclude "glove/*" --exclude "resnet_old.pth" --exclude "resnet.pth" --exclude "siton_dataset.csv" --exclude "train.zip" --exclude "VRD*"
    path_prefix = "/tmp/"
    import torch
    pretrained_model = torch.load(path_prefix + "models/resnet_old.pth")
else:
    import torchvision.models as models
    pretrained_model = models.resnet18(pretrained=True)
    path_prefix = "../"
# -

torch.cuda.is_available()

# +
# %load_ext autoreload
# %autoreload 2

import ast
import csv
import json
import numpy as np
import random
import time
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import sys
import os

sys.path.append(os.path.abspath("../activelearning"))
from synthetic_data import SyntheticDataGenerator
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from plot import plot_probs, plot_train_loss
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
from lf_utils import apply_lfs, analyze_lfs
from visualrelation import VisualRelationDataset, VisualRelationClassifier, WordEmb, FlatConcat
# -

df_train = pd.read_csv("../data/visual_genome/VG_train.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})
df_test = pd.read_csv("../data/visual_genome/VG_test.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})

# +
# df_vis = pd.read_csv("../../../s3_home/uploads/siton_dataset.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})

# +
OTHER = 0
SITON = 1
# ABSTAIN = -1

def lf_siton_subject_object(x):
    if x.subject_category in ["person", "woman", "man", "child", "dog", "cat"]:
        if x.object_category in ["bench", "chair", "floor", "horse", "grass", "table", "sofa"]:
            return SITON
    return OTHER

def lf_not_person(x):
    if x.subject_category != "person":
        return OTHER
    return SITON

def lf_siton_object(x):
    if x.object_category in ["bench", "chair", "floor", "horse", "grass", "table", "sofa"]:
            return SITON
    return OTHER

def lf_ydist(x):
    if x.subject_y_max < x.object_y_max and x.subject_y < x.object_y:
        return SITON
    return OTHER

def lf_xdist(x):
    if (x.subject_x_max > x.object_x) and (x.subject_x_max < x.object_x_max): 
        return SITON
    return OTHER

def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) > 100:
        return SITON
    return OTHER

def lf_area(x):
    if (x.subject_w * x.subject_h) / (x.object_w * x.object_h) < 0.8:
        return SITON
    return OTHER

# lfs = [lf_siton_subject_object, lf_not_person, lf_siton_object, lf_area, lf_dist, lf_ydist, lf_xdist]
lfs = [lf_area, lf_dist, lf_siton_subject_object]

# cliques = [[0],[1],[2],[3],[4],[5],[6]]
# cliques=[[0],[1,2,3],[4]]
cliques=[[0],[1],[2]]
# -

L_train = apply_lfs(df_train, lfs)
L_test = apply_lfs(df_test, lfs)

y_train = df_train.y.values
y_test = df_test.y.values

analyze_lfs(L_train, y_train, lfs)

df_vis = pd.concat([df_train, df_test])

L = apply_lfs(df_vis, lfs)

# +
# import csv
# word_embs = pd.read_csv(
#             path_prefix + "data/word_embeddings/glove.6B.100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
#         ).T
# word_embs = list(word_embs.columns)

# valid_embeddings = (df_vis["channels"] == 3) & df_vis.object_category.isin(word_embs) & df_vis.subject_category.isin(word_embs) & ~df_vis["object_category"].str.contains(" ") & ~df_vis["subject_category"].str.contains(" ")

# df_vis_final = df_vis[valid_embeddings]
# df_vis_final.index = list(range(len(df_vis_final)))

# L_final = L[valid_embeddings]

df_vis_final = df_vis.copy()
L_final = L.copy()

# np.random.seed(254)
np.random.seed(98)
indices_shuffle = np.random.permutation(df_vis_final.shape[0])

split_nr = int(np.ceil(0.9*df_vis_final.shape[0]))
train_idx, test_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]

df_train = df_vis_final.iloc[train_idx]
df_test = df_vis_final.iloc[test_idx]
df_train.index = list(range(len(df_train)))
df_test.index = list(range(len(df_test)))

L_train = L_final[train_idx,:]
L_test = L_final[test_idx,:]

y_train = df_train.y.values
y_test = df_test.y.values

# +
# np.random.seed(485)
# indices_shuffle = np.random.permutation(df_train.shape[0]+df_test.shape[0])

# split_nr = int(df_train.shape[0])
# train_idx, test_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]

# +
# df_train.to_csv("../data/visual_genome/VG_train.csv", index=False)

# +
# df_test.to_csv("../data/visual_genome/VG_test.csv", index=False)
# -

L_train.sum(axis=0)

class_balance = np.array([1-df_test.y.mean(), df_test.y.mean()])

# +
set_seed(243)
lm = LabelModel(n_epochs=200,
                    lr=1e-1)

Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
lm.analyze(y_train)
# -

plt.plot(range(0,200), lm.losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

true_probs = lm.predict_true(y_train)

lm.analyze(y_train, true_probs)

feature_tensor_train = torch.load("../data/visual_genome/train_embeddings.pt")
feature_tensor_test = torch.load("../data/visual_genome/test_embeddings.pt")

feature_tensor = torch.cat([feature_tensor_train, feature_tensor_test])

feature_tensor_train = feature_tensor[train_idx,:]
feature_tensor_test = feature_tensor[test_idx,:]

dataset_train = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train))
dataset_predict = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train))
dataset_test = CustomTensorDataset(feature_tensor_test, torch.Tensor(y_test))

# +
# subset_points = random.sample(range(len(df_train)), 30)
# df_train_subset=df_train.iloc[subset_points]
# df_train_subset.index = range(len(df_train_subset))
batch_size=64

dataset_train.Y = true_probs#.clamp(0,1)

dl_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
dl_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

final_model=VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix, soft_labels=True)

final_model.reset()
final_model.fit(dl_train)
preds_test = final_model.predict(dl_test)
preds_train = final_model.predict()

# +
# plot_train_loss(final_model.average_losses)
# -

# final_model.analyze(df_train.y.values, preds_train)
final_model.analyze(df_test.y.values, preds_test)

dataset_train = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train))
dataset_predict = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train))
dataset_test = CustomTensorDataset(feature_tensor_test, torch.Tensor(y_test))

exp_kwargs = dict(nr_trials=10,
                  al_it=50,
                  label_matrix=L_train,
                  y_train=y_train,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=243, 
                  penalty_strength=1e3, 
                  batch_size=32,
                  discr_model_frequency=5,
                  final_model=VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix),
                  train_dataset=dataset_predict,
                  test_dataset=dataset_test,
                  label_matrix_test=L_test,
                  y_test=y_test
                  )

np.random.seed(50)
exp_kwargs["seeds"] = np.random.randint(0,1000,10)
metrics_maxkl, queried_maxkl, probs_maxkl, entropies_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

plot_metrics(process_exp_dict(metrics_maxkl, "f"), plot_train=True)

import pickle
with open("paper_results/vg_maxkl.pkl", "wb") as f:
    pickle.dump(metrics_maxkl, f)

np.random.seed(308)
exp_kwargs["seeds"] = np.random.randint(0,1000,10)
metrics_margin, queried_margin, probs_margin = active_weasul_experiment(**exp_kwargs, query_strategy="margin")

with open("paper_results/vg_margin.pkl", "wb") as f:
    pickle.dump(metrics_margin, f)

np.random.seed(154)
exp_kwargs["seeds"] = np.random.randint(0,1000,10)
metrics_nashaat, queried_nashaat, probs_nashaat = active_weasul_experiment(**exp_kwargs, query_strategy="nashaat")

with open("paper_results/vg_nashaat.pkl", "wb") as f:
    pickle.dump(metrics_nashaat, f)

# +
maxkl_df = process_exp_dict(metrics_maxkl, "Active WeaSuL").reset_index(level=0).rename(columns={"level_0": "Run"})
nashaat_df = process_exp_dict(metrics_nashaat, "Nashaat et al.").reset_index(level=0).rename(columns={"level_0": "Run"})
maxkl_df["Dash"] = "n"
nashaat_df["Dash"] = "n"

joined_df = pd.concat([maxkl_df, nashaat_df, accuracy_df])

# +
joined_df = joined_df[joined_df["Metric"] == "Accuracy"]
joined_df = joined_df[joined_df["Set"] == "test"]

colors = ["#000000", "#2b4162", "#368f8b", "#ec7357", "#e9c46a"]

sns.set(style="whitegrid", palette=sns.color_palette(colors))

fig, axes = plt.subplots(1,2, figsize=(16,8), sharey=True)

sns.lineplot(data=joined_df[joined_df["Model"] == "Generative"], x="Number of labeled points", y="Value",
            hue="Approach", ci=68, n_boot=100, estimator="mean", style="Dash",
            hue_order=["*","Active WeaSuL", "Nashaat et al.", "Weak Supervision"], ax=axes[0])

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles[2:], labels=labels[2:5], loc="lower right")
axes[0].title.set_text("Generative")

sns.lineplot(data=joined_df[joined_df["Model"] == "Discriminative"], x="Number of labeled points", y="Value",
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

import pickle
with open("paper_results/vg_maxkl_3.pkl", "wb") as f:
    pickle.dump(metrics_maxkl, f)

# +
final_model_kwargs = dict(lr=1e-3,
                          n_epochs=3)

set_seed(97)

batch_size = 32

features = dataset_predict.X.clone()
dataset_train.Y = y_train

al_exp_kwargs = dict(
    nr_trials=10,
    al_it=100,
    model=VisualRelationClassifier(pretrained_model, **final_model_kwargs, data_path_prefix=path_prefix, soft_labels=False),
    batch_size=batch_size,
    seeds = np.random.randint(0,1000,10),
    features = features,
    y_train = df_train.y.values,
    y_test = df_test.y.values,
    train_dataset = dataset_train,
    predict_dataloader = torch.utils.data.DataLoader(dataset=dataset_predict, batch_size=256, shuffle=False),
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle=False),
)
# -

al_accuracies = active_learning_experiment(**al_exp_kwargs)

# +
accuracy_df = pd.DataFrame.from_dict(al_accuracies[0])
accuracy_df = accuracy_df.stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Value"})

accuracy_df["Metric"] = "Accuracy"
accuracy_df["Approach"] = "Active Learning"
accuracy_df["Model"] = "Discriminative"
accuracy_df["Set"] = "test"
accuracy_df["Dash"] = "n"


# -

def plot_metrics_separate(metric_df, filter_metrics=["Accuracy"], plot_train=False):

    if not plot_train:
        metric_df = metric_df[metric_df.Set != "train"]

    lines = list(metric_df.Run.unique())

    colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"][:len(lines)]

    metric_df = metric_df[metric_df["Metric"].isin(filter_metrics)]

    sns.set(style="whitegrid")
    ax = sns.relplot(data=metric_df, x="Number of labeled points", y="Value", col="Model",
                     kind="line", hue="Run", estimator=None, legend=False)

    show_handles = [ax.axes[0][0].lines[i] for i in range(len(lines))]
    show_labels = lines
    ax.axes[len(ax.axes)-1][len(ax.axes[0])-1].legend(handles=show_handles, labels=show_labels, loc="lower right")

    ax.set_ylabels("")
    ax.set_titles("{col_name}")


plot_metrics_separate(accuracy_df)

plot_metrics(accuracy_df)







