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
    # ! pip install -r ../requirements.txt
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

# !nvcc --version

torch.cuda.is_available()

# +
# %load_ext autoreload
# %autoreload 2

import json
import numpy as np
import random
import time
import pandas as pd

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
from synthetic_data import SyntheticDataGenerator, SyntheticDataset
from experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict, active_learning_experiment
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed
from plot import plot_probs, plot_train_loss
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
from lf_utils import apply_lfs, analyze_lfs
from visualrelation import VisualRelationDataset, VisualRelationClassifier, WordEmb, FlatConcat

# +
# with open(path_prefix + '/data/visual_genome/relationships.json') as f:
#   visgen_rels = json.load(f)
# -

pred_list = ['carrying',
 'covered in',
 'covering',
 'eating',
 'flying in',
 'growing on',
 'hanging from',
 'lying on',
 'mounted on',
 'painted on',
 'parked on',
 'playing',
 'riding',
 'says',
 'sitting on',
 'standing on',
 'using',
 'walking in',
 'walking on',
 'watching',
 'wearing']

# +
# visgen_df = pd.json_normalize(visgen_rels, record_path=["relationships"], meta="image_id", sep="_")
# visgen_df["predicate"] = visgen_df["predicate"].str.lower()
# visgen_df_actions = visgen_df[visgen_df["predicate"].isin(pred_list)]
# visgen_df_actions.to_csv(path_prefix + "data/action_dataset.csv", index=False)

# +
# pred_action = "sitting on"

# +
# visgen_df_actions = pd.read_csv(path_prefix + "data/action_dataset.csv")

# +
# visgen_df_actions["y"] = visgen_df_actions["predicate"]
# visgen_df_actions["y"] = visgen_df_actions["y"].apply(lambda x: 1 if x == pred_action else 0)
# df_vis = visgen_df_actions.loc[:,["image_id", "predicate", "object_name", "object_h", "object_w", "object_y", "object_x", "subject_name", "subject_h", "subject_w", "subject_y", "subject_x", "y"]]
# df_vis = df_vis.dropna()
# df_vis = df_drop_duplicates(df_vis)
# # df_vis = balance_dataset(df_vis)

# +
# df_vis["object_x_max"] = df_vis["object_x"] + df_vis["object_w"]
# df_vis["object_y_max"] = df_vis["object_y"] + df_vis["object_h"]
# df_vis["subject_x_max"] = df_vis["subject_x"] + df_vis["subject_w"]
# df_vis["subject_y_max"] = df_vis["subject_y"] + df_vis["subject_h"]

# df_vis["object_bbox"] = tuple(df_vis[["object_y", "object_y_max", "object_x", "object_x_max"]].values)
# df_vis["subject_bbox"] = tuple(df_vis[["subject_y", "subject_y_max", "subject_x", "subject_x_max"]].values)

# df_vis = df_vis.rename(columns={"object_name": "object_category", "subject_name": "subject_category", "image_id": "source_img"})

# df_vis.source_img = df_vis.source_img.astype(str) + ".jpg"

# +
# from PIL import Image
# df_vis["channels"] = df_vis["source_img"].apply(lambda x: len(np.array(Image.open(path_prefix + "data/visual_genome/VG_100K" + "/" + x)).shape))

# +
# df_vis.to_csv(path_prefix + "data/siton_dataset.csv", index=False)

# +
import ast
df_vis = pd.read_csv("../../../s3_home/uploads/siton_dataset.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})

# df_vis = pd.read_csv("../data/siton_dataset.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})

# +
# all_img = list(df_vis.source_img.drop_duplicates())

# +
# [img for img in all_img if img not in files]

# +
# with open("image_files.txt", "r") as f:
#     files = f.readlines()
#     image_files = [file.strip() for file in files]
    
# subset_images = list(df_vis["source_img"].drop_duplicates())
# missing_images = [image for image in subset_images if image not in image_files]
# len(missing_images)

# +
# for image in missing_images:
# #     ! cp ../data/visual_genome/VG_100K/$image ../data/visual_genome/missing_VG/$image

# +
# predicate_counts = visgen_df.groupby("predicate")["image_id"].count().sort_values(ascending=False)
# predicate_counts[predicate_counts > 1000]

# +
# pd.set_option('display.max_rows',102)
# pd.DataFrame(df_train.groupby("y")["source_img"].count())
# -

OTHER = 0

# +
# WEAR = 1

# def lf_wear_object(x):
#     if x.subject_name == "person":
#         if x.object_name in ["t-shirt", "jeans", "glasses", "skirt", "pants", "shorts", "dress", "shoes"]:
#             return WEAR
#     return OTHER

# def lf_area(x):
#     if (x.subject_w * x.subject_h) / (x.object_w * x.object_h) > 1:
#         return WEAR
#     return OTHER

# def lf_dist(x):
#     if ((x.subject_x - x.object_x) + (x.subject_y - x.object_y)) > 10:
#         return OTHER
#     return WEAR

# def lf_ydist(x):
#     if x.subject_y_max > x.object_y_max and x.subject_y < x.object_y:
#         return WEAR
#     return OTHER

# lfs = [lf_wear_object, lf_dist, lf_area]

# cliques=[[0],[1,2]]

# +
SITON = 1

def lf_siton_object(x):
    if x.subject_category in ["person", "woman", "man", "child", "dog", "cat"]:
        if x.object_category in ["bench", "chair", "floor", "horse", "grass", "table", "sofa"]:
            return SITON
    return OTHER

def lf_not_person(x):
    if x.subject_category != "person":
        return OTHER
    return SITON

def lf_ydist(x):
    if x.subject_y_max < x.object_y_max and x.subject_y < x.object_y:
        return SITON
    return OTHER

def lf_xdist(x):
    if x.subject_x_max < x.object_x or x.subject_x > x.object_x_max: 
        return OTHER
    return SITON

def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) > 100:
        return SITON
    return OTHER

def lf_area(x):
    if (x.subject_w * x.subject_h) / (x.object_w * x.object_h) < 0.8:
        return SITON
    return OTHER

lfs = [lf_siton_object, lf_area, lf_dist]

cliques = [[0],[1,2]]
# cliques=[[0],[1,2,3],[4]]
# -

L = apply_lfs(df_vis, lfs)

analyze_lfs(L, df_vis["y"], lfs)

class_balance = np.array([1-df_vis.y.mean(), df_vis.y.mean()])

# +
lm = LabelModel(n_epochs=200,
                    lr=1e-1)

Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
lm.analyze(df_vis.y.values)
# -


train_on = "probs" # probs or labels
n_epochs = 3
lr = 1e-3
batch_size=20

import csv
word_embs = pd.read_csv(
            path_prefix + "data/word_embeddings/glove.6B.100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        ).T
word_embs = list(word_embs.columns)

# +
# n_epochs = 3
# lr=1e-2
# batch_size = 64

valid_embeddings = (df_vis["channels"] == 3) & df_vis.object_category.isin(word_embs) & df_vis.subject_category.isin(word_embs) & ~df_vis["object_category"].str.contains(" ") & ~df_vis["subject_category"].str.contains(" ")

df_vis_final = df_vis[valid_embeddings]
df_vis_final.index = list(range(len(df_vis_final)))

# +
# torch.norm(torch.Tensor(al.unique_prob_dict[3]) - torch.Tensor(al.unique_prob_dict[2]))

# +
# al.unique_prob_dict[1]
# -
L_final = L[valid_embeddings]

np.random.seed(633)
indices_shuffle = np.random.permutation(df_vis_final.shape[0])


# +
split_nr = int(np.ceil(0.9*df_vis_final.shape[0]))
train_idx, test_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]

df_train = df_vis_final.iloc[train_idx]
df_test = df_vis_final.iloc[test_idx]
df_train.index = list(range(len(df_train)))
df_test.index = list(range(len(df_test)))

L_train = L_final[train_idx,:]
L_test = L_final[test_idx,:]

# +
lm = LabelModel(n_epochs=200,
                    lr=1e-1)

Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
lm.analyze(df_train.y.values)

# +
dataset_test = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_test,
                      Y=df_test["y"].values)

dataset_train = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_train,
                      Y=Y_probs.detach())

# +
# lm_metrics = {}
# for i in range(20):
    
#     lm = LabelModel(df=df_train,
#                         active_learning=False,
#                         add_cliques=True,
#                         add_prob_loss=False,
#                         n_epochs=200,
#                         lr=1e-1)

#     Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
#     lm.analyze()
#     lm.print_metrics()
#     lm_metrics[i] = lm.metric_dict

# +
# batch_size=8

# subset_points = random.sample(range(len(df_train)), 30)
# df_train_subset=df_train.iloc[subset_points]
# df_train_subset.index = range(len(df_train_subset))

# dataset_train = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
#                       df=df_train_subset,
#                       Y=df_train.y.values[subset_points])



# dl_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
# dl_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

# final_model=VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix, soft_labels=False)

# final_model.reset()
# final_model.fit(dl_train)
# preds_test = final_model.predict(dl_test)

# +
# final_model.analyze(df_test.y.values, preds_test)

# +
# plot_train_loss(final_model.average_losses)

# +
dataset_train = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_train,
                      Y=df_train.y.values)

dl_train = DataLoader(dataset_train, shuffle=False, batch_size=256)

final_model = VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix, soft_labels=False)

feature_tensor_train = torch.Tensor([])

for batch_features, batch_labels in dl_train:
    feature_tensor_train = torch.cat((feature_tensor_train, final_model.extract_concat_features(batch_features).to("cpu")))

# +
dataset_test = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_test,
                      Y=df_test.y.values)

dl_test = DataLoader(dataset_test, shuffle=False, batch_size=256)

final_model = VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix, soft_labels=False)

feature_tensor_test = torch.Tensor([])

for batch_features, batch_labels in dl_test:
    feature_tensor_test = torch.cat((feature_tensor_test, final_model.extract_concat_features(batch_features).to("cpu")))

# +
from torch.utils.data import TensorDataset

class CustomTensorDataset(TensorDataset):
    """Custom Tensor Dataset"""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

    def update(self, X, Y):
        """Update dataset content

        Args:
            X (torch.Tensor): Tensor with features (columns)
            Y (torch.Tensor): Tensor with labels
        """
        self.X = X.clone()
        self.Y = Y


# -

dataset_train = CustomTensorDataset(feature_tensor_train, torch.Tensor(df_train.y.values))

dataset_predict = CustomTensorDataset(feature_tensor_train, torch.Tensor(df_train.y.values))

dataset_test = CustomTensorDataset(feature_tensor, torch.Tensor(df_test.y.values))

exp_kwargs = dict(nr_trials=1,
                  al_it=30,
                  label_matrix=L_train,
                  y_train=df_train.y.values,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=243, 
                  penalty_strength=1, 
                  batch_size=128,
                  discr_model_frequency=5,
                  final_model=VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix=path_prefix),
                  train_dataset=dataset_train,
                  test_dataset=dataset_test,
                  label_matrix_test=L_test,
                  y_test=df_test.y.values
                  )

np.random.seed(543)
exp_kwargs["seeds"] = np.random.randint(0,1000,10)
metrics_maxkl, queried_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

import pickle
with open("paper_results/vg_maxkl_3.pkl", "wb") as f:
    pickle.dump(metrics_maxkl, f)

np.random.seed(543)
exp_kwargs["seeds"] = np.random.randint(0,1000,10)[2:]
metrics_maxkl, queried_maxkl = active_weasul_experiment(**exp_kwargs, query_strategy="maxkl")

import pickle
with open("paper_results/vg_maxkl_4.pkl", "wb") as f:
    pickle.dump(metrics_maxkl, f)

# +
final_model_kwargs = dict(lr=1e-3,
                          n_epochs=3)

set_seed(578)

# test_dataset = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
#                       df=df_test,
#                       Y=df_test.y.values)

# train_dataset = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
#                       df=df_train,
#                       Y=df_train.y.values)

# predict_dataset = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
#                       df=df_train,
#                       Y=df_train.y.values)

batch_size = 8

features = dataset_train.X.clone()

al_exp_kwargs = dict(
    nr_trials=1,
    al_it=200,
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
# with open("paper_results/vg_activelearning.pkl", "wb") as f:
#     pickle.dump(al_accuracies, f)
# -

with open("paper_results/vg_activelearning_2.pkl", "rb") as f:
    al_accuracies = pickle.load(f)

# +
accuracy_df = pd.DataFrame.from_dict(al_accuracies[0])
accuracy_df = accuracy_df.stack().reset_index().rename(columns={"level_0": "Number of labeled points", "level_1": "Run", 0: "Value"})

accuracy_df["Metric"] = "Accuracy"
accuracy_df["Strategy"] = "Active Learning"
accuracy_df["Model"] = "Discriminative"
accuracy_df["Set"] = "test"
accuracy_df["Dash"] = "n"
# -

plot_metrics(accuracy_df)

with open("paper_results/vg_maxkl.pkl", "rb") as f:
    al_metrics_maxkl = pickle.load(f)

plot_metrics(process_exp_dict(metrics_maxkl, "Active WeaSuL"))



# +
n_epochs = 3
lr=1e-3

def visual_genome_experiment(nr_al_it, nr_runs, randomness):
    al_metrics = {}
    al_metrics["lm_metrics"] = {}
    al_metrics["lm_test_metrics"] = {}
#     al_metrics["fm_metrics"] = {}
#     al_metrics["fm_test_metrics"] = {}

    for i in range(nr_runs):
        query_strategy = "relative_entropy"

        al = ActiveLearningPipeline(it=nr_al_it,
#                                     final_model=VisualRelationClassifier(pretrained_model, df_vis_final, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix),
                                    **al_kwargs,
                                    query_strategy=query_strategy,
                                    image_dir = "/tmp/data/visual_genome/VG_100K",
                                    randomness=randomness)

        Y_probs_al = al.refine_probabilities(label_matrix=L_train, cliques=cliques, class_balance=class_balance,
                                             label_matrix_test=L_test, y_test=df_test["y"].values, dl_train=dl_train, dl_test=dl_test)
#         Y_probs_al = al.refine_probabilities(label_matrix=L_final, cliques=cliques, class_balance=class_balance,
#                                              label_matrix_test=L_final, y_test=df_vis_final["y"].values, dl_train=dl, dl_test=dl)

        al_metrics["lm_metrics"][i] = al.metrics
        al_metrics["lm_test_metrics"][i] = al.test_metrics
#         al_metrics["fm_metrics"][i] = al.final_metrics
#         al_metrics["fm_test_metrics"][i] = al.final_test_metrics
        
    return al_metrics


# -

metrics_vis_re_gen = visual_genome_experiment(50, 5, 0)

# +
# metrics_vis_random = visual_genome_experiment(30, 10, 1)
# -

import pickle
with open("results/al_metrics_vis_split.pkl", "rb") as f:
    metrics_vis_re_gen = pickle.load(f)

import pickle
with open("results/al_metrics_vis_gen_2.pickle", "rb") as f:
    metrics_vis_re_gen = pickle.load(f)


# +
# import pickle
# with open("results/al_metrics_vis_gen.pickle", "wb") as f:
#     pickle.dump(metrics_vis_re_gen, f)
# -

def create_metric_df(al_metrics, nr_runs, metric_string, strategy_string, model_string):
    joined_metrics = pd.DataFrame()
    for i in range(nr_runs):
        int_df = pd.DataFrame.from_dict(al_metrics[metric_string][i]).drop("Labels", errors="ignore").T
        int_df = int_df.stack().reset_index().rename(columns={"level_0": "Active Learning Iteration", "level_1": "Metric", 0: "Value"})
        int_df["Run"] = str(i)

        joined_metrics = pd.concat([joined_metrics, int_df])

    joined_metrics["Value"] = joined_metrics["Value"].apply(pd.to_numeric)
    joined_metrics["Set"] = strategy_string
    joined_metrics["Model"] = model_string
    joined_metrics["Label"] = "AL"
    
    return joined_metrics


all_metrics_joined = pd.concat([create_metric_df(metrics_vis_re_gen, 5, "lm_metrics", "train", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 5, "lm_test_metrics", "test", "Generative")])

metrics_vis_re_gen["lm_metrics"].keys()

# +
import pickle
with open("results/al_metrics_vis_split.pkl", "rb") as f:
    metrics_vis_re_gen = pickle.load(f)

metrics_joined = pd.concat([create_metric_df(metrics_vis_re_gen, 1, "lm_metrics", "train", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 1, "lm_test_metrics", "test", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 1, "fm_metrics", "train", "Discriminative"),
                           create_metric_df(metrics_vis_re_gen, 1, "fm_test_metrics", "test", "Discriminative")])


# +
with open("results/al_metrics_vis_split_2.pkl", "rb") as f:
    metrics_vis_re_gen = pickle.load(f)

metrics_joined_2 = pd.concat([create_metric_df(metrics_vis_re_gen, 1, "lm_metrics", "train", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 1, "lm_test_metrics", "test", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 1, "fm_metrics", "train", "Discriminative"),
                           create_metric_df(metrics_vis_re_gen, 1, "fm_test_metrics", "test", "Discriminative")])

# +
with open("results/al_metrics_vis_split_3.pkl", "rb") as f:
    metrics_vis_re_gen = pickle.load(f)

metrics_joined_3 = pd.concat([create_metric_df(metrics_vis_re_gen, 1, "lm_metrics", "train", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 1, "lm_test_metrics", "test", "Generative"),
                           create_metric_df(metrics_vis_re_gen, 1, "fm_metrics", "train", "Discriminative"),
                           create_metric_df(metrics_vis_re_gen, 1, "fm_test_metrics", "test", "Discriminative")])
# -

all_metrics_joined = pd.concat([metrics_joined_2, metrics_joined_3])

metrics_joined = pd.concat([metrics_joined, pd.read_csv("results/vis_re.csv")])

# +
# metrics_joined.to_csv("results/vis_re.csv")
# -

all_metrics_joined = pd.read_csv("results/vis_re.csv")

metrics_joined

sns.set_theme(style="white")
colors = ["#086788",  "#e3b505","#ef7b45",  "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))

# +
# ax = sns.relplot(data=metrics_joined, x="Active Learning Iteration", y="Value", col="Metric", hue = "Set",
#                  ci=68, n_boot=1000, estimator="mean", kind="line", legend=False)
# ax.axes[0][0].set_ylim((0.2,0.9))
# plt.show()

# +
colors = ["#2b4162", "#721817", "#e9c46a", "#fa9f42", "#0b6e4f", "#96bdc6",  "#c09891", "#5d576b", "#c6dabf", "#368f8b", "#ec7357"]

pick_colors = [colors[9], colors[10]]

all_metrics_joined = all_metrics_joined.rename(columns={"Active Learning Iteration": "Number of labeled points"})

sns.set(style="whitegrid")
ax = sns.relplot(data=all_metrics_joined, x="Number of labeled points", y="Value", col="Metric", row="Model",
                 kind="line", estimator="mean", ci=68, hue="Set", style="Label",legend=False, palette=sns.color_palette(pick_colors))

show_handles = [ax.axes[0][0].lines[0], ax.axes[0][0].lines[1]]
show_labels = ["train", "test"]
ax.axes[0][3].legend(handles=show_handles, labels=show_labels, loc="lower right")

ax.set_ylabels("")
ax.set_titles("{col_name}")
# plt.savefig("plots/vis_gen_metrics.png")
plt.show()
# fig = ax.get_figure()

# -

ax = sns.relplot(data=metrics_joined, x="Active Learning Iteration", y="Value", col="Metric", hue = "Set", row="Model",
                 ci=None, estimator="mean",kind="line", legend=False)

ax = sns.relplot(data=metrics_joined, x="Active Learning Iteration", y="Value", col="Metric", hue = "Set", row="Model",
                 ci=68, n_boot=1000, estimator="mean",kind="line", legend=False)

ax = sns.relplot(data=pd.read_csv("results/vis_re.csv"), x="Active Learning Iteration", y="Value", col="Metric", hue = "Set", row="Model",
                 ci=68, n_boot=1000, estimator="mean",kind="line", legend=False)

al.plot_metrics()

metrics_df_vis = create_metric_df(metrics_vis, 10, "lm_metrics", "Relative Entropy")
metrics_df_vis_random = create_metric_df(metrics_vis_random, 10, "lm_metrics", "Random")

metrics_df_vis = metrics_df_vis[metrics_df_vis["Metric"] == "Accuracy"]
metrics_df_vis_random = metrics_df_vis_random[metrics_df_vis_random["Metric"] == "Accuracy"]

# +
# metrics_df_vis.groupby("Active Learning Iteration").agg(lambda x: x.quantile(0.25))

# +
# metrics_df_vis.groupby("Active Learning Iteration").agg(lambda x: x.quantile(0.5))
# -

metrics_vis_joined = pd.concat([metrics_df_vis, metrics_df_vis_random])



al_metrics_vis = visual_genome_experiment(30, 10)

import pickle
file_metrics = open("results/al_metrics_vis.pkl", "wb")
pickle.dump(al_metrics_vis, file_metrics)
file_metrics.close()


def visual_genome_experiment(nr_al_it, nr_runs):
    al_metrics = {}
    al_metrics["lm_metrics"] = {}
    al_metrics["fm_metrics"] = {}

    for i in range(nr_runs):
        query_strategy = "relative_entropy"

        al = ActiveLearningPipeline(it=nr_al_it,
                                    final_model=VisualRelationClassifier(pretrained_model, dl_al_test, df_vis_final, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix),
                                    **al_kwargs,
                                    query_strategy=query_strategy,
                                    image_dir = "/tmp/data/visual_genome/VG_100K",
                                    randomness=0)

        Y_probs_al = al.refine_probabilities(label_matrix=L_final, cliques=cliques, class_balance=class_balance, label_matrix_test=L_final, y_test=df_vis_final["y"].values)
        al.label_model.print_metrics()
        al.final_model.print_metrics()
        al_metrics["lm_metrics"][i] = al.metrics
        al_metrics["fm_metrics"][i] = al.final_metrics
        
    return al_metrics


al_metrics_vis_2 = visual_genome_experiment(30, 1)

file_metrics = open("results/al_metrics_vis_2.pkl", "wb")
pickle.dump(al_metrics_vis_2, file_metrics)
file_metrics.close()

al.plot_metrics(al.test_metrics)

al.plot_metrics(al.metrics)

al.final_model.losses

# +
mean_metrics = pd.DataFrame.from_dict(lm_metrics, orient="index").mean().reset_index().rename(columns={"index": "Metric"})
mean_metrics["std"] = pd.DataFrame.from_dict(lm_metrics, orient="index").sem().values
mean_metrics["Active Learning"] = "before"

mean_al_metrics = pd.DataFrame.from_dict(al_metrics, orient="index").mean().reset_index().rename(columns={"index": "Metric"})
mean_al_metrics["std"] = pd.DataFrame.from_dict(al_metrics, orient="index").sem().values
mean_al_metrics["Active Learning"] = "after"

metrics_joined = pd.concat([mean_metrics, mean_al_metrics])
# -

fig = px.bar(metrics_joined, x="Metric", y=0, error_y="std", color="Active Learning", barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(template="plotly_white", yaxis_title="", title_text="Label model performance before and after active learning (error bar = standard error)")
fig.show()

al.plot_metrics()

plot_train_loss(al.label_model.losses)

al.plot_parameters()

al.plot_iterations()

al.queried

plot_train_loss(al.label_model.losses)





# +
n_epochs = 3
lr=1e-2
batch_size = 256

valid_embeddings = (df_vis["channels"] == 3) & df_vis.object_category.isin(word_embs) & df_vis.subject_category.isin(word_embs) & ~df_vis["object_category"].str.contains(" ") & ~df_vis["subject_category"].str.contains(" ")

df_vis_final = df_vis[valid_embeddings]
df_vis_final.index = list(range(len(df_vis_final)))

dataset_al = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_vis_final, 
                      Y=Y_probs_al.clone().clamp(0,1).detach().numpy())

dl_al = DataLoader(dataset_al, shuffle=True, batch_size=batch_size)
dl_al_test = DataLoader(dataset_al, shuffle=False, batch_size=batch_size)

vc_al = VisualRelationClassifier(pretrained_model, dl_al_test, df_vis_final, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix)

probs_final_al = vc_al.fit(dl_al).predict()

vc_al.analyze()

vc_al.print_metrics()
# -

vc_al.losses[0].cpu().detach().numpy()

vc_al_losses = [t.cpu().item() for t in vc_al.losses]

import pickle
pickle.dump(vc_al_losses, open("results/discriminative_loss.p", "wb"))

plot_train_loss(vc_al_losses)

# +
# for image in df_vis_final["source_img"]:
# #     ! cp ../data/visual_genome/VG_100K/$image ../data/visual_genome/subset_VG/$image
# -






