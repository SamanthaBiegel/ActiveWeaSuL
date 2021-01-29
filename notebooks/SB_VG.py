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
DAP = True
    
if DAP:
    # ! pip install -r ../requirements.txt
    # ! aws s3 cp s3://user/gc03ye/uploads/glove /tmp/data/word_embeddings --recursive
    # ! aws s3 cp s3://user/gc03ye/uploads/resnet_old.pth /tmp/models/resnet_old.pth
    # ! aws s3 cp s3://user/gc03ye/uploads /tmp/data/visual_genome/VG_100K --recursive --exclude "glove/*" --exclude "resnet_old.pth" --exclude "resnet.pth" --exclude "siton_dataset.csv" --exclude "train.zip" --exclude "VRD*"
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
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from plot import plot_probs, plot_train_loss
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
from lf_utils import apply_lfs, analyze_lfs
from visualrelation import VisualRelationDataset, VisualRelationClassifier, WordEmb, FlatConcat
# -

if DAP:
    df_vis = pd.read_csv("../../../s3_home/uploads/siton_dataset.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})
else:
    df_vis = pd.read_csv("../data/siton_dataset.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})

# +
OTHER = 0
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

lfs = [lf_siton_object, lf_not_person, lf_area, lf_dist, lf_ydist]

cliques = [[0,1],[2],[3],[4]]
# cliques=[[0],[1,2,3],[4]]
# -

L = apply_lfs(df_vis, lfs)

analyze_lfs(L, df_vis["y"], lfs)

class_balance = np.array([1-df_vis.y.mean(), df_vis.y.mean()])

# +
lm = LabelModel(n_epochs=200,
                    lr=1e-1)

Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
lm.analyze(df_train.y.values)
# -


true_probs = lm.predict_true(df_train.y.values)

lm.analyze(df_train.y.values, true_probs)

# +
# batch_size=8

# subset_points = random.sample(range(len(df_train)), 30)
# df_train_subset=df_train.iloc[subset_points]
# df_train_subset.index = range(len(df_train_subset))

# dataset_train = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
#                       df=df_train_subset,
#                       Y=df_train.y.values[subset_points])

batch_size=20

dataset_train.Y = true_probs

dl_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
dl_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

final_model=VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=5, data_path_prefix=path_prefix, soft_labels=True)

final_model.reset()
final_model.fit(dl_train)
preds_test = final_model.predict(dl_test)
preds_train = final_model.predict()
# -

plot_train_loss(final_model.average_losses)

final_model.analyze(df_train.y.values, preds_train)
final_model.analyze(df_test.y.values, preds_test)

dataset_train = CustomTensorDataset(feature_tensor_train, torch.Tensor(df_train.y.values))

dataset_predict = CustomTensorDataset(feature_tensor_train, torch.Tensor(df_train.y.values))

dataset_test = CustomTensorDataset(feature_tensor_test, torch.Tensor(df_test.y.values))

exp_kwargs = dict(nr_trials=1,
                  al_it=30,
                  label_matrix=L_train,
                  y_train=df_train.y.values,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=243, 
                  penalty_strength=1e3, 
                  batch_size=128,
                  discr_model_frequency=1,
                  final_model=VisualRelationClassifier(pretrained_model, lr=1e-4, n_epochs=10, data_path_prefix=path_prefix),
                  train_dataset=dataset_predict,
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

# +
final_model_kwargs = dict(lr=1e-2,
                          n_epochs=3)

set_seed(578)

batch_size = 64

features = dataset_predict.X.clone()

al_exp_kwargs = dict(
    nr_trials=6,
    al_it=50,
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

with open("paper_results/vg_activelearning.pkl", "wb") as f:
    pickle.dump(al_accuracies, f)

with open("paper_results/vg_activelearning.pkl", "rb") as f:
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






