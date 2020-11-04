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
DAP = False
    
if DAP:
    # ! pip install -r requirements.txt
    # ! aws s3 cp s3://user/gc03ye/uploads/VRD /tmp/data/VRD --recursive
    # ! aws s3 cp s3://user/gc03ye/uploads/glove /tmp/data/glove --recursive
    # ! aws s3 cp s3://user/gc03ye/uploads/resnet_old.pth /tmp/models/resnet_old.pth
    path_prefix = "/tmp/"
    pretrained_model = torch.load(path_prefix + "models/resnet_old.pth")
else:
    import torchvision.models as models
    pretrained_model = models.resnet18(pretrained=True)
    path_prefix = "../"

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
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
from lm_utils import apply_lfs, analyze_lfs
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
# -

import ast
df_vis = pd.read_csv(path_prefix + "data/siton_dataset.csv", converters={"object_bbox": ast.literal_eval, "subject_bbox": ast.literal_eval})

df_vis.source_img.drop_duplicates()

df_vis[df_vis.subject_category == "dog"][:30]

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
WEAR = 1

def lf_wear_object(x):
    if x.subject_name == "person":
        if x.object_name in ["t-shirt", "jeans", "glasses", "skirt", "pants", "shorts", "dress", "shoes"]:
            return WEAR
    return OTHER

def lf_area(x):
    if (x.subject_w * x.subject_h) / (x.object_w * x.object_h) > 1:
        return WEAR
    return OTHER

def lf_dist(x):
    if ((x.subject_x - x.object_x) + (x.subject_y - x.object_y)) > 10:
        return OTHER
    return WEAR

def lf_ydist(x):
    if x.subject_y_max > x.object_y_max and x.subject_y < x.object_y:
        return WEAR
    return OTHER

lfs = [lf_wear_object, lf_dist, lf_area]

cliques=[[0],[1,2]]

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

lfs = [lf_area, lf_dist, lf_siton_object]

cliques = [[0,1],[2]]
# cliques=[[0],[1,2,3],[4]]
# -

L = apply_lfs(df_vis, lfs)

analyze_lfs(L, df_vis["y"], lfs)

class_balance = np.array([1-df_vis.y.mean(), df_vis.y.mean()])





lm_metrics = {}
for i in range(50):
    
    lm = LabelModel(df=df_vis,
                        active_learning=False,
                        add_cliques=True,
                        add_prob_loss=False,
                        n_epochs=200,
                        lr=1e-1)

    Y_probs = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
    lm.analyze()
    lm.print_metrics()
    lm_metrics[i] = lm.metric_dict

plot_train_loss(lm.losses)

metrics = ["accuracy", "precision", "recall", "f1"]
train_on = "probs" # probs or labels
n_epochs = 3
lr = 1e-3
batch_size=20

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'df': df_vis,
             'n_epochs': 200,
             'batch_size': batch_size,
             'lr': 1e-1
            }

torch.norm(torch.Tensor(al.unique_prob_dict[3]) - torch.Tensor(al.unique_prob_dict[2]))

al.unique_prob_dict[1]

al_metrics = {}
for i in range(1):
    it = 20
    query_strategy = "relative_entropy"

    al = ActiveLearningPipeline(it=it,
                                **al_kwargs,
                                query_strategy=query_strategy,
                                randomness=0)

    Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
    al.label_model.print_metrics()
    al_metrics[i] = al.label_model.metric_dict

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

al.ground_truth_labels[al.queried]

al.plot_metrics()

plot_train_loss(al.label_model.losses)

al.plot_parameters()

al.plot_iterations()

al.queried

plot_train_loss(al.label_model.losses)

import csv
word_embs = pd.read_csv(
            "../data/word_embeddings/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        ).T
word_embs = list(word_embs.columns)



# +
n_epochs = 10
lr=1e-2
batch_size = 256

valid_embeddings = (df_vis["channels"] == 3) & df_vis.object_category.isin(word_embs) & df_vis.subject_category.isin(word_embs) & ~df_vis["object_category"].str.contains(" ") & ~df_vis["subject_category"].str.contains(" ")

df_vis_final = df_vis[valid_embeddings]
df_vis_final.index = list(range(len(df_vis_final)))

dataset_al = VisualRelationDataset(image_dir=path_prefix + "data/visual_genome/VG_100K", 
                      df=df_vis_final, 
                      Y=Y_probs_al.clone().clamp(0,1).detach().numpy()[valid_embeddings])

dl_al = DataLoader(dataset_al, shuffle=True, batch_size=batch_size)
dl_al_test = DataLoader(dataset_al, shuffle=False, batch_size=batch_size)

vc_al = VisualRelationClassifier(pretrained_model, dl_al_test, df_vis_final, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix)

probs_final_al = vc_al.fit(dl_al).predict()

vc_al.analyze()

vc_al.print_metrics()
# -

for image in df_vis_final["source_img"]:
    # ! cp ../data/visual_genome/VG_100K/$image ../data/visual_genome/subset_VG/$image

from scipy.stats import entropy
entropy([1/2, 1/2], qk=[0.99, 0.01])

entropy(pk=[0.55, 0.45], qk=[0.5, 0.5])

entropy(pk=[1-1e-4, 1e-4], qk=[1-1e-5, 1e-5])

entropy(qk=[0.5, 0.5], pk=[0.6,0.4])

lm_posteriors = al.unique_prob_dict[0]
lm_posteriors = np.concatenate([1-lm_posteriors[:, None], lm_posteriors[:, None]], axis=1).clip(1e-6, 1-1e-6)

len(lm_posteriors)

bucket_gt = al.ground_truth_labels[np.where(al.unique_inverse == 1)]

bucket_gt[bucket_gt != -1].size



al.unique_inverse

df_vis_final






