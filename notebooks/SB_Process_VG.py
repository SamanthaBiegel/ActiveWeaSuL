# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
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

# %%
# %load_ext autoreload
# %autoreload 2

import json
import numpy as np
import random
import time
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from PIL import Image
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

# %%
with open(path_prefix + 'data/visual_genome/relationships.json') as f:
  visgen_rels = json.load(f)

# %%
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

# %%
visgen_df = pd.json_normalize(visgen_rels, record_path=["relationships"], meta="image_id", sep="_")
visgen_df["predicate"] = visgen_df["predicate"].str.lower()
visgen_df_actions = visgen_df[visgen_df["predicate"].isin(pred_list)]
visgen_df_actions.to_csv(path_prefix + "data/action_dataset.csv", index=False)

# %%
pred_action = "sitting on"

# %%
visgen_df_actions = pd.read_csv(path_prefix + "data/action_dataset.csv")

# %%
visgen_df_actions["y"] = visgen_df_actions["predicate"]
visgen_df_actions["y"] = visgen_df_actions["y"].apply(lambda x: 1 if x == pred_action else 0)
df_vis = visgen_df_actions.loc[:,["image_id", "predicate", "object_name", "object_h", "object_w", "object_y", "object_x", "subject_name", "subject_h", "subject_w", "subject_y", "subject_x", "y"]]
df_vis = df_vis.dropna()
df_vis = df_drop_duplicates(df_vis)
# df_vis = balance_dataset(df_vis)

# %%
df_vis["object_x_max"] = df_vis["object_x"] + df_vis["object_w"]
df_vis["object_y_max"] = df_vis["object_y"] + df_vis["object_h"]
df_vis["subject_x_max"] = df_vis["subject_x"] + df_vis["subject_w"]
df_vis["subject_y_max"] = df_vis["subject_y"] + df_vis["subject_h"]

df_vis["object_bbox"] = tuple(df_vis[["object_y", "object_y_max", "object_x", "object_x_max"]].values)
df_vis["subject_bbox"] = tuple(df_vis[["subject_y", "subject_y_max", "subject_x", "subject_x_max"]].values)

df_vis = df_vis.rename(columns={"object_name": "object_category", "subject_name": "subject_category", "image_id": "source_img"})

df_vis.source_img = df_vis.source_img.astype(str) + ".jpg"

# %%
df_vis["channels"] = df_vis["source_img"].apply(lambda x: len(np.array(Image.open(path_prefix + "data/visual_genome/VG_100K" + "/" + x)).shape))

# %%
df_vis.to_csv(path_prefix + "data/siton_dataset.csv", index=False)

# %%
# all_img = list(df_vis.source_img.drop_duplicates())

# with open("image_files.txt", "r") as f:
#     files = f.readlines()
#     image_files = [file.strip() for file in files]
    
# subset_images = list(df_vis["source_img"].drop_duplicates())
# missing_images = [image for image in subset_images if image not in image_files]
# len(missing_images)

# %%
# for image in missing_images:
# #     ! cp ../data/visual_genome/VG_100K/$image ../data/visual_genome/missing_VG/$image

# %%
# predicate_counts = visgen_df.groupby("predicate")["image_id"].count().sort_values(ascending=False)
# predicate_counts[predicate_counts > 1000]

# %%
# pd.set_option('display.max_rows',102)
# pd.DataFrame(df_train.groupby("y")["source_img"].count())

# %%
word_embs = pd.read_csv(
            path_prefix + "data/word_embeddings/glove.6B.100d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
        ).T
word_embs = list(word_embs.columns)

# %%
valid_embeddings = (df_vis["channels"] == 3) & df_vis.object_category.isin(word_embs) & df_vis.subject_category.isin(word_embs) & ~df_vis["object_category"].str.contains(" ") & ~df_vis["subject_category"].str.contains(" ")

df_vis_final = df_vis[valid_embeddings]
df_vis_final.index = list(range(len(df_vis_final)))

# %%
L_final = L[valid_embeddings]

# %%
np.random.seed(633)
indices_shuffle = np.random.permutation(df_vis_final.shape[0])

# %%
split_nr = int(np.ceil(0.9*df_vis_final.shape[0]))
train_idx, test_idx = indices_shuffle[:split_nr], indices_shuffle[split_nr:]

df_train = df_vis_final.iloc[train_idx]
df_test = df_vis_final.iloc[test_idx]
df_train.index = list(range(len(df_train)))
df_test.index = list(range(len(df_test)))

L_train = L_final[train_idx,:]
L_test = L_final[test_idx,:]

# %%

# %%

# %%

# %%

# %%

# %%
