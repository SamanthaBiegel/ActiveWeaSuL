# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# # ! pip install -r ../../requirements.txt

# +
# # ! aws s3 cp s3://user/gc03ye/uploads/VRD /tmp/data/VRD --recursive

# +
# path to use when running on DAP
# path_prefix = "/tmp/"

path_prefix = ""

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import random
import time
import scipy as sp
import cvxpy as cp
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torchvision.models as models
import torch

import sys
import os
sys.path.append(os.path.abspath("../../../../public_repos/snorkel/snorkel"))
from classification.training.trainer import Trainer
from classification.multitask_classifier import MultitaskClassifier

from snorkel.labeling import LFAnalysis, labeling_function, PandasLFApplier
from labeling.model import SnorkelLabelModel
from snorkel.classification import DictDataLoader

from model import SceneGraphDataset, create_model
from utils import load_vrd_data

sys.path.append(os.path.abspath("../../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs
from label_model import LabelModel
from pipeline import ActiveLearningPipeline

from visualrelation import VisualRelationDataset, VisualRelationClassifier, WordEmb, FlatConcat

from torch.utils.data import DataLoader
import torch.nn as nn

# +
df_train, df_valid, df_test = load_vrd_data(path_prefix=path_prefix, sample=False)

print("Train Relationships: ", len(df_train))
print("Dev Relationships: ", len(df_valid))
print("Test Relationships: ", len(df_test))
# -

df_train = df_train.rename(columns={"label": "y"})
df_valid = df_valid.rename(columns={"label": "y"})
df_test = df_test.rename(columns={"label": "y"})

# # **Define labeling functions**

SITON = 0
OTHER = 1
ABSTAIN = -1


# +
@labeling_function()
def lf_siton_object(x):
    if x.subject_category == "person":
        if x.object_category in [
            "bench",
            "chair",
            "floor",
            "horse",
            "grass",
            "table",
        ]:
            return SITON
    return OTHER

@labeling_function()
def lf_not_person(x):
    if x.subject_category != "person":
        return OTHER
    return SITON


# -

YMIN = 0
YMAX = 1
XMIN = 2
XMAX = 3


# +
@labeling_function()
def lf_ydist(x):
    if x.subject_bbox[YMAX] < x.object_bbox[YMAX] and x.subject_bbox[YMIN] < x.object_bbox[YMIN]:
        return OTHER
    return SITON

@labeling_function()
def lf_xdist(x):
    if x.subject_bbox[XMAX] < x.object_bbox[XMIN] or x.subject_bbox[XMIN] > x.object_bbox[XMAX]:
        return SITON
    return OTHER


@labeling_function()
def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) >= 500:
        return SITON
    return OTHER



def area(bbox):
    return (bbox[YMAX] - bbox[YMIN]) * (bbox[XMAX] - bbox[XMIN])

@labeling_function()
def lf_area(x):
    if area(x.subject_bbox) / area(x.object_bbox) < 0.8:
        return SITON
    return OTHER


# +
lfs = [
#     lf_ride_object,
    lf_siton_object,
    lf_not_person,
#     lf_ydist,
    lf_xdist,
    lf_dist,
    lf_area,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_valid = applier.apply(df_valid)
L_test = applier.apply(df_test)

# +
Y_train = np.array(df_train["y"])
Y_valid = np.array(df_valid["y"])
Y_test = np.array(df_test["y"])

Y_true = Y_train.copy()
# -

LFAnalysis(L_train, lfs).lf_summary(Y_train)

# Class balance
df_train["y"].mean()

# # **Initial fit label model**

class_balance = np.array([0.25, 0.75])
cliques=[[0,1],[2,3],[4]]

# +
lm = LabelModel(df=df_train,
                active_learning=False,
                add_cliques=True,
                add_prob_loss=False,
                n_epochs=500,
                lr=1e-1)
    
Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
lm.analyze()
lm.print_metrics()

# +
# label_model = SnorkelLabelModel(cardinality=cardinality)
# label_model.fit(L_train, class_balance=class_balance)
# _, Y_probs = label_model.predict(L_train, return_probs=True)
# label_model.score(L_valid, Y_valid, metrics=metrics)
# -

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'df': df_train,
             'n_epochs': 200
            }

# +
it = 20
query_strategy = "margin"
    
al = ActiveLearningPipeline(it=it,
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L_train, cliques=cliques, class_balance=class_balance)
al.label_model.analyze()
al.label_model.print_metrics()
# -

al.color_df()

# # **Train discriminative model on probabilistic labels**

# +
metrics = ["accuracy", "precision", "recall", "f1"]
train_on = "probs" # probs or labels
batch_size = 20
n_epochs = 3
lr = 1e-3

pretrained_model = models.resnet18(pretrained=True)

# +
dataset = VisualRelationDataset(image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y_probs.clone().detach().numpy())

dl = DataLoader(dataset, shuffle=True, batch_size=batch_size)
dl_test = DataLoader(dataset, shuffle=False, batch_size=batch_size)

vc = VisualRelationClassifier(pretrained_model, WordEmb(), FlatConcat(), dl, dl_test, df_train, n_epochs=n_epochs, lr=lr)

probs_final = vc.fit().predict()

vc.analyze()

vc.print_metrics()

# +
# dataset = VisualRelationDataset(image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
#                       df=df_train[:20], 
#                       Y=Y_probs.detach()[:20])

# dl_test = DataLoader(dataset, shuffle=False, batch_size=20)

# +
dataset_al = VisualRelationDataset(image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y_probs_al.clone().detach().numpy())

dl_al = DataLoader(dataset_al, shuffle=True, batch_size=batch_size)
dl_al_test = DataLoader(dataset_al, shuffle=False, batch_size=batch_size)

vc_al = VisualRelationClassifier(pretrained_model, WordEmb(), FlatConcat(), dl_al, dl_al_test, df_train, n_epochs=n_epochs, lr=lr)

probs_final_al = vc_al.fit().predict()

vc_al.analyze()

vc_al.print_metrics()
# -

# # Compare to Snorkel final model

# +
train_on="probs"
first_probs = Y_probs.clone().detach().numpy()
# first_probs = Y_probs

if train_on == "probs":
    Y = first_probs
    with_prob = True
if train_on == "labels":
    Y = torch.LongTensor(first_labels)
    with_prob = False

dl_train = DictDataLoader(
    SceneGraphDataset(name="train_dataset", 
                      split="train", 
                      image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y),
    batch_size=batch_size,
    shuffle=True,
)

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)
model = create_model(cnn, with_prob=with_prob)

trainer = Trainer(
    n_epochs=n_epochs,  # increase for improved performance
    lr=lr,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"}
)
trainer.fit(model, [dl_train])
# +
dl_train_test = DictDataLoader(
    SceneGraphDataset(name="train_dataset", 
                      split="train", 
                      image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=df_train["y"].values),
    batch_size=batch_size,
    shuffle=False,
)

score = model.score([dl_train_test], as_dataframe=True)
# -

score

# +
train_on="probs"
first_probs = Y_probs_al.clone().detach().numpy()

if train_on == "probs":
    Y = first_probs
    with_prob = True
if train_on == "labels":
    Y = torch.LongTensor(first_labels)
    with_prob = False

dl_train = DictDataLoader(
    SceneGraphDataset(name="train_dataset", 
                      split="train", 
                      image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y),
    batch_size=batch_size,
    shuffle=True,
)

dl_train_test = DictDataLoader(
    SceneGraphDataset(name="train_dataset", 
                      split="train", 
                      image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y),
    batch_size=batch_size,
    shuffle=False,
)

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)
model = create_model(cnn, with_prob=with_prob)

trainer = Trainer(
    n_epochs=n_epochs,  # increase for improved performance
    lr=lr,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"}
)
trainer.fit(model, [dl_train])

score = model.score([dl_train_test], as_dataframe=True)
# -

trainer.metrics

preds = model.predict(dl_train)

preds


# +
df_train_all, df_valid_all, df_test_all = load_vrd_data(path_prefix=path_prefix, sample=False)

print("Train Relationships: ", len(df_train_all))
print("Dev Relationships: ", len(df_valid_all))
print("Test Relationships: ", len(df_test_all))
# -

df_train_all[df_train_all["source_img"] == "8538884882_02c6a81024_b.jpg"]




