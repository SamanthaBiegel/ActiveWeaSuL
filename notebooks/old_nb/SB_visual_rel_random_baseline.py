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
import seaborn as sns
import torchvision.models as models
import torch

import sys
import os
sys.path.append(os.path.abspath("../../snorkel/snorkel"))
from labeling.model.label_model import LabelModel

from snorkel.labeling import LFAnalysis, labeling_function, PandasLFApplier
from snorkel.classification import DictDataLoader, Trainer

from model import SceneGraphDataset, create_model
from utils import load_vrd_data

# +
df_train, df_valid, df_test = load_vrd_data()

print("Train Relationships: ", len(df_train))
print("Dev Relationships: ", len(df_valid))
print("Test Relationships: ", len(df_test))
# -

# # **Define labeling functions**

RIDE = 0
CARRY = 1
OTHER = 2
ABSTAIN = -1


# +
# Category-based LFs
@labeling_function()
def lf_ride_object(x):
    if x.subject_category == "person":
        if x.object_category in [
            "bike",
            "snowboard",
            "motorcycle",
            "horse",
            "bus",
            "truck",
            "elephant",
        ]:
            return RIDE
    return ABSTAIN


@labeling_function()
def lf_carry_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bag", "surfboard", "skis"]:
            return CARRY
    return ABSTAIN


@labeling_function()
def lf_carry_subject(x):
    if x.object_category == "person":
        if x.subject_category in ["chair", "bike", "snowboard", "motorcycle", "horse"]:
            return CARRY
    return ABSTAIN


@labeling_function()
def lf_not_person(x):
    if x.subject_category != "person":
        return OTHER
    return ABSTAIN


# -

YMIN = 0
YMAX = 1
XMIN = 2
XMAX = 3


# +
# Distance-based LFs
@labeling_function()
def lf_ydist(x):
#     if x.subject_bbox[YMIN] < x.object_bbox[YMIN]:
#     if x.subject_bbox[XMIN] < x.object_bbox[XMIN]:
#     if area(x.object_bbox) > area(x.subject_bbox) and x.subject_bbox[YMIN] < x.object_bbox[YMAX]:
    if x.subject_bbox[YMIN] > x.object_bbox[YMIN]:
        return CARRY
    return ABSTAIN


@labeling_function()
def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) >= 350:
#     if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) <= 1000:
        return OTHER
    return ABSTAIN


def area(bbox):
    return (bbox[YMAX] - bbox[YMIN]) * (bbox[XMAX] - bbox[XMIN])


# Size-based LF
@labeling_function()
def lf_area(x):
    if area(x.subject_bbox) / area(x.object_bbox) <= 0.5:
        return OTHER
    return ABSTAIN


# +
lfs = [
    lf_ride_object,
    lf_carry_object,
    lf_carry_subject,
    lf_not_person,
    lf_ydist,
    lf_dist,
    lf_area,
]

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
L_valid = applier.apply(df_valid)
L_test = applier.apply(df_test)

# +
Y_train = np.array(df_train["label"])
Y_valid = np.array(df_valid["label"])
Y_test = np.array(df_test["label"])

Y_true = Y_train.copy()
# -

LFAnalysis(L_valid, lfs).lf_summary(Y_valid)

# # **Initial fit label model**

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train)

label_model.score(L_valid, Y_valid, metrics=["f1_micro"])

# # **Train discriminative model on probabilistic labels**

train_on = "probs" # probs or labels

# +
first_labels, first_probs = label_model.predict(L_train, return_probs=True)

if train_on == "probs":
    Y = first_probs
    with_prob = True
if train_on == "labels":
    Y = torch.LongTensor(first_labels)
    with_prob = False

dl_train = DictDataLoader(
    SceneGraphDataset(name="train_dataset", 
                      split="train", 
                      image_dir="data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y),
    batch_size=16,
    shuffle=True,
)

dl_valid = DictDataLoader(
    SceneGraphDataset(name="valid_dataset", 
                      split="valid", 
                      image_dir="data/VRD/sg_dataset/sg_train_images", 
                      df=df_valid, 
                      Y=Y_valid),
    batch_size=16,
    shuffle=False,
)

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)
model = create_model(cnn, with_prob=with_prob)

trainer = Trainer(
    n_epochs=3,  # increase for improved performance
    lr=1e-3,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"},
)
trainer.fit(model, [dl_train])

model.score([dl_valid])
# -

# # **Add AL label to label matrix**

wl_al_train = np.full_like(Y_train, -1)
wl_al_valid = np.full_like(Y_valid, -1)
wl_al_test = np.full_like(Y_test, -1)

L_train = np.concatenate([L_train, wl_al_train.reshape(len(wl_al_train),1)], axis=1)
L_valid = np.concatenate([L_valid, wl_al_valid.reshape(len(wl_al_valid),1)], axis=1)
L_test = np.concatenate([L_test, wl_al_test.reshape(len(wl_al_test),1)], axis=1)

# # **Iteratively refine label matrix**
#

# +
LM = LabelModel(cardinality=3)
LM.fit(L_train)

# Get current label parameters
weights = pd.DataFrame([LM.get_weights()])

# +
it = 10

for i in range(it):
    Y_hat, Y_probs = LM.predict(L_train, return_probs=True)
    
    # Pick a random point
    all_abstain = L_train.sum(axis=1) == -L_train.shape[1]
    indices = np.array(range(len(Y_true)))
    random.seed(random.SystemRandom().random())
    sel_idx = random.choice(indices[~all_abstain])
    L_train[sel_idx, LM.m-1] = Y_true[sel_idx]
    print("Iteration:", i+1, " Label combination", L_train[sel_idx,:len(lfs)], " True label:",Y_true[sel_idx], "Estimated label:", Y_hat[sel_idx], " selected index:", sel_idx)
    
    # Fit label model on refined label matrix
    LM.fit(L_train)
    
    # Add current label parameters
    weights = weights.append([LM.get_weights()])
# -

# # **Train discriminative model on refined labels**

# +
second_labels, second_probs = LM.predict(L_train, return_probs=True)

if train_on == "probs":
    Y = second_probs
    with_prob = True
if train_on == "labels":
    Y = torch.LongTensor(second_labels)
    with_prob = False

dl_train = DictDataLoader(
    SceneGraphDataset(name="train_dataset", 
                      split="train", 
                      image_dir="data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=Y),
    batch_size=16,
    shuffle=True,
)

dl_valid = DictDataLoader(
    SceneGraphDataset(name="valid_dataset", 
                      split="valid", 
                      image_dir="data/VRD/sg_dataset/sg_train_images", 
                      df=df_valid, 
                      Y=Y_valid),
    batch_size=16,
    shuffle=False,
)

# initialize pretrained feature extractor
cnn = models.resnet18(pretrained=True)
model = create_model(cnn, with_prob=with_prob)

trainer = Trainer(
    n_epochs=3,  # increase for improved performance
    lr=1e-3,
    checkpointing=True,
    checkpointer_config={"checkpoint_dir": "checkpoint"},
)

trainer.fit(model, [dl_train])

model.score([dl_valid])


# +
# model.predict(dl_valid)
# -

# # **Analyze weights**

weights.index = list(range(it+1))
weights.columns = ["wl1", "wl2", "wl3", "wl4", "wl5", "wl6", "wl7", "wl8"]
weights = weights.reset_index().rename(columns={"index": "iteration"})
weights = weights.melt(id_vars="iteration")

fig = px.line(weights, x="iteration", y="value", color="variable", line_group="variable")
fig.show()







first_labels[first_labels != second_labels]

Y_true[first_labels != second_labels]


