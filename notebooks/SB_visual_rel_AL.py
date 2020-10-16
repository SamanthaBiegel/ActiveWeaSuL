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
# -

torch.cuda.is_available()

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
#         "wear"
    ]

classify = ["sit on"]
df_train, df_test = load_vr_data(classify=classify, include_predicates=semantic_predicates, path_prefix=path_prefix, drop_duplicates=True, balance=balance, validation=False)

print("Train Relationships: ", len(df_train))
print("Test Relationships: ", len(df_test))

# +
# pd.set_option('display.max_rows',102)
# pd.DataFrame(df_train.groupby("y")["source_img"].count())
# -

df_train

# # **Define labeling functions**

SITON = 1
# WEAR = 1
OTHER = 0
ABSTAIN = -1


# +
def lf_siton_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bench", "chair", "floor", "horse", "grass", "table"]:
            return SITON
    return OTHER

def lf_not_person(x):
    if x.subject_category != "person":
        return OTHER
    return SITON


# +
# def lf_wear_object(x):
#     if x.subject_name == "person":
#         if x.object_name in ["t-shirt", "jeans", "glasses", "skirt", "pants", "shorts", "dress", "shoes"]:
#             return WEAR
#     return OTHER

# def lf_area(x):
#     if area(x.subject_bbox) / area(x.object_bbox) > 1:
#         return WEAR
#     return OTHER

# def lf_dist(x):
#     if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) >= 100:
#         return OTHER
#     return WEAR
# -

YMIN = 0
YMAX = 1
XMIN = 2
XMAX = 3


# +
def lf_ydist(x):
    if x.subject_bbox[YMAX] < x.object_bbox[YMAX] and x.subject_bbox[YMIN] < x.object_bbox[YMIN]:
        return SITON
    return OTHER

def lf_xdist(x):
    if x.subject_bbox[XMAX] < x.object_bbox[XMIN] or x.subject_bbox[XMIN] > x.object_bbox[XMAX]: 
        return OTHER
    return SITON

def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) >= 500:
        return OTHER
    return SITON

def area(bbox):
    return (bbox[YMAX] - bbox[YMIN]) * (bbox[XMAX] - bbox[XMIN])

def lf_area(x):
    if area(x.subject_bbox) / area(x.object_bbox) < 0.8:
        return SITON
    return OTHER


# +
# lfs = [lf_siton_object, lf_not_person, lf_ydist, lf_dist, lf_area]
lfs = [lf_siton_object, lf_dist, lf_area]
# lfs = [lf_wear_object, lf_dist, lf_area, lf_ydist]

L_train = apply_lfs(df_train, lfs)
# L_test = apply_lfs(df_test, lfs)
# -

analyze_lfs(L_train, df_train["y"], lfs)

# # **Initial fit label model**

# +
class_balance = np.array([1-df_train.y.mean(), df_train.y.mean()])

# cliques=[[0,1],[2,3],[4]]
cliques=[[0],[1,2]]


# -


lm_metrics = {}
for i in range(1):
    lm = LabelModel(df=df_train,
                    active_learning=False,
                    add_cliques=True,
                    add_prob_loss=False,
                    n_epochs=500,
                    lr=1e-1)

    Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
    lm.analyze()
    lm_metrics[i] = lm.metric_dict
    lm.print_metrics()

# +
# # %%time
# np.where((Y_probs_al[:,1].detach().numpy() == np.max(Y_probs_al[:,1].detach().numpy())) & (al.ground_truth_labels == -1) &~ al.all_abstain)[0]

# +
# # %%time
# [i for i, j in enumerate(Y_probs_al[:,1].detach().numpy()) if (j == np.max(Y_probs_al[:,1].detach().numpy())) and (al.ground_truth_labels[i] == -1) and not (al.all_abstain[i])]
# -

plot_train_loss(lm.losses)

metrics = ["accuracy", "precision", "recall", "f1"]
train_on = "probs" # probs or labels
n_epochs = 3
lr = 1e-3
batch_size=20

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'df': df_train,
             'n_epochs': 200,
             'batch_size': batch_size
            }

dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train,
                      Y=Y_probs.clone().detach().numpy())
dl_test = DataLoader(dataset, shuffle=False, batch_size=batch_size)


# +
# al_metrics = {}
# for i in range(50):
it = 20
query_strategy = "margin"

al = ActiveLearningPipeline(it=it,
#                             final_model=VisualRelationClassifier(pretrained_model, dl_test, df_train, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix),
                            **al_kwargs,
                            query_strategy=query_strategy,
                            randomness=0.2)

Y_probs_al = al.refine_probabilities(label_matrix=L_train, cliques=cliques, class_balance=class_balance)
al.label_model.print_metrics()
# -
al.plot_metrics(true_label_counts=False)

al.plot_iterations()

plot_train_loss(al.label_model.losses, "Epoch", "Label")

# +
# mean_metrics = pd.DataFrame.from_dict(lm_metrics, orient="index").mean().reset_index().rename(columns={"index": "Metric"})
# mean_metrics["std"] = pd.DataFrame.from_dict(lm_metrics, orient="index").sem().values
# mean_metrics["Active Learning"] = "before"

# mean_al_metrics = pd.DataFrame.from_dict(al_metrics, orient="index").mean().reset_index().rename(columns={"index": "Metric"})
# mean_al_metrics["std"] = pd.DataFrame.from_dict(al_metrics, orient="index").sem().values
# mean_al_metrics["Active Learning"] = "after"

# metrics_joined = pd.concat([mean_metrics, mean_al_metrics])

# +
# fig = px.bar(metrics_joined, x="Metric", y=0, error_y="std", color="Active Learning", barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel)
# fig.update_layout(template="plotly_white", yaxis_title="", title_text="Label model performance before and after active learning")
# fig.show()
# -

al.plot_true_vs_predicted_posteriors()

# +
# w_c = df_train.iloc[np.where((al.first_labels != al.df["y"]) & (al.second_labels == al.df["y"]))]

# +
# c_w = df_train.iloc[np.where((al.first_labels == al.df["y"]) & (al.second_labels != al.df["y"]))]
# -

lm.get_true_mu()

lf01 = L_train[:,[0,1]]
_, inv, count = np.unique(lf01, return_counts=True, return_inverse=True, axis=0)
inv[0]


(df_train.y.values[inv == 1] == 0).sum() / lm.N

lf23 = L_train[:,[2,3]]
_, inv, count = np.unique(lf23, return_counts=True, return_inverse=True, axis=0)
inv[0]

(df_train.y.values[inv == 0] == 0).sum() / lm.N

lf4 = L_train[:,4]
_, inv, count = np.unique(lf4, return_counts=True, return_inverse=True, axis=0)
inv[0]


(df_train.y.values[inv == 1] == 0).sum() / lm.N

0.2717391304347826*0.15489130434782608*0.057065217391304345/0.25/0.0571

0.3179347826086957*0.10869565217391304*0.22010869565217392/0.25/0.0571

import scipy
scipy.stats.binom_test(6,11,0.8)

L_train[0,:]

# +
lambda_combs, lambda_index, lambda_counts = np.unique(lm.label_matrix, axis=0, return_counts=True, return_inverse=True)
new_counts = lambda_counts.copy()
rows_not_abstain, cols_not_abstain = np.where(lambda_combs != -1)
for i, comb in enumerate(lambda_combs):
    nr_non_abstain = (comb != -1).sum()
    if nr_non_abstain < lm.nr_wl:
        if nr_non_abstain == 0:
            new_counts[i] = 0
        else:
            match_rows = np.where((lambda_combs[:, cols_not_abstain[rows_not_abstain == i]] == lambda_combs[i, cols_not_abstain[rows_not_abstain == i]]).all(axis=1))       
            new_counts[i] = lambda_counts[match_rows].sum()

P_lambda = torch.Tensor((new_counts/lm.N)[lambda_index][:, None])
# -

df_train.iloc[lambda_index == 17]

# +
# P_lambda[lm.predict_true()[:,1] > 1]
# -

0.035326086956521736*0.10869565217391304*0.22010869565217392/0.25/0.0054

(0.057065217391304345*0.29347826086956524*0.2717391304347826)/0.25/0.0054

# +
lambda_combs, lambda_index, lambda_counts = np.unique(np.concatenate([lm.label_matrix,df_train.y.values[:,None]],axis=1), axis=0, return_counts=True, return_inverse=True)

P_Y_lambda = np.zeros((lm.N, 2))

P_Y_lambda[df_train.y.values == 0, 0] = ((lambda_counts/lm.N)[lambda_index]/P_lambda.squeeze())[df_train.y.values == 0]
P_Y_lambda[df_train.y.values == 0, 1] = 1 - P_Y_lambda[df_train.y.values == 0, 0]

P_Y_lambda[df_train.y.values == 1, 1] = ((lambda_counts/lm.N)[lambda_index]/P_lambda.squeeze())[df_train.y.values == 1]
P_Y_lambda[df_train.y.values == 1, 0] = 1 - P_Y_lambda[df_train.y.values == 1, 1]                                                               
# -

P_Y_lambda[0]

lm._analyze(torch.Tensor(P_Y_lambda), df_train.y.values)

lm.print_metrics()

# +
true_probs = lm.predict_true()[:, 1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=P_Y_lambda[:,1], y=true_probs, mode='markers', showlegend=False, marker_color=np.array(px.colors.qualitative.Pastel)[0]))
fig.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100), line=dict(dash="longdash", color=np.array(px.colors.qualitative.Pastel)[1]), showlegend=False))

fig.update_yaxes(range=[0, 1], title_text="True from Junction Tree ")
fig.update_xaxes(range=[0, 1], title_text="True from P(Y, lambda)")
fig.update_layout(template="plotly_white", width=1000, height=500)
fig.show()
# -

fig = go.Figure(go.Scatter(x=P_lambda.squeeze(), y=np.array(true_probs)-P_Y_lambda[:,1], mode="markers"))
fig.update_layout(template="plotly_white", xaxis_title="P(lambda)", title_text="Deviation true and junction tree posteriors")
fig.show()

df_train[lambda_index == 9].y.mean()

lm.predict_true()[13]

(lf0[df_train.y == 1] == 0).sum() / lm.N

(df_train.y[lf0 == 0] == 1).sum() / lm.N

(lf0[df_train.y == 1] == 1).sum() / lm.N

lm.mu

w_c[w_c.y == 0]

w_c[w_c.y == 1]

c_w

al.color_df()

al.plot_metrics()

al.plot_parameters()

al.label_model.get_true_mu()[8:18,1]

lm.cov_O

al.color_cov()

df_train[df_train.y == 1]

lm.metric_dict

lm.metric_dict

al.plot_metrics()

al.plot_parameters([1,3,5,7,9])



# +
# al.label_model.get_true_mu()[8:18,1]

# +
# al.color_cov()

# +
# pretrained_model = torch.load("../models/resnet.pth")

# +
# torch.save(pretrained_model, "../models/resnet.pth")
# -

# # **Train discriminative model on probabilistic labels**

metrics = ["accuracy", "precision", "recall", "f1"]
train_on = "probs" # probs or labels
batch_size = 20
n_epochs = 3
lr = 1e-3

# +
# cliques = [[0,1],[1,3],[2,3],[3,4],[0,5],[1,5],[2,5],[4,5]]

# +
# al.label_model.cliques

# +
# cliques

# +
# L_train[al.queried[10:19]]

# +
# df_train.iloc[al.queried[10:19]]

# +
# al.plot_parameters()

# +
# al.plot_metrics()
# -

# # **Train discriminative model on probabilistic labels**

df_train[df_train["object_category"] == "cake piece"]

# +
n_epochs = 10
lr=1e-2

dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train,
                      Y=Y_probs.clone().clamp(0,1).detach().numpy())

dl = DataLoader(dataset, shuffle=True, batch_size=batch_size)
dl_test = DataLoader(dataset, shuffle=False, batch_size=batch_size)

vc = VisualRelationClassifier(pretrained_model, dl_test, df_train, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix)

probs_final = vc.fit(dl).predict()

vc.analyze()

vc.print_metrics()
# -

from plot import plot_train_loss
plot_train_loss(vc.average_losses, "Batches", model="Discriminative")

# +
# dataset = VisualRelationDataset(image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
#                       df=df_train[:20], 
#                       Y=Y_probs.detach()[:20])

# dl_test = DataLoader(dataset, shuffle=False, batch_size=20)

# +
dataset_al = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train, 
                      Y=Y_probs_al.clone().clamp(0,1).detach().numpy())

dl_al = DataLoader(dataset_al, shuffle=True, batch_size=batch_size)
dl_al_test = DataLoader(dataset_al, shuffle=False, batch_size=batch_size)

vc_al = VisualRelationClassifier(pretrained_model, dl_al_test, df_train, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix)

probs_final_al = vc_al.fit(dl_al).predict()

vc_al.analyze()

vc_al.print_metrics()
# -

fig = go.Figure(go.Scatter(x=list(range(len(vc_al.average_losses))), y=vc_al.average_losses))
fig.update_layout(xaxis_title="Batch", yaxis_title="Loss", title_text="Final model - Training Loss", template="plotly_white")

# +
dataset_al = VisualRelationDataset(image_dir=path_prefix + "data/VRD/sg_dataset/sg_train_images", 
                      df=df_train, 
                      Y=lm.predict_true().clone().detach().numpy())

dl_al = DataLoader(dataset_al, shuffle=True, batch_size=batch_size)
dl_al_test = DataLoader(dataset_al, shuffle=False, batch_size=batch_size)

vc_true = VisualRelationClassifier(pretrained_model, dl_al, dl_al_test, df_train, n_epochs=n_epochs, lr=lr)

probs_final_true = vc_true.fit().predict()

vc_true.analyze()

vc_true.print_metrics()
# -

al.label_model._analyze(Y_probs, al.y)




