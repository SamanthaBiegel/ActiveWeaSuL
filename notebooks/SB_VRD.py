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
# #     ! pip install -r ../requirements.txt
    # ! aws s3 cp s3://user/gc03ye/uploads/VRD/ /tmp/data/annotations --recursive --exclude "sg_dataset*"
# #     ! aws s3 cp s3://user/gc03ye/uploads/VRD/sg_dataset/sg_train_images/ /tmp/data/images/train_images --recursive
# #     ! aws s3 cp s3://user/gc03ye/uploads/VRD/sg_dataset/sg_test_images/ /tmp/data/images/test_images --recursive
# #     ! aws s3 cp s3://user/gc03ye/uploads/glove /tmp/data/word_embeddings --recursive
# #     ! aws s3 cp s3://user/gc03ye/uploads/resnet_old.pth /tmp/models/resnet_old.pth
    import torch
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
from discriminative_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveWeaSuLPipeline
from vr_utils import load_vr_data, balance_dataset, df_drop_duplicates
from lf_utils import apply_lfs, analyze_lfs
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
# -

df_train

# +
# df_test.to_csv("datasets/vrd_test_set.csv", index=False)

# df_train = pd.read_csv("datasets/vrd_train_set.csv")
# df_test = pd.read_csv("datasets/vrd_test_set.csv")

# +
# pd.set_option('display.max_rows',102)
# pd.DataFrame(df_train.groupby("y")["source_img"].count())
# -

df_train.mean()

df_train["subject_bbox"][0]

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
lfs = [lf_siton_object, lf_not_person, lf_ydist, lf_dist, lf_area]
# lfs = [lf_siton_object, lf_dist, lf_area]
# lfs = [lf_wear_object, lf_dist, lf_area, lf_ydist]

L_train = apply_lfs(df_train, lfs)
L_test = apply_lfs(df_test, lfs)
# -

analyze_lfs(L_train, df_train["y"], lfs)

analyze_lfs(L_test, df_test["y"], lfs)

# +
from snorkel.labeling.analysis import LFAnalysis

LFAnalysis(L_train).lf_summary()
# -

# # **Initial fit label model**

# +
class_balance = np.array([1-df_train.y.mean(), df_train.y.mean()])

cliques=[[0,1],[2,3],[4]]
# cliques=[[0],[1,2]]


# -


lm_metrics = {}
for i in range(1):
    lm = LabelModel(y_true=df_train.y.values,
                    n_epochs=500,
                    lr=1e-1)

    Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
    lm.analyze(df_train.y.values)
#     lm_metrics[i] = lm.metric_dict

# +
# # %%time
# np.where((Y_probs_al[:,1].detach().numpy() == np.max(Y_probs_al[:,1].detach().numpy())) & (al.ground_truth_labels == -1) &~ al.all_abstain)[0]

# +
# # %%time
# [i for i, j in enumerate(Y_probs_al[:,1].detach().numpy()) if (j == np.max(Y_probs_al[:,1].detach().numpy())) and (al.ground_truth_labels[i] == -1) and not (al.all_abstain[i])]
# -

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

# +
dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/test_images", 
                      df=df_test,
                      Y=df_test["y"].values)
dl_test = DataLoader(dataset, shuffle=False, batch_size=batch_size)

dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train,
                      Y=df_train["y"].values)
dl_train = DataLoader(dataset, shuffle=False, batch_size=batch_size)


# +
dataset.Y = Y_probs.clone().detach().numpy()

dl = DataLoader(dataset, shuffle=True, batch_size=batch_size)
# -

dl_test.s

dl_test

# +
al_metrics = {}
al_metrics["lm_metrics"] = {}
al_metrics["lm_test_metrics"] = {}
al_metrics["fm_metrics"] = {}
al_metrics["fm_test_metrics"] = {}

for i in range(10):
    it = 50
    query_strategy = "relative_entropy"

    al = ActiveLearningPipeline(it=it,
                                final_model=VisualRelationClassifier(pretrained_model, df_train, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix),
                                **al_kwargs,
                                image_dir=path_prefix + "data/images/train_images",
                                query_strategy=query_strategy,
                                randomness=0)

    Y_probs_al = al.refine_probabilities(label_matrix=L_train, cliques=cliques, class_balance=class_balance,
                                         label_matrix_test=L_test, y_test=df_test["y"].values, dl_train=dl_train, dl_test=dl_test)
    al.label_model.print_metrics()
    al_metrics["lm_metrics"][i] = al.metrics
    al_metrics["lm_test_metrics"][i] = al.test_metrics
#     al_metrics["fm_metrics"][i] = al.final_metrics
#     al_metrics["fm_test_metrics"][i] = al.final_test_metrics
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
# metrics_joined = pd.concat([create_metric_df(al_metrics, 4, "lm_metrics", "train", "Generative"),
#                            create_metric_df(al_metrics, 4, "lm_test_metrics", "test", "Generative"),
#                            create_metric_df(al_metrics, 4, "fm_metrics", "train", "Discriminative"),
#                            create_metric_df(al_metrics, 4, "fm_test_metrics", "test", "Discriminative"),
#                            pd.read_csv("results/vrd_incl_optimal.csv")])
# metrics_joined_test = pd.read_csv("results/vrd_incl_optimal_10_trials.csv")


metrics_joined_re = pd.concat([create_metric_df(al_metrics, 10, "lm_metrics", "train", "Generative"),
                           create_metric_df(al_metrics, 10, "lm_test_metrics", "test", "Generative")])

metrics_joined_marg = pd.concat([create_metric_df(al_metrics, 10, "lm_metrics", "train", "Generative"),
                           create_metric_df(al_metrics, 10, "lm_test_metrics", "test", "Generative")])

metrics_joined_random = pd.concat([create_metric_df(al_metrics, 10, "lm_metrics", "train", "Generative"),
                           create_metric_df(al_metrics, 10, "lm_test_metrics", "test", "Generative")])

metrics_joined_re["Strategy"] = "MaxKL"
metrics_joined_marg["Strategy"] = "Margin"

metrics_joined = pd.concat([metrics_joined_re, metrics_joined_marg])
# +
# metrics_joined.to_csv("results/vrd_incl_optimal_10_trials.csv", index=False)
# -

sns.set_theme(style="white")
colors = ["#086788",  "#e3b505","#ef7b45",  "#739e82", "#d88c9a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))

al.label_model._analyze(true_test, df_test.y.values)

al.label_model.predict_true()

lm.predict_true()

lm_train = al.label_model._analyze(al.label_model.predict_true(), df_train["y"].values)
# lm_test = al.label_model._analyze(true_test, df_test.y.values)
psi_test, _ = al.label_model._get_psi(L_test, cliques, len(lfs))
lm_test = al.label_model._analyze(al.label_model._predict(L_test, psi_test, al.label_model.get_true_mu()[:, 1][:, None], df_train.y.mean()), df_test.y)

fm_train = {'MCC': 0.5929464302605295, 'Precision': 0.7248322147651006, 'Recall': 0.627906976744186, 'Accuracy': 0.8677581863979849}
fm_test = {'MCC': 0.5378129202989759, 'Precision': 0.7096774193548387, 'Recall': 0.55, 'Accuracy': 0.8540540540540541}

fm_train_full = {'MCC': 0.7963334186630372, 'Precision': 0.7745098039215687, 'Recall': 0.9186046511627907, 'Accuracy': 0.924433249370277}
fm_test_full = {'MCC': 0.7447663283072967, 'Precision': 0.72, 'Recall': 0.9, 'Accuracy': 0.9027027027027027}

metrics_joined["Label"] = "AL"


# +
def create_optimal_df(perf_dict, model_string, set_string, label_string):

    optimal_lm = pd.DataFrame({"Accuracy": np.repeat(perf_dict["Accuracy"], 51), "MCC": np.repeat(perf_dict["MCC"], 51), "Precision": np.repeat(perf_dict["Precision"], 51), "Recall": np.repeat(perf_dict["Recall"], 51)})
    optimal_lm = optimal_lm.stack().reset_index().rename(columns={"level_0": "Active Learning Iteration", "level_1": "Metric", 0: "Value"})
    optimal_lm["Model"] = model_string
    optimal_lm["Set"] = set_string
    optimal_lm["Label"] = label_string
    optimal_lm["Run"] = 0
    
    return optimal_lm

optimals_df = pd.concat([create_optimal_df(lm_train, "Generative", "train", "*"),
           create_optimal_df(lm_test, "Generative", "test", "*")])
#           create_optimal_df(fm_train, "Discriminative", "train", "*"),
#           create_optimal_df(fm_test, "Discriminative", "test", "*"),
#           create_optimal_df(fm_train_full, "Discriminative", "train", "**"),
#           create_optimal_df(fm_test_full, "Discriminative", "test", "**")])
# -

metrics_joined_optimal = pd.concat([metrics_joined, optimals_df])

metrics_joined = pd.read_csv("results/vrd_incl_optimal_10_trials.csv")

ax = sns.relplot(data=metrics_joined_optimal, x="Active Learning Iteration", y="Value", col="Metric", kind="line", ci=68, n_boot=1000, hue="Set", style="Label",legend=True)
(ax.set_titles("{col_name}"))

optimals_df

metrics_joined

# +
colors = ["#2b4162", "#721817", "#e9c46a", "#fa9f42", "#0b6e4f", "#96bdc6",  "#c09891", "#5d576b", "#c6dabf", "#368f8b", "#ec7357"]

pick_colors = [colors[9], colors[10]]

metrics_joined_optimal = metrics_joined_optimal.rename(columns={"Active Learning Iteration": "Number of labeled points"})

sns.set(style="whitegrid")
ax = sns.relplot(data=metrics_joined_optimal, x="Number of labeled points", y="Value", col="Metric", row="Set",
                 kind="line", ci=68, n_boot=100, hue="Strategy", style="Label",legend=False, palette=sns.color_palette(pick_colors))

show_handles = [ax.axes[0][0].lines[0], ax.axes[0][0].lines[1]]
show_labels = ["MaxKL", "Margin"]
ax.axes[1][3].legend(handles=show_handles, labels=show_labels, loc="lower right")

ax.set_ylabels("")
ax.set_titles("{col_name}")
plt.show()
# fig = ax.get_figure()
# ax.savefig("plots/vrd_metrics.png")
# +
it = 30
query_strategy = "relative_entropy"

al = ActiveLearningPipeline(it=it,
#                             final_model=VisualRelationClassifier(pretrained_model, dl_test, df_train, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix),
                            **al_kwargs,
                            image_dir=path_prefix + "data/images/train_images",
                            query_strategy=query_strategy,
                            randomness=0)

Y_probs_al = al.refine_probabilities(label_matrix=L_train, cliques=cliques, class_balance=class_balance, label_matrix_test=L_test, y_test=df_test["y"].values)
al.label_model.print_metrics()
# -



n_epochs=1
batch_size=20

# +
dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train,
                      Y=Y_probs.clone().clamp(0,1).detach().numpy())
dl = DataLoader(dataset, shuffle=True, batch_size=batch_size)

vc = VisualRelationClassifier(pretrained_model=pretrained_model, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix)

vc = vc.fit(dl_train)

probs_final = vc.predict()

probs_final_test = vc.predict(dl_test)

print(vc.analyze(df_train.y.values))
print(vc.analyze(df_test.y.values, probs_final_test))
# -

al.label_model.predict_true()

lm._predict(L_test, psi_test, lm.mu, df_test.y.mean())

# +
lambda_combs, lambda_index, lambda_counts = np.unique(np.concatenate([L_test, df_test.y.values[:, None]], axis=1), axis=0, return_counts=True, return_inverse=True)

P_Y_lambda = np.zeros((L_test.shape[0], 2))

for i, j in zip([0, 1], [1, 0]):
    P_Y_lambda[df_test.y.values == i, i] = ((lambda_counts/L_test.shape[0])[lambda_index]/lm.P_lambda.squeeze())[df_test.y.values == i]
    P_Y_lambda[df_test.y.values == i, j] = 1 - P_Y_lambda[df_test.y.values == i, i]

true_test = torch.Tensor(P_Y_lambda)
# -



# +
dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train,
                      Y=al.label_model.predict_true().detach().numpy())
dl = DataLoader(dataset, shuffle=True, batch_size=batch_size)

vc_al = VisualRelationClassifier(pretrained_model, df_train, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix, soft_labels=True)

vc_al = vc_al.fit(dl)

probs_final_al = vc_al._predict(dl_train)

probs_final_al_test = vc_al._predict(dl_test)

print(vc_al._analyze(probs_final_al, df_train["y"].values))
print(vc_al._analyze(probs_final_al_test, df_test["y"].values))
# -

print(vc_al._analyze(probs_final_al, df_train["y"].values))
print(vc_al._analyze(probs_final_al_test, df_test["y"].values))

probs_final_al

Y_probs_al

dl.shuffle

plot_train_loss([loss.cpu().detach().numpy() for loss in vc_al.losses])

plot_train_loss(vc_al.average_losses)





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
probs_final_test = vc_al._predict(dl_al_test)

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



# ## Active learning

random.seed(None)
df_1 = df_train.iloc[random.sample(range(len(df_train)),2)]

random.sample(range(len(df_train)),2)

# +
initial_seed = random.sample(range(len(df_train)),80)
df_1 = df_train.iloc[initial_seed]

labels = df_1.y.values
df_1.index = range(len(df_1))

dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_1,
                      Y=labels)

dl = DataLoader(dataset, shuffle=True, batch_size=10)
vc = VisualRelationClassifier(pretrained_model, df_1, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix, soft_labels=False)

probs_final = vc.fit(dl)._predict(dl_train)
# probs_final_test = vc._predict(dl_test)

vc.analyze()

vc.print_metrics()
# -

vc._analyze(probs_final, df_train.y.values)

# +
from tqdm import tqdm_notebook as tqdm


dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/test_images", 
                          df=df_test,
                          Y=df_test["y"].values)
dl_test = DataLoader(dataset, shuffle=False, batch_size=batch_size)

dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                      df=df_train,
                      Y=df_train["y"].values)
dl_train = DataLoader(dataset, shuffle=False, batch_size=100)


accuracy_dict = {}
for j in tqdm(range(1)):
    accuracies = []
    accuracies_test = []
    
    random.seed(None)
    initial_seed = random.sample(range(len(df_train)),2)
    df_1 = df_train.iloc[initial_seed]
    print(df_1)
    queried = initial_seed
    
    labels = df_1.y.values
    df_1.index = range(len(df_1))

    dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                          df=df_1,
                          Y=labels)

    dl = DataLoader(dataset, shuffle=True, batch_size=len(queried))
    vc = VisualRelationClassifier(pretrained_model, df_1, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix, soft_labels=False)

    probs_final = vc.fit(dl)._predict(dl_train)
#     probs_final_test = vc._predict(dl_test)

    accuracy = vc._analyze(probs_final, df_train.y.values)["Accuracy"]
    print(accuracy)

    accuracies.append(accuracy)
#     accuracies_test.append(vc._analyze(probs_final_test, df_test.y.values)["Accuracy"])

    for i in tqdm(range(30,60)):
        dist_boundary = torch.abs(probs_final[:, 1] - probs_final[:, 0])
        dist_boundary[queried] = 1
        queried.append(torch.argmin(dist_boundary).item())

        df_1 = df_train.iloc[queried]

        labels = df_1.y.values
        df_1.index = range(len(df_1))

        dataset = VisualRelationDataset(image_dir=path_prefix + "data/images/train_images", 
                          df=df_1,
                          Y=labels)

        dl = DataLoader(dataset, shuffle=True, batch_size=len(queried))
        vc = VisualRelationClassifier(pretrained_model, df_1, n_epochs=n_epochs, lr=lr, data_path_prefix=path_prefix, soft_labels=False)

        probs_final = vc.fit(dl)._predict(dl_train)
    #             probs_final_test = vc._predict(dl_test)

        accuracy = vc._analyze(probs_final, df_train.y.values)["Accuracy"]
        print(accuracy)

        accuracies.append(accuracy)
        #         accuracies_test.append(vc._analyze(probs_final_test, df_test.y.values)["Accuracy"])

#         accuracy_dict[j] = accuracies

# +
accuracy_df = pd.DataFrame.from_dict(accuracy_dict)
accuracy_df = accuracy_df.stack().reset_index().rename(columns={"level_0": "Active Learning Iteration", "level_1": "Run", 0: "Accuracy"})

accuracy_df["Metric"] = "Accuracy"
accuracy_df["Strategy"] = "Active Learning"
accuracy_df["Model"] = "Discriminative"

# +
accuracy_df = accuracy_df.rename(columns={"Active Learning Iteration": "Number of labeled points"})

colors = ["#2b4162", "#ec7357", "#368f8b", "#2b4162", "#e9c46a", "#721817", "#fa9f42", "#0b6e4f", "#96bdc6",  "#c09891", "#5d576b", "#c6dabf"]
sns.set(style="whitegrid", palette=sns.color_palette(colors), rc={'figure.figsize':(15,10)})
ax = sns.lineplot(data=accuracy_df, x="Number of labeled points", y="Accuracy", hue="Strategy", ci=68, legend=False)
show_handles = [ax.axes.lines[0], ax.axes.lines[1]]
show_labels = ["AWSL", "Active Learning"]
ax.axes.legend(handles=show_handles, labels=show_labels, loc="lower right")
# plt.show()

fig = ax.get_figure()
# -

df_train.iloc[queried]

np.unique(queried)

df_train.mean()

probs_final


