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
#     display_name: 'Python 3.7.6 64-bit (''thesis'': conda)'
#     language: python
#     name: python37664bitthesiscondab786fa2bf8ea4d8196bc19b4ba146cff
# ---

# +
# %load_ext autoreload
# %autoreload 2

import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

sys.path.append(os.path.abspath("../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_train_loss
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -

L_train = np.load("../data/spouse/L_train_spouse.npy")
L_dev = np.load("../data/spouse/L_dev_spouse.npy")
y_dev = np.load("../data/spouse/Y_dev_spouse.npy")

class_balance = np.array([0.93,0.07])
cliques = [[0],[1],[2],[3],[4],[5],[6],[7],[8]]

df = pd.DataFrame(data=y_dev, columns=["y"])

# +
# L_train[L_train == -1] = 0
# L_dev[L_dev == -1] = 0

# +
lm = LabelModel(df=df,
                active_learning=False,
                add_cliques=True,
                add_prob_loss=False,
                n_epochs=200,
                lr=1e-1)

Y_probs = lm.fit(label_matrix=L_train, cliques=cliques, class_balance=class_balance).predict()
Y_probs_dev = lm._predict(L_dev, lm.mu, 0.07)
lm._analyze(Y_probs_dev, y_dev)
# -

lm.mu

lm.P_lambda[:6]



0.0116*0.0039/0.0896

L_train[:6]

Y_probs[:30]

unique_probs.clip(0,1)

al_kwargs = {'add_prob_loss': False,
             'add_cliques': True,
             'active_learning': "probs",
             'df': df,
             'n_epochs': 200
            }

# +
# it = 30
# query_strategy = "relative_entropy"

# al = ActiveLearningPipeline(it=it,
#                             **al_kwargs,
#                             query_strategy=query_strategy,
#                             randomness=0)

# Y_probs_al = al.refine_probabilities(label_matrix=L_train, cliques=cliques, class_balance=class_balance, label_matrix_test=L_dev, y_test=y_dev)
# al.label_model.print_metrics()
# -

probs=Y_probs.clone().detach().numpy()[:, 1]
probs = probs[~np.isnan(probs)]
unique_probs, unique_idx, unique_inverse = np.unique(probs, return_index=True, return_inverse=True)

randomness=0
N = len(L_train)
nr_wl = L_train.shape[1]

ground_truth_labels = np.full_like(L_train[:,0], -1)
all_abstain = (L_train == -1).sum(axis=1) == nr_wl

# +
prob_dict = {}
unique_prob_dict = {}

prob_dict[0] = Y_probs[:, 1].clone().detach().numpy()
unique_prob_dict[0] = prob_dict[0][unique_idx]


# -

def relative_entropy(iteration):
        
    lm_posteriors = unique_prob_dict[iteration]
    lm_posteriors = np.concatenate([1-lm_posteriors[:, None], lm_posteriors[:, None]], axis=1).clip(1e-5, 1-1e-5)

    rel_entropy = np.zeros(len(lm_posteriors))
    sample_posteriors = np.zeros(lm_posteriors.shape)

    for i in range(len(lm_posteriors)):
        bucket_items = ground_truth_labels[np.where(unique_inverse == i)[0]]
        bucket_gt = bucket_items[bucket_items != -1]
        bucket_gt = np.array(list(bucket_gt) + [np.round(unique_prob_dict[0][i])])

        eps = 1e-2/(len(bucket_gt))
        sample_posteriors[i, 1] = bucket_gt.mean().clip(eps, 1-eps)

        sample_posteriors[i, 0] = 1 - sample_posteriors[i, 1]

        rel_entropy[i] = entropy(lm_posteriors[i, :], sample_posteriors[i, :])

        if -1 not in list(bucket_items):
            rel_entropy[i] = 0

    max_buckets = np.where(rel_entropy == np.max(rel_entropy))[0]

    random.seed(None)
    pick_bucket = random.choice(max_buckets)

    bucket_values = rel_entropy

    return np.where((unique_inverse == pick_bucket) & (ground_truth_labels == -1) &~ all_abstain)[0]


def query(probs, iteration):
    """Choose data point to label from label predictions"""

    random.seed(None)
    pick = random.uniform(0, 1)

    if pick < randomness:
        indices = [i for i in range(N) if ground_truth_labels[i] == -1 and not all_abstain[i]]

    elif query_strategy == "relative_entropy":
        indices = relative_entropy(iteration)

    else:
        logging.warning("Provided active learning strategy not valid, setting to margin")
        self.query_strategy = "margin"
        return self.query(probs, iteration)

    random.seed(None)

    return random.choice(indices)


from scipy.stats import entropy

sel_idx = query(Y_probs, 0)

ground_truth_labels[sel_idx] = 1

ground_truth_labels[340] = 1

# +
lm = LabelModel(df=df,
                    active_learning="probs",
                    add_cliques=True,
                    add_prob_loss=False,
                    n_epochs=200,
                    lr=1e-1)

Y_probs = lm.fit(label_matrix=L_train,
                   cliques=cliques, class_balance=class_balance, ground_truth_labels=ground_truth_labels).predict()
Y_probs_dev = lm._predict(L_dev, lm.mu, 0.07)
lm._analyze(Y_probs_dev, y_dev)
# -

sel_idx = query(Y_probs, 1)

ground_truth_labels[3719] = 1

Y_probs = lm.fit(label_matrix=L_train,
                   cliques=cliques, class_balance=class_balance, ground_truth_labels=ground_truth_labels).predict()
Y_probs_dev = lm._predict(L_dev, lm.mu, 0.07)
lm._analyze(Y_probs_dev, y_dev)

Y_probs






