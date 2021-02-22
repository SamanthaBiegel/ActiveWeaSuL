# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup

# %% [markdown]
# ## Import packages

# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import sys
from operator import itemgetter
from IPython.display import display

from scipy.stats import entropy
import plotly.express as px
import seaborn as sns
import pyperclip

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_fscore_support
)
    
from sklearn import tree
from sklearn.tree import _tree, export_text
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

# Importing the methods from Samantha's work (I've changed the )
if '../ActiveWeaSuL' not in sys.path:
    sys.path.append('../ActiveWeaSuL')

# from activelearning.synthetic_data import SyntheticDataGenerator
# from activelearning.experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict
# from activelearning.logisticregression import LogisticRegression
# from activelearning.discriminative_model import DiscriminativeModel

from activeweasul.label_model import LabelModel as WSLabelModel
# from activeweasul.label_model_original import LabelModel as WSLabelModelOrig

# from activelearning.active_weasul import ActiveWeaSuLPipeline
# from activelearning.plot import plot_probs, plot_train_loss

# %% [markdown]
# ## Functions

# %%
def tree_to_code(tree, feature_names, fname='tree'):
    """Convert a Descision Tree into a function (as string)"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = [f"def {fname}(sample):"]

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = f"sample['{feature_name[node]}']"
            threshold = tree_.threshold[node]
            rules.append("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            rules.append("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            rules.append(f"{indent}# N. of samples {tree_.n_node_samples[node]} ({tree_.value[node][0]})")
            rules.append("{}return {}".format(indent, tree_.value[node].argmax()))

    recurse(0, 1)
    return "\n".join(rules)


# %%
def get_metrics(y, y_pred, weight):
    """Create a dataframe with the metrics"""
    return pd.DataFrame(
        np.vstack((
            [np.nan, accuracy_score(y, y_pred, sample_weight=weight)],
            [np.nan, matthews_corrcoef(y, y_pred, sample_weight=weight)],
            precision_recall_fscore_support(y, y_pred, sample_weight=weight)
        )), columns=['y0', 'y1'], 
        index=['acc', 'matthews_corr', 'precision', 'recall', 'f1', 'support']
    )


# %%
def plot_cov_o_inv(cov_o_inv, psi_wl, 
                   figsize=(6, 6), range_=None):
    min_, max_ = (-50, 100)

    fig, ax = plt.subplots(figsize=figsize)
    ticks = np.arange(0, cov_o_inv.shape[0], 1)
    tick_labels = np.zeros(cov_o_inv.shape[0], np.int16)
    for k, v in psi_wl.items():
        tick_labels[v] = int(k)
    plt.xticks(ticks, tick_labels)#ticks // 2)
    plt.yticks(ticks, tick_labels)#ticks // 2)
    values = (
        np.maximum(np.minimum(cov_o_inv, max_), min_)
        if range_ is not None else cov_o_inv
    )
    pos = ax.imshow(values)
    for (j, i), label in np.ndenumerate(cov_o_inv):
        ax.text(i, j, '{:.1f}'.format(label), ha='center', va='center')
    fig.colorbar(pos, ax=ax)
    
    
def get_cov_o_inv(lm, L, cliques):
    psi, psi_wl = lm.get_psi(
        label_matrix=L, 
        cliques=cliques
    )
    cov_o = np.cov(psi.T, bias=True)
    return np.linalg.pinv(cov_o), psi_wl


# %% [markdown]
# ## Download the data

# %%
download_data = False

if download_data:
    print("Downloading the data...")
    # !mkdir -p ../datasets/
    # Download Occupancy data
    # !wget -O ../datasets/occupancy_data.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip    
    # !unzip ../datasets/occupancy_data.zip -d ../datasets/occupancy_data/
    # Download Credit Card data
    # !wget -O '../datasets/default of credit card clients.xls' https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

# %% [markdown]
# # Occupancy Dataset

# %% [markdown]
# ## Load the data

# %%
df_occup = pd.read_csv('../datasets/occupancy_data/datatraining.txt')
df_occup['date'] = pd.to_datetime(df_occup['date']).astype(int)

# %% [markdown]
# ## Split into train and test

# %%
df_occup_tr, df_occup_ts = (
    train_test_split(df_occup, test_size=.2, stratify=df_occup.Occupancy, random_state=1234)
)

X_tr, y_tr = df_occup_tr.iloc[:, :-1], df_occup_tr.iloc[:, -1]
X_ts, y_ts = df_occup_ts.iloc[:, :-1], df_occup_ts.iloc[:, -1]

# %%
X_tr.index[-5:].tolist()

# %% [markdown]
# ## Generate Labeling Functions
#
# We can used the output from the Random Forest to generate the labeling functions. Bellow, I coppied and pasted all the 10 decistion trees. We have to check if all of them are interesting and drop those that are not.
#
# **Note**: For each `return` there is a comment with the number of points under that leaf and the classification distribution (class 0 and class 1, respectively)

# %%
clf_occup = RandomForestClassifier(n_estimators=20, max_depth=4, 
                                   bootstrap=False, max_leaf_nodes=5, 
                                   random_state=1234, max_features=1)
# A want "bad" LFs, so I'll fit to the test set 
# and validate on the training set
clf_occup.fit(X_ts, y_ts)

# %%
lf_metrics = []
for t in clf_occup.estimators_:
    pred = t.predict(X_tr)
    lf_metrics.append((
        accuracy_score(y_tr, pred), f1_score(y_tr, pred)
    ))
df_lf_metrics = (
    pd.DataFrame(lf_metrics, columns=['accuracy', 'f1'])
    .sort_values(['f1', 'accuracy'], ascending=False)
    .drop_duplicates()
)
display(df_lf_metrics.describe())
df_lf_metrics.tail(10)

# %%
estimators = itemgetter(
    *df_lf_metrics.tail(10).index
)(clf_occup.estimators_)

# The list+set is used to drop duplicates
rules_occup = [
    tree_to_code(estimator, df_occup.columns[:-1], 'lf{:02d}'.format(i)) 
    for i, estimator in enumerate(estimators, 1)
]

s_output = (
    'from snorkel.labeling import labeling_function\n\n' +
    '\n'.join(['@labeling_function()\n' + r + '\n' for r in rules_occup])
)

pyperclip.copy(s_output)
print(s_output)

# %% [markdown]
# ## Check for correlated LFs

# %%
import lfs.occupancy as lf_oc

# Apply the LFs
lfs = [
    lf_oc.lf01, lf_oc.lf02, lf_oc.lf03, lf_oc.lf04, lf_oc.lf05, 
    lf_oc.lf06, lf_oc.lf07, lf_oc.lf08, lf_oc.lf09, lf_oc.lf10
]

# 
applier = PandasLFApplier(lfs)
# L_train = applier.apply(df_occup)
L = applier.apply(df_occup)
# Contcatenate with y
L_y = np.hstack((L, df_occup.Occupancy.values[:, None]))

# # Compute the inverse of the cov. of O and plot
lm_mc = WSLabelModel(n_epochs=200, lr=1e-1)
# lm_mc.fit(L_train_y, 
#           cliques=[[i] for i  in range(L_train_y.shape[1])],
#           class_balance=df_card['default payment next month'].value_counts().values/df_card.shape[0])

# %%
cov_o_inv, psi_wl = get_cov_o_inv(
    lm_mc, L_y, cliques=[[0], [1], [2], [3], [4], 
                         [5], [6], [7], [8], [9], [10]]
)
plot_cov_o_inv(np.abs(cov_o_inv), psi_wl, figsize=(15,15), range_=(0, 200))

# %%
LFAnalysis(L=L, lfs=lfs).lf_summary(df_occup.Occupancy.values)

# %% [markdown]
# ### Selecting uncorrelated functions

# %%
class_balance = (df_occup.Occupancy.value_counts() / df_occup.shape[0]).values
cliques = [[1], [3], [6], [0], [2], [7], [8]]
# cliques = [[1, 4], [3, 5], [6, 9], [0], [2], [7], [8]]
# cliques = [[i] for i in range(len(lfs))]
y = df_occup.Occupancy

# %% [markdown]
# ## Matrix completion

# %%
lm_mc = WSLabelModel(n_epochs=600, lr=1e-1)
# Fit and predict on train set
y_mc = lm_mc.fit(label_matrix=L,
                 cliques=cliques,
                 class_balance=class_balance).predict().detach().numpy()

plt.plot(lm_mc.losses)
get_metrics(df_occup.Occupancy, y_mc[:, 1] > 0.5, None)

# %% [markdown]
# ## Snorkel

# %%
lm_sn = LabelModel()
# Fit and predict on train set
lm_sn.fit(L,
          class_balance=class_balance)
y_sn = lm_sn.predict_proba(L)

# %%
get_metrics(df_occup.Occupancy, y_sn[:, 1] > 0.5, None)

# %% [markdown]
# # Credit Card Dataset

# %% [markdown]
# ## Load the data

# %%
df_card = pd.read_excel('../datasets/default of credit card clients.xls', index_col=0, skiprows=1)

# %% [markdown]
# ## Generate Labeling Functions
#
# We can used the output from the Random Forest to generate the labeling functions. Bellow, I coppied and pasted all the 10 decistion trees. We have to check if all of them are interesting and drop those that are not.
#
# **Note**: For each `return` there is a comment with the number of points under that leaf and the classification distribution (class 0 and class 1, respectively)

# %%
X, y = df_card.iloc[:, :-1], df_card.iloc[:, -1]

clf_card = RandomForestClassifier(n_estimators=20, max_depth=4, 
#                                   min_samples_leaf=0.,
                                  bootstrap=False, max_leaf_nodes=4, 
                                  random_state=1234)
clf_card.fit(X, y)

# %%
lf_metrics = []
for t in clf_card.estimators_:
    pred = t.predict(X)
    lf_metrics.append((
        accuracy_score(y, pred), f1_score(y, pred)
    ))
df_lf_metrics = (
    pd.DataFrame(lf_metrics, columns=['accuracy', 'f1'])
    .sort_values(['f1', 'accuracy'], ascending=False)
    .drop_duplicates().head(10)
)
# display(df_lf_metrics.describe())
df_lf_metrics.head(10)

# %%
rules_card = sorted(list(
    tree_to_code(estimator, df_card.columns[:-1], 'lf{:02d}'.format(i)) 
    for i, estimator in enumerate(clf_card.estimators_, 1)
))

s_output = 'from snorkel.labeling import labeling_function\n\n'

pyperclip.copy(
    'from snorkel.labeling import labeling_function\n\n' + 
    '\n'.join(['@labeling_function()\n' + r + '\n' for r in rules_card])
)

# %% [markdown]
# ## Check for correlated LFs

# %%
import lfs.credit_card as lf_cc

# Apply the LFs
lfs = [
    lf_cc.lf01, lf_cc.lf02, lf_cc.lf03, lf_cc.lf04, lf_cc.lf05, 
    lf_cc.lf06, lf_cc.lf07, lf_cc.lf08, lf_cc.lf09, lf_cc.lf10
]

# 
applier = PandasLFApplier(lfs)
# L_train = applier.apply(df_occup)
L = applier.apply(df_card)
# Contcatenate with y
L_y = np.hstack((L, df_card['default payment next month'].values[:, None]))

# # Compute the inverse of the cov. of O and plot
# lm_mc = WSLabelModel(n_epochs=200, lr=1e-1)
# lm_mc.fit(L_train_y, 
#           cliques=[[i] for i  in range(L_train_y.shape[1])],
#           class_balance=df_card['default payment next month'].value_counts().values/df_card.shape[0])

# %%
cov_o_inv = get_cov_o_inv(lm_mc, L_y, cliques=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
plot_cov_o_inv(cov_o_inv, figsize=(15,15))

# %%
LFAnalysis(L=L_train, lfs=lfs).lf_summary(df_card['default payment next month'].values)

# %% [markdown]
# ### Selecting uncorrelated functions

# %%
class_balance = (df_card['default payment next month'].value_counts() / df_card.shape[0]).values
# cliques = [[2, 6, 8, 9], [0], [1], [3], [4], [5], [7], ]
cliques = [[i] for i in range(len(lfs))]
y = df_card['default payment next month']

# %% [markdown]
# ## Matrix completion

# %%
lm_mc = WSLabelModel(n_epochs=600, lr=1e-1)
# Fit and predict on train set
y_mc = lm_mc.fit(label_matrix=L_train,
                 cliques=cliques,
                 class_balance=class_balance).predict().detach().numpy()

plt.plot(lm_mc.losses)
get_metrics(df_card['default payment next month'], y_mc[:, 1] > 0.5, None)

# %% [markdown]
# ## Snorkel

# %%
lm_sn = LabelModel()
# Fit and predict on train set
lm_sn.fit(L_train,
          class_balance=class_balance)
y_sn = lm_sn.predict_proba(L_train)

# %%
get_metrics(df_card['default payment next month'], y_sn[:, 1] > 0.5, None)

# %%
