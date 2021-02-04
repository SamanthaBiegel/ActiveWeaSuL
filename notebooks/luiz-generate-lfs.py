# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

from scipy.stats import entropy
import plotly.express as px
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn import tree
from sklearn.tree import _tree, export_text

import matplotlib.pyplot as plt
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

from lfs.occupancy import labeling_functions

# Importing the methods from Samantha's work (I've changed the )
if '../ActiveWeaSuL' not in sys.path:
    sys.path.append('../ActiveWeaSuL')

# from activelearning.synthetic_data import SyntheticDataGenerator
# from activelearning.experiments import process_metric_dict, plot_metrics, active_weasul_experiment, process_exp_dict
# from activelearning.logisticregression import LogisticRegression
# from activelearning.discriminative_model import DiscriminativeModel

from activeweasul.label_model2 import LabelModel as WSLabelModel2
from activeweasul.label_model import LabelModel as WSLabelModel

# from activelearning.active_weasul import ActiveWeaSuLPipeline
# from activelearning.plot import plot_probs, plot_train_loss

# %% [markdown]
# ## Functions

# %%
def tree_to_code(tree, feature_names):
    """Convert a Descision Tree into a function (as string)"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = ["def tree(sample):"]

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
# ## Generate Labeling Functions
#
# We can used the output from the Random Forest to generate the labeling functions. Bellow, I coppied and pasted all the 10 decistion trees. We have to check if all of them are interesting and drop those that are not.
#
# **Note**: For each `return` there is a comment with the number of points under that leaf and the classification distribution (class 0 and class 1, respectively)

# %%
clf_occup = RandomForestClassifier(n_estimators=10, max_depth=3, bootstrap=False)
clf_occup.fit(df_occup.iloc[:, :-1], df_occup.iloc[:, -1])

# %%
# The list+set is used to drop duplicates
rules_occup = list(set([
    tree_to_code(estimator, df_occup.columns[:-1]) 
    for estimator in clf_occup.estimators_
]))

# %% [markdown]
# ## Evaluate the LFs
#
#

# %%
# # plot tree
# plt.figure(figsize=(70,12))  # set plot size (denoted in inches)
# tree.plot_tree(clf, fontsize=10)
# plt.savefig('/tmp/tree.png')

# %%
# Print the decision tree
# print(export_text(clf))

# %%
from operator import itemgetter 

# Drop lfs 2 and 6 (highly correlated)
lfs=itemgetter(*(0, 1, 3, 4, 5, 7, 8, 9))(labeling_functions)
# lfs=labeling_functions

applier = PandasLFApplier(lfs)
L_train = applier.apply(df_occup)

# %%
LFAnalysis(L=L_train, lfs=lfs).lf_summary()

# %%
class_balance = (df_occup.Occupancy.value_counts() / df_occup.shape[0]).values
cliques = [[i] for i in range(len(lfs))]
y = df_occup.Occupancy

# %% [markdown]
# ## Matrix completion

# %%
lm_mc = WSLabelModel2(n_epochs=200, lr=1e-1)
# Fit and predict on train set
y_mc = lm_mc.fit(label_matrix=L_train,
                 cliques=cliques,
                 class_balance=class_balance).predict().detach().numpy()

# %%
px.histogram(pd.Series(y_mc[:, 1]), marginal="violin")

# %%
get_metrics(df_occup.Occupancy, y_mc[:, 1] > 0.5, None)

# %% [markdown]
# ## Snorkel

# %%
lm_sn = LabelModel()
# Fit and predict on train set
lm_sn.fit(L_train,
          class_balance=class_balance)
y_sn = lm_sn.predict_proba(L_train)

# %%
px.histogram(pd.Series(y_sn[:, 1]), marginal="violin")

# %%
get_metrics(df_occup.Occupancy, y_sn[:, 1] > 0.5, None)

# %%
# Credit Card Dataset

# %%
df_card = pd.read_excel('../datasets/default of credit card clients.xls', index_col=0, skiprows=1)

# %%
clf_card = RandomForestClassifier(n_estimators=10, max_depth=2, bootstrap=False)
clf_card.fit(df_card.iloc[:, :-1], df_card.iloc[:, -1])

# %%
rules_card = list(set([
    tree_to_code(estimator, df_card.columns[:-1]) 
    for estimator in clf_card.estimators_
]))
