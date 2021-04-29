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

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.models as models

sys.path.append(os.path.abspath("../activeweasul"))
from synthetic_data import SyntheticDataGenerator
from logisticregression import LogisticRegression
from discriminative_model import DiscriminativeModel
from label_model import LabelModel
from active_weasul import ActiveWeaSuLPipeline, set_seed, CustomTensorDataset
from experiment_utils import process_metric_dict, active_weasul_experiment, process_exp_dict, process_entropies, add_weak_supervision_baseline, synthetic_al_experiment, active_learning_experiment
from visualrelation import VisualRelationDataset, VisualRelationClassifier
from vr_utils import load_vr_data
from lf_utils import apply_lfs, analyze_lfs
# -

# # Figure 3: Artifical Data

# ## Generating data

# We first create an artificial dataset with 2 classes of equal size. We generate a training set that contains 10.000 points and a test set that contains 3.000 points.

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])

p_z = 0.5

set_seed(932)
data = SyntheticDataGenerator(N, p_z, centroids)
df_train = data.sample_dataset().create_df()
y_train = df_train.y.values

# +
N = 3000

set_seed(466)
data = SyntheticDataGenerator(N, p_z, centroids)
df_test = data.sample_dataset().create_df()
y_test = df_test.y.values

# +
cmap = clr.LinearSegmentedColormap.from_list('', ['#368f8b',"#BBBBBB",'#ec7357'], N=200)

plt.figure(figsize=(5,5))
plt.scatter(x=df_train.x1, y=df_train.x2, c=y_train, s=50, edgecolor="black", cmap=cmap)
plt.xlim(-3.6,2.9)
plt.ylim(-3,3.5)
plt.show()
# -

# ## Creating labelling functions

# We apply 3 labelling functions to the dataset. Each labeling function assigns labels based on the value of one of the features. For example, the first labelling function assigns class 1 if $x_2$ is smaller than 0.4. We put them together in a label matrix that has a column for each labelling function.

# +
df_train.loc[:, "wl1"] = (df_train.x2<0.4)*1
df_train.loc[:, "wl2"] = (df_train.x1<-0.3)*1
df_train.loc[:, "wl3"] = (df_train.x1<-1)*1

label_matrix_train = df_train[["wl1", "wl2", "wl3"]].values

# +
df_test.loc[:, "wl1"] = (df_test.x2<0.4)*1
df_test.loc[:, "wl2"] = (df_test.x1<-0.3)*1
df_test.loc[:, "wl3"] = (df_test.x1<-1)*1

label_matrix_test = df_test[["wl1", "wl2", "wl3"]].values
# -

# ## Running experiments

# We set up experiments to compare the performance of Active WeaSuL to other approaches, such as only active learning. Each experiments runs 10 trials, where every trial starts from the same weak supervision setting at iteration 0 before going through the Active WeaSuL pipeline.

# +
# Experiment settings
nr_trials = 10
al_it = 30

# Active WeaSuL settings
starting_seed = 36
penalty_strength = 1

# Label model settings
class_balance = np.array([1 - p_z, p_z])
cliques = [[0], [1, 2]]

# Discriminative model settings
batch_size = 256
discriminative_model_kwargs = dict(input_dim=2,
                          output_dim=2,
                          lr=1e-1,
                          n_epochs=100)
# -

# Plotting settings
font_size = 25
legend_size = 25
tick_size = 20
n_boot = 10000
linewidth = 4

# +
# Collect all experiment parameters

# Common
exp_kwargs = dict(nr_trials=nr_trials,
                  al_it=al_it,
                  y_train=y_train,
                  batch_size=batch_size,
                  y_test=y_test)

# Active WeaSuL
aw_exp_kwargs = dict(label_matrix=label_matrix_train,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=starting_seed,
                  penalty_strength=penalty_strength,
                  discriminative_model=LogisticRegression(**discriminative_model_kwargs, early_stopping=True, patience=5),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1", "x2"]].values), Y=y_train),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1", "x2"]].values), Y=y_test),
                  label_matrix_test=label_matrix_test)
# -

# ### Active WeaSuL

np.random.seed(284)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl, entropies_maxkl = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="maxkl")

# ### Nashaat et al.

# Now we run the experiment on our artificial dataset for our implementation of the approach by Nashaat et al.

np.random.seed(25)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat, _ = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="nashaat")

# ### Increase number of active learning iterations

# Since the impact on performance across iterations is not clearly visible during the first 30 iterations for Nashaat et al., we run Active WeaSuL and the Nashaat et al. method for up to 1000 active learning iterations.

exp_kwargs["al_it"] = 1000
aw_exp_kwargs["discr_model_frequency"] = 50

# Active WeaSuL 1000 iterations
np.random.seed(284)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_maxkl_1000, _ = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="maxkl")

# Nashaat 1000 iterations
np.random.seed(25)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat_1000, _ = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="nashaat")

exp_kwargs["al_it"] = al_it
aw_exp_kwargs["discr_model_frequency"] = 1

# ### Active learning

# Next, we run active learning by itself.

# +
set_seed(76)

train_dataset = CustomTensorDataset(X=df_train.loc[[0],["x1", "x2"]], Y=y_train[0])
predict_dataset = CustomTensorDataset(X=torch.Tensor(df_train.loc[:,["x1","x2"]].values), Y=torch.Tensor(y_train))
test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.loc[:,["x1","x2"]].values), Y=torch.Tensor(y_test))

# Active learning parameters
al_exp_kwargs = dict(
    seeds = np.random.randint(0,1000,nr_trials),
    features = df_train.loc[:, ["x1","x2"]],
    train_dataset = train_dataset,
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=df_test.loc[:, ["x1", "x2"]].values
)
# -

metrics_activelearning = synthetic_al_experiment(**exp_kwargs, **al_exp_kwargs)

# ### Process results

# +
# Transform results into dataframes and merge

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(metrics_activelearning, "Active learning by itself")]).reset_index(level=0).rename(columns={"level_0": "Run"})

# Add weak supervision baseline
metric_dfs_incl_ws = add_weak_supervision_baseline(metric_dfs, al_it)

metric_dfs_1000 = pd.concat([process_exp_dict(metrics_maxkl_1000, "Active WeaSuL"),
                             process_exp_dict(metrics_nashaat_1000, "Nashaat et al.")]).reset_index(level=0).rename(columns={"level_0": "Run"})

# +
# Filter to plot only accuracy metric and test set

metric_dfs_incl_ws = metric_dfs_incl_ws[metric_dfs_incl_ws["Metric"] == "Accuracy"]
metric_dfs_incl_ws = metric_dfs_incl_ws[metric_dfs_incl_ws["Set"] == "test"]
metric_dfs_incl_ws["Value"] = metric_dfs_incl_ws["Value"].fillna(0.5)

metric_dfs_1000 = metric_dfs_1000[metric_dfs_1000["Metric"] == "Accuracy"]
metric_dfs_1000 = metric_dfs_1000[metric_dfs_1000["Set"] == "test"]
# -

# ## Figure 3

# +
plotting_kwargs = dict(x="Number of labeled points", y="Value", hue="Approach",
                       ci=68, n_boot=n_boot, estimator="mean", linewidth=linewidth)

fig, axes = plt.subplots(1,3, figsize=(22.5,8), sharey=True)

# Style
colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
plt.tight_layout()

# First column
sns.lineplot(data=metric_dfs_incl_ws[metric_dfs_incl_ws["Model"] == "Generative"],
            legend=False, **plotting_kwargs,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself"], ax=axes[0])
axes[0].set_title("Generative model (30 iterations)", size=font_size)

# Second column
sns.lineplot(data=metric_dfs_incl_ws[metric_dfs_incl_ws["Model"] == "Discriminative"], **plotting_kwargs,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"], ax=axes[1])
axes[1].get_legend().remove()
axes[1].set_title("Discriminative model (30 iterations)", fontsize=font_size)

# Third column
sns.lineplot(data=metric_dfs_1000[metric_dfs_1000["Model"] == "Discriminative"],
                  hue_order=["Active WeaSuL", "Nashaat et al."], **plotting_kwargs)
axes[2].set_title("Discriminative model (1000 iterations)", fontsize=font_size)

# Legend
handles, labels = axes[1].get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
leg = axes[2].legend(handles=handles, labels=labels, loc="lower right", title="Method", fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"

# Axes
for i in range(3): 
    axes[i].tick_params(axis='both', which='major', labelsize=tick_size)
    axes[i].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("Accuracy", fontsize=font_size)
plt.ylim(0.5, 1)

plt.show()
# -
# # Figure 4 and 6: Visual Relationship Detection

# The weak supervision setup for the dataset in this section builds on https://github.com/snorkel-team/snorkel-tutorials/tree/master/visual_relation.

pretrained_model = models.resnet18(pretrained=True)
path_prefix = "../data/VRD/"

# ## Loading and preparing data

# We select a subset of meaningful predicates from the VRD dataset and label "sit on" versus other predicates.

# +
semantic_predicates = ["carry", "cover", "fly", "look", "lying on", "park on", "sit on", "stand on", "ride"]
classify = ["sit on"]

df_train, df_test = load_vr_data(classify=classify, include_predicates=semantic_predicates,
                                 path_prefix=path_prefix, drop_duplicates=True, validation=False)

y_train = df_train.y.values
y_test = df_test.y.values

print("Train Relationships: ", len(df_train))
print("Test Relationships: ", len(df_test))
# -

# We first extract features from the images and object categories, then save resulting embeddings for fast fine-tuning.

# +
dataset_train = VisualRelationDataset(image_dir=path_prefix + "/images/train_images", df=df_train, Y=y_train)
dl_train = DataLoader(dataset_train, shuffle=False, batch_size=256)

discriminative_model = VisualRelationClassifier(pretrained_model, lr=1e-3, n_epochs=3, data_path_prefix="../data/", soft_labels=False)

feature_tensor_train = torch.Tensor([])
for batch_features, batch_labels in dl_train:
    feature_tensor_train = torch.cat((feature_tensor_train, discriminative_model.extract_concat_features(batch_features).to("cpu")))

# +
dataset_test = VisualRelationDataset(image_dir=path_prefix + "images/test_images", df=df_test, Y=y_test)
dl_test = DataLoader(dataset_test, shuffle=False, batch_size=256)

feature_tensor_test = torch.Tensor([])
for batch_features, batch_labels in dl_test:
    feature_tensor_test = torch.cat((feature_tensor_test, discriminative_model.extract_concat_features(batch_features).to("cpu")))
# -

# ## Creating labelling functions

# We define three different labelling functions. The first classifies based on the subject and object categories in the image. The second is based on the distance between the object and the subject. The third is based on the relative sizes of the bounding boxes of the object and the subject.

# +
SITON = 1
OTHER = 0

def lf_siton_object(x):
    if x.subject_category == "person":
        if x.object_category in ["bench", "chair", "floor", "horse", "grass", "table"]:
            return SITON
    return OTHER

def lf_dist(x):
    if np.linalg.norm(np.array(x.subject_bbox) - np.array(x.object_bbox)) >= 100:
        return OTHER
    return SITON

def area(bbox):
    return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])

def lf_area(x):
    if area(x.subject_bbox) / area(x.object_bbox) < 0.8:
        return SITON
    return OTHER


# +
lfs = [lf_siton_object, lf_dist, lf_area]

label_matrix_train = apply_lfs(df_train, lfs)
label_matrix_test = apply_lfs(df_test, lfs)
# -

# ## Running experiments

# +
# Update some settings for this dataset

al_it = 250

discriminative_model_kwargs = dict(lr=1e-3,
                          n_epochs=100)

batch_size = 20

class_balance = np.array([1-df_train.y.mean(), df_train.y.mean()])

# +
# Collect all experiment parameters

# Common
exp_kwargs = dict(nr_trials=nr_trials,
                  al_it=al_it,
                  y_train=y_train,
                  batch_size=batch_size,
                  y_test=y_test)

# Active WeaSuL
aw_exp_kwargs = dict(label_matrix=label_matrix_train,
                     cliques=cliques,
                     class_balance=class_balance,
                     starting_seed=starting_seed,
                     penalty_strength=penalty_strength,
                     discriminative_model=VisualRelationClassifier(pretrained_model, **discriminative_model_kwargs, data_path_prefix="../data/", patience=5, early_stopping=True),
                     discr_model_frequency=1,
                     train_dataset = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train)),
                     test_dataset = CustomTensorDataset(feature_tensor_test, torch.Tensor(y_test)),
                     label_matrix_test=label_matrix_test)
# -

# ### Active WeaSuL

np.random.seed(50)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_maxkl, entropies_maxkl = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="maxkl")

# ### Nashaat et al.

np.random.seed(25)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_nashaat, _ = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="nashaat")

# ### Active learning

# +
set_seed(76)

train_dataset = CustomTensorDataset(feature_tensor_train[0,:], torch.Tensor(y_train[0]))
predict_dataset = CustomTensorDataset(feature_tensor_train, torch.Tensor(y_train))
test_dataset = CustomTensorDataset(feature_tensor_test, torch.Tensor(y_test))

discriminative_model_kwargs["n_epochs"] = 2

# Active learning parameters
al_exp_kwargs = dict(
    seeds = np.random.randint(0,1000,nr_trials),
    model = VisualRelationClassifier(pretrained_model, **discriminative_model_kwargs, early_stopping=False, data_path_prefix="../data/", soft_labels=False),
    features = feature_tensor_train,
    train_dataset = train_dataset,
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=feature_tensor_test
)

discriminative_model_kwargs["n_epochs"] = 100
# -

metrics_activelearning = active_learning_experiment(**exp_kwargs, **al_exp_kwargs)

# ### Process results

# +
# Transform results into dataframes and merge

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(metrics_activelearning, "Active learning by itself")]).reset_index(level=0).rename(columns={"level_0": "Run"})

# Add weak supervision baseline
metric_dfs_incl_ws = add_weak_supervision_baseline(metric_dfs, al_it)

# +
# Filter to plot only F1 metric and test set

metric_dfs_incl_ws = metric_dfs_incl_ws[metric_dfs_incl_ws["Metric"] == "F1"]
metric_dfs_incl_ws = metric_dfs_incl_ws[metric_dfs_incl_ws["Set"] == "test"]
metric_dfs_incl_ws["Value"] = metric_dfs_incl_ws["Value"].fillna(0)
# -

# ## Figure 4B

# +
fig, axes = plt.subplots(1, 1, figsize=(15,8))

sns.lineplot(data=metric_dfs_incl_ws[metric_dfs_incl_ws["Model"] == "Discriminative"], **plotting_kwargs,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"])

axes.set_title("Discriminative model", fontsize=font_size)

# Legend
handles, labels = axes.get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
leg = axes.legend(handles=handles, labels=labels[0:6], loc="lower right", title="Method",
                     fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"

# Axes
axes.tick_params(axis='both', which='major', labelsize=tick_size)
axes.set_xlabel("Number of active learning iterations", fontsize=font_size)
axes.set_ylabel("F1", fontsize=font_size)

plt.tight_layout()

plt.show()
# -

# ## Sampling strategies

# Now we compare the proposed maxKL sampling strategy to a few other approaches for the visual relationship data set. We compare to margin sampling, which samples based on the distance to the classification boundary, and random sampling.
#
# We also keep track of the entropy of sampled buckets across active learning iterations.

# +
# Switch to 50 iterations to compare sampling approaches

exp_kwargs["al_it"] = 50
# -

# ### Margin sampling

np.random.seed(70)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_margin, entropies_margin = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="margin")

# ### Random sampling

np.random.seed(70)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,nr_trials)
metrics_random, entropies_random = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="margin", randomness=1)

# ### Process metrics

# +
# Transform results into dataframes and merge

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "MaxKL"),
                        process_exp_dict(metrics_margin, "Margin"),
                        process_exp_dict(metrics_random, "Random")])

# +
# Filter to plot only F1 metric, test set and first 50 iterations

metric_dfs = metric_dfs[metric_dfs["Metric"].isin(["F1"])]
metric_dfs = metric_dfs[metric_dfs.Set == "test"]
metric_dfs = metric_dfs[metric_dfs["Number of labeled points"] < 51]
# -

# ### Process entropies

entropies_df = pd.concat([process_entropies(entropies_maxkl, "MaxKL"), 
                          process_entropies(entropies_margin, "Margin"), 
                          process_entropies(entropies_random, "Random")])

# ## Figure 6A

# +
lines = list(metric_dfs.Approach.unique())

fig, axes = plt.subplots(1,2, figsize=(15,8), sharey=True)

# Style
colors = ["#2b4162", "#CC7178", "#598B2C", "#e9c46a"][:len(lines)]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
plt.tight_layout()

# Column 1
sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Generative"], **plotting_kwargs,
            hue_order=["MaxKL", "Margin", "Random"], ax=axes[0])
axes[0].set_title("Generative model", fontsize=font_size)

# Column 2
sns.lineplot(data=metric_dfs[metric_dfs["Model"] == "Discriminative"],
             legend=False, **plotting_kwargs, hue_order=["MaxKL", "Margin", "Random"], ax=axes[1])
axes[1].set_title("Discriminative model", fontsize=font_size)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
axes[0].legend(handles=handles, labels=labels, loc="lower right", title="Sampling method", fontsize=legend_size, title_fontsize=legend_size)

# Axes
axes[0].tick_params(axis='both', which='major', labelsize=tick_size)
axes[1].tick_params(axis='both', which='major', labelsize=tick_size)
axes[0].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[1].set_xlabel("Number of active learning iterations", fontsize=font_size)
axes[0].set_ylabel("F1", fontsize=font_size)

plt.show()
# -

# ## Figure 6B

# +
fig, ax = plt.subplots(1,1,figsize=(8,8))

plt.tight_layout()

sns.lineplot(data=entropies_df, x="Number of labeled points", y="Entropy", hue="Approach", ci=68, n_boot=n_boot,
                  legend=False, linewidth=linewidth, hue_order=["MaxKL", "Margin","Random"], ax=ax)
ax.set_title("Diversity of sampled buckets", fontsize=font_size)

# Axes
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel("Number of active learning iterations", fontsize=font_size)
ax.set_ylabel("Diversity (entropy)", fontsize=font_size)

plt.show()
# -

# # Figure 5: Spam Detection

# The weak supervision setup for the dataset in this section builds on https://github.com/snorkel-team/snorkel-tutorials/tree/master/spam. We use the first 7 labelling functions and directly import the resulting label matrices.

path_prefix = "../data/spam/"

# ## Loading and preparing data

# +
label_matrix_train = pickle.load(open(path_prefix + "L_train.pkl", "rb"))
label_matrix_test = pickle.load(open(path_prefix + "L_test.pkl", "rb"))

df_train = pickle.load(open(path_prefix + "X_train.pkl", "rb"))
df_test = pickle.load(open(path_prefix + "X_test.pkl", "rb"))

y_train = pickle.load(open(path_prefix + "Y_train.pkl", "rb"))
y_test = pickle.load(open(path_prefix + "Y_test.pkl", "rb"))

# +
# Drop the data points where all of the labelling functions abstain

indices_keep = label_matrix_train.sum(axis=1) != -7
label_matrix_train = label_matrix_train[indices_keep]
y_train = y_train[indices_keep]

df_train = pd.DataFrame.sparse.from_spmatrix(df_train)
df_train = df_train.iloc[indices_keep].reset_index()
df_test = pd.DataFrame.sparse.from_spmatrix(df_test).reset_index()
# -

# ## Running experiments

# +
starting_seed = 34
penalty_strength = 1e6
al_it = 100

p_z = 0.58
class_balance = np.array([1 - p_z, p_z])
cliques = [[0],[1],[2],[3],[4],[5],[6]]

discriminative_model_kwargs = dict(input_dim=df_train.shape[1],
                          output_dim=2,
                          lr=1e-2,
                          n_epochs=100)

# +
# Collect all experiment parameters

# Common
exp_kwargs = dict(nr_trials=nr_trials,
                  al_it=al_it,
                  y_train=y_train,
                  batch_size=batch_size,
                  y_test=y_test)

# Active WeaSuL
aw_exp_kwargs = dict(label_matrix=label_matrix_train,
                  cliques=cliques,
                  class_balance=class_balance,
                  starting_seed=starting_seed,
                  penalty_strength=penalty_strength,
                  discriminative_model=LogisticRegression(**discriminative_model_kwargs),
                  discr_model_frequency=1,
                  train_dataset = CustomTensorDataset(X=torch.Tensor(df_train.values), Y=torch.Tensor(y_train)),
                  test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.values), Y=torch.Tensor(y_test)),
                  label_matrix_test=label_matrix_test)
# -

# ### Active WeaSuL

np.random.seed(284)
aw_exp_kwargs["seeds"]= np.random.randint(0, 1000, 10)
metrics_maxkl, _ = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="maxkl")

# ### Nashaat et al.

np.random.seed(25)
aw_exp_kwargs["seeds"]= np.random.randint(0,1000,10)
metrics_nashaat, _ = active_weasul_experiment(**exp_kwargs, **aw_exp_kwargs, query_strategy="nashaat")

# ### Active learning

# +
set_seed(76)

train_dataset = CustomTensorDataset(X=df_train.loc[[0],], Y=y_train[0])
predict_dataset = CustomTensorDataset(X=torch.Tensor(df_train.values), Y=torch.Tensor(y_train))
test_dataset = CustomTensorDataset(X=torch.Tensor(df_test.values), Y=torch.Tensor(y_test))

# Active learning parameters
al_exp_kwargs = dict(
    seeds = np.random.randint(0,1000,nr_trials),
    features = df_train,
    train_dataset = train_dataset,
    predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False),
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    test_features=df_test
)
# -

metrics_activelearning = synthetic_al_experiment(**exp_kwargs, **al_exp_kwargs)

# ### Process results

# +
# Transform results into dataframes and merge

metric_dfs = pd.concat([process_exp_dict(metrics_maxkl, "Active WeaSuL"),
                        process_exp_dict(metrics_nashaat, "Nashaat et al."),
                        process_exp_dict(metrics_activelearning, "Active learning by itself")]).reset_index(level=0).rename(columns={"level_0": "Run"})

# Add weak supervision baseline
metric_dfs_incl_ws = add_weak_supervision_baseline(metric_dfs, al_it)

# +
# Filter to plot only F1 metric and test set

metric_dfs_incl_ws = metric_dfs_incl_ws[metric_dfs_incl_ws["Metric"] == "F1"]
metric_dfs_incl_ws = metric_dfs_incl_ws[metric_dfs_incl_ws["Set"] == "test"]
metric_dfs_incl_ws["Value"] = metric_dfs_incl_ws["Value"].fillna(0)
# -

# ## Figure 5

# +
fig, axes = plt.subplots(1, 1, figsize=(15,8))

# Style
colors = ["#2b4162", "#368f8b", "#ec7357", "#e9c46a"]
sns.set(style="whitegrid", palette=sns.color_palette(colors))
plt.tight_layout()

sns.lineplot(data=metric_dfs_incl_ws[metric_dfs_incl_ws["Model"] == "Discriminative"], **plotting_kwargs,
            hue_order=["Active WeaSuL", "Nashaat et al.", "Weak supervision by itself", "Active learning by itself"])

axes.set_title("Discriminative model", fontsize=font_size)

# Legend
handles, labels = axes.get_legend_handles_labels()
[ha.set_linewidth(linewidth) for ha in handles]
leg = axes.legend(handles=handles, labels=labels[0:6], loc="lower right", title="Method",
                     fontsize=legend_size, title_fontsize=legend_size)
leg._legend_box.align = "left"

# Axes
axes.tick_params(axis='both', which='major', labelsize=tick_size)
axes.set_xlabel("Number of active learning iterations", fontsize=font_size)
axes.set_ylabel("F1", fontsize=font_size)

plt.show()
# -


