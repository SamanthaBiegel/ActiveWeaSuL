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

import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

sys.path.append(os.path.abspath("../activelearning"))
from data import SyntheticData
from final_model import DiscriminativeModel
from plot import plot_probs, plot_accuracies
from label_model import LabelModel
from pipeline import ActiveLearningPipeline
# -


pd.options.display.expand_frame_repr = False 
np.set_printoptions(suppress=True, precision=16)

# # Sample data and weak labels

# +
N = 10000
centroids = np.array([[0.1, 1.3], [-0.8, -0.5]])
# centroids = np.array([[0.3, 0.5], [-0.8, -0.5]])
# centroids = np.array([[0.1, 0.3], [-0.8, -0.5]])

p_z = 0.5
# -

wl_dict = {0: [-0.3, -1, 0.5], 1: [0.2]}

data = SyntheticData(N, p_z, centroids)
df = data.sample_data_set().create_df()

plot_probs(df, data.y.astype(str), soft_labels=False)

# # Create weak labels

# +
df.loc[:, "wl1"] = (df["x2"]<0.4)*1
df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
df.loc[:, "wl3"] = (df["x1"]<-1)*1
# df.loc[:, "wl4"] = (df["x1"]<0.5)*1

# df.loc[:, "wl1"] = (df["x2"]<0.2)*1
# df.loc[:, "wl2"] = (df["x1"]<-0.3)*1
# df.loc[:, "wl3"] = (df["x1"]<-1)*1
# df.loc[:, "wl4"] = (df["x1"]<0.5)*1
# df.loc[:, "wl5"] = (df["x2"]<0.7)*1

# df.loc[:, "wl1"] = (df["x2"]<0.2)*1
# df.loc[:, "wl2"] = (df["x1"]<-0.5)*1
# df.loc[:, "wl3"] = (df["x1"]<-0.9)*1

# -

print("Accuracy wl1:", (df["y"] == df["wl1"]).sum()/len(data.y))
print("Accuracy wl2:", (df["y"] == df["wl2"]).sum()/len(data.y))
print("Accuracy wl3:", (df["y"] == df["wl3"]).sum()/len(data.y))
# print("Accuracy wl4:", (df["y"] == df["wl4"]).sum()/len(data.y))

# +
def random_LF(y, fp, fn, abstain):
    ab = np.random.uniform()
    
    if ab < abstain:
        return -1
    
    threshold = np.random.uniform()
    
    if y == 1 and threshold < fn:
        return 0
        
    elif y == 0 and threshold < fp:
        return 1
        
    
    
    return y

df.loc[:, "wl1"] = [random_LF(y, fp=0.1, fn=0.2, abstain=0) for y in df["y"]]
df.loc[:, "wl2"] = [random_LF(y, fp=0.05, fn=0.4, abstain=0) for y in df["y"]]
df.loc[:, "wl3"] = [random_LF(y, fp=0.2, fn=0.3, abstain=0) for y in df["y"]]

# +
# label_matrix = np.array(df[["wl1", "wl2", "wl3", "wl4", "wl5", "y"]])

label_matrix = np.array(df[["wl1", "wl2", "wl3", "y"]])
# -

# # Label model with cliques

# +
final_model_kwargs = {'input_dim': 2,
                      'output_dim': 2,
                      'lr': 0.001,
                      'batch_size': 256,
                      'n_epochs': 250}

class_balance = np.array([0.5, 0.5])

# +
# L = label_matrix[:, :-1]
# cliques=[[0, 4],[1,2,3]]

L = label_matrix[:, :-1]
cliques=[[0],[1,2]]

# L = label_matrix
# cliques=[[0, 3],[1,2, 3]]

for i in range(10):

    lm = LabelModel(final_model_kwargs=final_model_kwargs, df=df, active_learning=False, add_cliques=True, add_prob_loss=False, n_epochs=200, lr=1e-1)
    Y_probs_cliques = lm.fit(label_matrix=L, cliques=cliques, class_balance=class_balance).predict()
    print(lm.accuracy())
# -

lm.wl_idx

lm.psi[:, lm.wl_idx["1_2"]]

lm.psi[:, lm.wl_idx["2"]]

lm.psi[:, lm.wl_idx["1"]]

L[L[:,3] != -1]

lm.predict()

lm.mu

lm.get_true_mu()

probs_test = lm.predict_true()

plot_probs(df, Y_probs_cliques.detach().numpy(), soft_labels=True, subset=None)

plot_probs(df, probs_test.detach().numpy(), soft_labels=True, subset=None)

# # Final model

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final = fm.fit(features=data.X, labels=Y_probs_cliques.detach().numpy()).predict()
fm.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=False)
probs_final_true = fm.fit(features=data.X, labels=data.y).predict()
fm.accuracy()

plot_probs(df, probs_final.detach().numpy(), soft_labels=True, subset=None)

plot_probs(df, probs_final_true.detach().numpy(), soft_labels=True, subset=None)

# # Active learning

# +
it = 100
active_learning = "probs"
add_cliques=True
add_prob_loss=False

if active_learning == "cov":
    cliques=[[0],[1,2],[3]]
    wl_al = np.full_like(df["y"], -1)
    L = np.concatenate([label_matrix[:,:-1], wl_al.reshape(len(wl_al),1)], axis=1)
if active_learning == "probs":
    cliques=[[0],[1,2]]
    L = label_matrix[:, :-1]
    
al = ActiveLearningPipeline(it=it,
                            final_model_kwargs=final_model_kwargs,
                            df=df,
                            active_learning=active_learning,
                            add_cliques=add_cliques,
                            add_prob_loss=add_prob_loss)

Y_probs_al = al.refine_probabilities(label_matrix=L, cliques=cliques, class_balance=class_balance)
# -

wl_al = np.full_like(df["y"], -1)
L = np.concatenate([label_matrix[:,:-1], wl_al.reshape(len(wl_al),1)], axis=1)

L[al.queried, 3] = data.y[al.queried]

L = np.concatenate([L, data.y[:, None]], axis=1)

al.accuracy()

fm = DiscriminativeModel(df, **final_model_kwargs, soft_labels=True)
probs_final_al = fm.fit(features=data.X, labels=Y_probs_al.detach().numpy()).predict()
fm.accuracy()

plot_probs(df, Y_probs_al.detach().numpy(), soft_labels=True, subset=None)

plot_probs(df, probs_final_al.detach().numpy(), soft_labels=True, subset=None)

probs_df = pd.DataFrame.from_dict(al.prob_dict)
probs_df = probs_df.stack().reset_index().rename(columns={"level_0": "x", "level_1": "iteration", 0: "prob_y"})
probs_df = probs_df.merge(df, left_on = "x", right_index=True)

# +
fig = px.scatter(probs_df, x="x1", y="x2", color="prob_y", animation_frame="iteration", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0,-1]], color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)
fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
                  width=1000, height=1000, xaxis_title="x1", yaxis_title="x2", template="plotly_white")

fig.show()
# -
unique_probs_df = pd.DataFrame.from_dict(al.unique_prob_dict)
unique_probs_df = unique_probs_df.stack().reset_index().rename(columns={"level_0": "Configuration", "level_1": "Iteration", 0: "P_Y_1"})


fig = px.line(unique_probs_df, x="Iteration", y="P_Y_1", color="Configuration")
fig.show()



# +
df_res = pd.DataFrame(accuracies)

df_res.columns = ["no_al_LM", "no_al_final", "al_LM", "al_final"]

df_res.index.name = "run"

df_res = df_res.stack().reset_index().rename(columns={"level_1": "type", 0: "accuracy"})

fig = px.line(df_res, x = "run", y = "accuracy", color="type")
fig.update_yaxes(range=[0, 1])
fig.show()

fig = px.line(probs_df, x = "run", y = "accuracy", color="type")
# fig.update_yaxes(range=[0.5, 1])
fig.show()

# +
mean_accuracies = {keys: np.array(values).mean() for keys, values in accuracies.items()}

plot_probs(df, probs)

plot_probs(df, probs_al)

# +

df_res = pd.DataFrame(accuracies)

df_res.columns = ["no_al_LM", "no_al_final", "al_LM", "al_final", "supervised"]
# df_res.columns = ["no_al_LM", "no_al_final", "al_LM", "al_final"]

df_res.index.name = "run"

df_res.index = df_res.index + 1

df_res = df_res.stack().reset_index().rename(columns={"level_1": "type", 0: "accuracy"})

fig = px.line(df_res, x = "run", y = "accuracy", color="type", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[1,0,-2,-1,3]])
fig.update_yaxes(range=[0.7, 1])
fig.update_layout(template="plotly_white", width=1200, height=700)
fig.show()
# -

feature_matrix = df[["x1", "x2", "y"]]
df_inv_cov = pd.DataFrame(np.linalg.pinv(np.cov(feature_matrix.T)))
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_inv_cov, ax=ax, vmin=-4, vmax=4, center=0, annot=True, linewidths=.5, cmap="RdBu_r", square=True, xticklabels=True, yticklabels=True, fmt='.3g')

wl_al = np.full_like(df["y"], -1)
L = np.concatenate([label_matrix[:,:-1], wl_al.reshape(len(wl_al),1)], axis=1)

pick_indices = np.random.choice(np.arange(N_total), size = (1, 8000), replace=False)

L[pick_indices[0], 3] = df.iloc[pick_indices[0]]["y"]

lm.label_matrix

# +
# tmp_L = np.concatenate([L, df["y"].values[:,np.newaxis], axis=1)

# tmp_L_onehot = ((y_set == tmp_L[..., None]) * 1).reshape(N_total, -1)

# df_inv_cov = pd.DataFrame(np.linalg.pinv(np.cov(tmp_L_onehot.T)))
                        
df_inv_cov = pd.DataFrame(lm.cov_O_inverse.detach().numpy())

labels=[("L1", 0), ("L1", 1), ("L2", 0), ("L2", 1), ("L3", 0), ("L3", 1), ("L_AL", 0), ("L_AL", 1), ("Y", 0), ("Y", 1), ("L23", "00"), ("L23", "10"), ("L23", "01"), ("L23", "11")]

df_inv_cov.columns = pd.MultiIndex.from_tuples(labels, names=["label", "class"])
df_inv_cov.index = pd.MultiIndex.from_tuples(labels, names=["label", "class"])
# -

df_inv_cov = df_inv_cov.sort_values(by="label", axis=0).sort_values(by="label", axis=1)

idx1 = [6,7]
idx2 = [8,9]

lambda_al_Y = ((tmp_L_onehot[:, np.newaxis, idx1[0]:(idx1[-1]+1)]
                        * tmp_L_onehot[:, idx2[0]:(idx2[-1]+1), np.newaxis]).reshape(len(tmp_L_onehot), -1))

df_inv_cov

fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_inv_cov, ax=ax, vmin=-1, vmax=1, center=0, annot=True, linewidths=.5, cmap="RdBu_r", square=True, xticklabels=True, yticklabels=True, fmt='.3g')
# fig.savefig("cov.png")
fig.show()

# +
tmp_L = np.concatenate([L, np.array(df["y"]).reshape(len(df["y"]),1)], axis=1)

tmp_L_onehot = np.concatenate([((y_set == tmp_L[..., None]) * 1).reshape(N_total, -1), lambda_al_Y], axis=1)

df_inv_cov = pd.DataFrame(np.linalg.pinv(np.cov(tmp_L_onehot.T)))

labels=[("L1", 0), ("L1", 1), ("L2", 0), ("L2", 1), ("L3", 0), ("L3", 1), ("L_AL", 0), ("L_AL", 1), ("Y", 0), ("Y", 1), ("L_AL_Y", "00"), ("L_AL_Y", "01"), ("L_AL_Y", "10"), ("L_AL_Y", "11")]

df_inv_cov.columns = pd.MultiIndex.from_tuples(labels, names=["label", "class"])
df_inv_cov.index = pd.MultiIndex.from_tuples(labels, names=["label", "class"])
# -

Y_probs

# Probabilistic labels without active learning
plot_probs(df, Y_probs)
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_inv_cov, ax=ax, vmin=-4, vmax=4, center=0, annot=True, linewidths=.5, cmap="RdBu_r", square=True, xticklabels=True, yticklabels=True, fmt='.3g')
fig.savefig("cov2.png")



# +
# prob_label_df = pd.DataFrame.from_dict(prob_label_dict)
# prob_label_df = prob_label_df.stack().reset_index().rename(columns={"level_0": "x", "level_1": "iteration", 0: "prob_y"})
# prob_label_df = prob_label_df.merge(df, left_on = "x", right_index=True)

# +


# fig2 = px.scatter(prob_label_df, x="x1", y="x2", color="prob_y", animation_frame="iteration", color_discrete_sequence=np.array(px.colors.diverging.Geyser)[[0,-1]], color_continuous_scale=px.colors.diverging.Geyser, color_continuous_midpoint=0.5)
# fig2.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),
#                   width=1000, height=1000, xaxis_title="x1", yaxis_title="x2", template="plotly_white")

# fig2.show()

# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig),
#     dcc.Graph(figure=fig2)
# ])

# app.run_server(debug=True, use_reloader=False)

# +
x = list(range(len(accuracies_it["prob_labels"])))
x_gap = list(range(0, len(accuracies_it["prob_labels"]), 20))
x_gap_2 = list(range(0, len(accuracies_it["prob_labels"]), 100))


fig = go.Figure(data=go.Scatter(x=x, y=accuracies_it["prob_labels"], mode="lines", line_color=px.colors.diverging.Geyser[0], name="prob labels"))
# fig = go.Figure()

fig.add_trace(go.Scatter(x=x_gap, y=accuracies_it["final_labels"], mode="lines", line_color=px.colors.diverging.Geyser[-1], name="final labels"))

fig.add_trace(go.Scatter(x=x, y=accuracies_it["supervised"], mode="lines", line_color=px.colors.diverging.Geyser[3], name="supervised"))

# fig.add_trace(go.Scatter(x=x_gap_2, y=accuracies_it_random["final_labels"], mode="lines", line_color=px.colors.diverging.Geyser[-2], name="final labels random"))

# fig.add_trace(go.Scatter(x=x, y=accuracies_it_random["prob_labels"], mode="lines", line_color=px.colors.diverging.Geyser[1], name="prob labels random"))

fig.update_layout(template="plotly_white", xaxis_title="iteration", yaxis_title="accuracy", width=1200, height=700)
fig.update_yaxes(range=[0.7, 1])

fig.show()

# +
mean_accuracies = {keys: np.array(values).mean() for keys, values in accuracies.items()}

df_accuracies = pd.DataFrame.from_dict(mean_accuracies, orient="index", columns=["Mean accuracy"])
df_accuracies["Active learning"], df_accuracies["Labels"] = df_accuracies.index.str.split('_').str
df_accuracies.set_index(["Labels", "Active learning"]).sort_values(["Active learning"])
pd.pivot_table(df_accuracies, columns="Labels", index="Active learning")
# -

# # Final model active learning

probs_al_0 = preds = np.concatenate([1 - al.prob_dict[0][:, np.newaxis], al.prob_dict[0][:, np.newaxis]], axis=1)
probs_al_99 = preds = np.concatenate([1 - al.prob_dict[99][:, np.newaxis], al.prob_dict[99][:, np.newaxis]], axis=1)

probs_final_0 = fit_predict_fm(df[["x1", "x2"]].values, probs_al_0, **final_model_kwargs, soft_labels=True)
probs_final_99 = fit_predict_fm(df[["x1", "x2"]].values, probs_al_99, **final_model_kwargs, soft_labels=True)

lm._accuracy(probs_final_0, df["y"].values)

lm._accuracy(probs_final_99, df["y"].values)

plot_probs(df, probs_final_0, soft_labels=True, subset=None)

plot_probs(df, probs_final_99, soft_labels=True, subset=None)

# # Train model on queried data points

queried = np.array(queried)

# +
_, probs_q = fit_predict_fm(df[["x1", "x2"]].values, df["y"].values, **final_model_kwargs, soft_labels=False, subset=queried)

get_overall_accuracy(probs_q, df["y"])
# -

plot_probs(df, probs_q, soft_labels=True, subset=queried)

# # Train model on random subset

random_idx = np.random.choice(range(N_total), al_kwargs["it"])

# +
_, probs_r = fit_predict_fm(df[["x1", "x2"]].values, df["y"].values, **final_model_kwargs, soft_labels=False, subset=random_idx)

get_overall_accuracy(probs_r, df["y"])
# -

plot_probs(df, probs_r, soft_labels=True, subset=random_idx)




