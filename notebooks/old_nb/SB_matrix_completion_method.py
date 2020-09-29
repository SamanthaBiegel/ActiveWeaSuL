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
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from snorkel.labeling.model import LabelModel


# +
N = 10000

x1 = np.random.normal(loc=0, scale=1, size=N)
x2 = np.random.normal(loc=0, scale=1, size=N)


# -

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def softmax(z):
    z_x = np.exp(z - np.max(z))
    return z_x / np.sum(z_x, axis=1, keepdims=True)


# +
coef = [4, 10, 10]

Z = coef[0] + coef[1]*x1 + coef[2]*x2

h = sigmoid(Z)

y = np.random.binomial(1, h, N)
# -

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

x_dec = np.linspace(-3,3, 1000)
y_dec = (- coef[0] - coef[1]*x_dec)/coef[2]

# +
fig = px.scatter(x=df["x1"], y=df["x2"], color=h)
fig.add_trace(go.Scatter(x=x_dec, y=y_dec, mode="lines", name="decision boundary"))
fig.update_layout(xaxis=dict(range=[-4,4]), yaxis=dict(range=[-4,4],scaleanchor="x", scaleratio=1), width=700, height=700, xaxis_title="x1", yaxis_title="x2")
fig.show()

# df["y"].astype("category")

# +
reg = LogisticRegression(solver="lbfgs")
reg.fit(np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1),y)

print(reg.intercept_)
print(reg.coef_)
# -

# # Create clusters

# +
N_1 = 5000
N_2 = 5000
N_total = N_1 + N_2
centroid_1 = np.array([0.1, 1.3])
centroid_2 = np.array([-0.8, -0.5])
centroids = np.concatenate([centroid_1.reshape(1,-1), centroid_2.reshape(1,-1)], axis=0)
scale=0.5

x1_1 = np.random.normal(loc=centroid_1[0], scale=scale, size=N_1).reshape(-1, 1)
x2_1 = np.random.normal(loc=centroid_1[1], scale=scale, size=N_1).reshape(-1, 1)
x1_2 = np.random.normal(loc=centroid_2[0], scale=scale, size=N_2).reshape(-1, 1)
x2_2 = np.random.normal(loc=centroid_2[1], scale=scale, size=N_2).reshape(-1, 1)

X_1 = np.concatenate([x1_1, x2_1], axis=1)
X_2 = np.concatenate([x1_2, x2_2], axis=1)
X = np.concatenate([X_1, X_2])
# -

# Create decision boundary between the two clusters: the boundary goes through the point that is exactly between the two cluster means. 

# +
# point in the middle of cluster means
bp1 = centroids.sum(axis=0)/2

# vector between cluster means
diff = centroids[1,:]-centroids[0,:]

# slope of decision boundary is perpendicular to slope of line that goes through the cluster means
slope = diff[1]/diff[0]
perp_slope = -1/slope

# solve for intercept using slope and middle point
b = bp1[1] - perp_slope*bp1[0]
# -


# Now we have the decision boundary line: $x2 = b + a*x1$ or $0 = b + a*x1 - x2$, so the coefficients $\beta$ are b, a and -1

# +
coef = [b, perp_slope, -1]
coef = [4*co for co in coef]

Z = coef[0] + coef[1]*X[:,0] + coef[2]*X[:,1]

h = sigmoid(Z)

y = np.random.binomial(1, h, N_1+N_2)#.astype("str")
# -

x_dec = np.linspace(centroid_2[0]-2, centroid_1[0]+2, 1000)
y_dec = (- coef[0] - coef[1]*x_dec)/coef[2]

df = pd.DataFrame({'x1': X[:,0], 'x2': X[:,1], 'y': y})

coef

# +
# y = h > 0.5

# +
# fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5])

# fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers", marker=dict(color=y)), row=1, col=1)
# fig.add_trace(go.Scatter(x=x_dec, y=y_dec, name="decision boundary"), row=1, col=1)
# fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers", marker=dict(color=h)), row=1, col=2)
# fig.add_trace(go.Scatter(x=x_dec, y=y_dec, name="decision boundary"), row=1, col=2)
# fig.update_layout(height=600)
# fig.update_xaxes(title="x1", row=1, col=1)
# fig.update_xaxes(title="x1", row=1, col=2)
# fig.update_yaxes(title="x2", row=1, col=1)
# fig.update_yaxes(title="x2", row=1, col=2)
# fig.show()

# ,range=[centroid_2[0]-2,centroid_1[0]+2]
# ,range=[centroid_2[1]-2,centroid_1[1]+2],scaleanchor="x", scaleratio=1

fig = px.scatter(x=X[:,0], y=X[:,1], color=h)
fig.add_trace(go.Scatter(x=x_dec, y=y_dec, mode="lines", name="decision boundary"))
fig.update_layout(xaxis=dict(range=[centroid_2[0]-2,centroid_1[0]+2]), yaxis=dict(range=[centroid_2[1]-2,centroid_1[1]+2],scaleanchor="x", scaleratio=1), width=700, height=700, xaxis_title="x1", yaxis_title="x2")
fig.show()

# +
reg = LogisticRegression(solver="lbfgs")
reg.fit(np.concatenate([X[:,0].reshape(-1, 1), X[:,1].reshape(-1, 1)], axis=1), y)

print(reg.intercept_)
print(reg.coef_)


# -

# Create weak labels by looking at the data and making a split based on one feature. This way we can only get two weak labels if we want to keep it in two-dimensional feature space for experiments, so maybe not that useful for now. Snorkel mentions they require at least three weak labels.

# +
# wl1 = (X[:,0]<-0.3)*1
# wl2 = (X[:,1]<0.4)*1

# +
# print("Accuracy wl1:", (y == wl1).sum()/len(y))

# +
# print("Accuracy wl2:", (y == wl2).sum()/len(y))

# +
# label_matrix = np.concatenate([wl1.reshape(-1,1), wl2.reshape(-1,1)], axis=1)

# +
# x_dec = np.linspace(-5,5, 1000)

# fig = px.scatter(x=df["x1"], y=df["x2"], color=df["y"].astype("category"))
# fig.add_trace(go.Scatter(x=x_dec, y=y_dec, mode="lines", name="decision boundary"))
# fig.add_trace(go.Scatter(x=x_dec, y=np.repeat(0.3, N), mode="lines", name="wl2"))
# fig.add_trace(go.Scatter(x=np.repeat(-0.3, N), y=x_dec, mode="lines", name="wl1"))
# fig.update_layout(xaxis=dict(range=[-4,4]), yaxis=dict(range=[-4,4],scaleanchor="x", scaleratio=1), width=1000, height=1000, xaxis_title="x1", yaxis_title="x2")
# fig.show()

# +
# plt.figure(figsize=(5,3))
# sns.heatmap(pd.DataFrame(label_matrix[df["y"] == 1]).corr(), annot=True)
# plt.figure(figsize=(5,3))
# sns.heatmap(pd.DataFrame(label_matrix[df["y"] == 0]).corr(), annot=True)
# -

# # Create weak labels

# Create weak labels by randomly flipping from the ground truth targets.

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

# +
df.loc[:, "wl1"] = [random_LF(y, fp=0.1, fn=0.2, abstain=0) for y in df["y"]]
df.loc[:, "wl2"] = [random_LF(y, fp=0.05, fn=0.4, abstain=0) for y in df["y"]]
df.loc[:, "wl3"] = [random_LF(y, fp=0.2, fn=0.3, abstain=0) for y in df["y"]]

# df.loc[:, "wl1"] = [random_LF(y, fp=0.15, fn=0.15) for y in df["y"]]
# df.loc[:, "wl2"] = [random_LF(y, fp=0.25, fn=0.25) for y in df["y"]]
# df.loc[:, "wl3"] = [random_LF(y, fp=0.3, fn=0.3) for y in df["y"]]
# -

print("Accuracy wl1:", (df["y"] == df["wl1"]).sum()/len(y))

print("Accuracy wl2:", (df["y"] == df["wl2"]).sum()/len(y))

print("Accuracy wl3:", (df["y"] == df["wl3"]).sum()/len(y))

df

label_matrix = np.array(df[["wl1", "wl2", "wl3", "y"]])

# +
# label_matrix = label_matrix[y == 2,:]

# +
# plt.figure(figsize=(5,3))
# sns.heatmap(pd.DataFrame(label_matrix[y == 1]).corr(), annot=True)
# plt.figure(figsize=(5,3))
# sns.heatmap(pd.DataFrame(label_matrix[y == 2]).corr(), annot=True)
# -

y_dim = len(np.unique(label_matrix)) # number of classes
y_set = np.unique(label_matrix) # array of classes
nr_wl = label_matrix.shape[1] # number of weak labels

nr_wl

# # Compute covariances and expectations

# Convert label matrix to onehot format. The dimensions will be (nr of data points, nr of weak labels * nr of classes).

label_matrix_onehot = (np.array([0,1]) == label_matrix[...,None])*1
lm_sh = label_matrix_onehot.shape
label_matrix_onehot = label_matrix_onehot.reshape(lm_sh[0],lm_sh[1]*lm_sh[2])

# +
# nr_columns = (y_dim - 1)*nr_wl
# drop_columns = y_dim * np.array(range(nr_columns))
# select_columns = [i for i in list(range(label_matrix_onehot.shape[1])) if i not in drop_columns]
# onehot_minus_one = label_matrix_onehot[:,select_columns]
# -

covariance_matrix = np.cov(label_matrix_onehot.T)


def _color_zeros(val):
    color = 'red' if np.abs(val) < 0.1 else 'green'
    return 'color: %s' % color


pd.DataFrame(covariance_matrix)

pd.DataFrame(np.linalg.pinv(covariance_matrix)).style.applymap(_color_zeros)

# <!-- Create overlap matrix, that counts the number of cases for each combination of class votes for each weak label pair. Dividing by the total amount of cases we get the estimated joint and marginal labeling probabilities. -->

# +
# overlap_matrix = (label_matrix_onehot.T @ label_matrix_onehot)/label_matrix_onehot.shape[0]

# +
# overlap_matrix
# -

# <!-- Each element of mult_matrix is the product of the two classes that are represented by its row and column. Multiplying that by the probabilities and taking the sum per weak label pair we get their joint expectations. -->

# +
# mult_matrix = (np.tile(np.unique(label_matrix), nr_wl)).reshape(1,-1).T @ (np.tile(np.unique(label_matrix), nr_wl)).reshape(1,-1)
# pre_sum = overlap_matrix * mult_matrix

# +
# H,W = nr_wl, nr_wl
# m,n = pre_sum.shape
# expectations = np.einsum('ijkl->ik',pre_sum.reshape(H,m//H,W,n//W))

# +
# expectations

# +
# marginal_exp = (np.tile(np.unique(label_matrix), nr_wl)*np.diag(overlap_matrix)).reshape(nr_wl, y_dim).sum(axis=1)

# +
# marginal_exp

# +
# covariance_matrix = expectations.copy()

# +
# mask = np.zeros(covariance_matrix.shape)

# for i in range(len(covariance_matrix)):
#     for j in range(len(covariance_matrix)):
#         covariance_matrix[i,j] = expectations[i,j]-marginal_exp[i]*marginal_exp[j]
#         if i != j:
#             mask[i,j] = 1

# +
# covariance_matrix
# -

# <!-- Should figure out: is this really what they mean in the Snorkel paper, and how does it work when there are >2 classes? -->

# +
# pd.DataFrame(np.linalg.pinv(overlap_matrix))
# -

# We have a setting of a binary target variable and a singleton separator set, so we have a rank-one matrix completion problem that we need to solve. So we have $z \in \mathbb{R}^{d_O x 1}$

# We can also compute the covariance of Y since we have uniform class balance.

# exp_Y and cov_S should come from prior knowledge, can't use the information from y because it is a latent variable

# The marginal expectation of each weak label is just the sum product of each class label and the probability for the weak label to vote for that class.

# $\text{marginal expectation} = E\left[\lambda_{i}\right] =\sum_{k} k \cdot P\left(\lambda_{i}=k\right)$
#

# $\text{expectations} = E\left[\lambda_{i} \lambda_{j}\right]=\sum_{k, l} k l P\left(\lambda_{i}=k, \lambda_{j}=l\right)$

# The covariance matrix is then computed by taking the joint expectations subtracted by the product of the two marginal expectations of the weak label pair.

# $\begin{aligned}
# \operatorname{Cov}\left(\lambda_{i}, \lambda_{j}\right) &=E\left[\left(\lambda_{i}-\mu_{\lambda_{i}}\right)\left(\lambda_{j}-\mu_{\lambda_{j}}\right)\right] \\
# &=\sum_{k, l}\left(k-\mu_{\lambda_{i}}\right)\left(l-\mu_{\lambda_{j}}\right) P_{\lambda_{i}, \lambda_{j}}(k, l) \\
# &=E\left[\lambda_{i} \lambda_{j}\right]-\mu_{\lambda_{i}} \mu_{\lambda_{j}}
# \end{aligned}$

marginal_Y = [0.5, 0.5]
sq_exp = (np.unique(label_matrix)**2*marginal_Y).sum()
exp_Y = (np.unique(label_matrix)*marginal_Y).sum()
cov_S = sq_exp - exp_Y*exp_Y

# The expectations of the observable part are the means of the indicator variable columns

exp_O = label_matrix_onehot.mean(axis=0)[:-y_dim]

exp_O

# Filter out the observable part of the covariance matrix

cov_O = covariance_matrix[:-y_dim, :-y_dim]

# Create graph structure mask:
# Put ones wherever the weak labels are conditionally independent, this indicates where we know the inverse covariance should be zero so use this to solve for $z$

# +
mask = np.ones(cov_O.shape)

for i in range(int(cov_O.shape[0]/y_dim)):
    mask[i*y_dim:(i+1)*y_dim,i*y_dim:(i+1)*y_dim] = 0
# -

mask

pd.DataFrame(cov_O).style.applymap(_color_zeros)

# # Find optimal z and mu

# We want to find z such that: $\hat{z}=\operatorname{argmin}_{z}\left\|\Sigma_{O}^{-1}+z z^{T}\right\|_{\Omega}$

z = np.random.normal(0, 1, (y_dim*(nr_wl-1), 1)) # random initialization


def loss_func(z):
    return torch.norm((torch.Tensor(np.linalg.pinv(cov_O)) + z@z.T)[torch.BoolTensor(mask)]) ** 2


z = nn.Parameter(torch.Tensor(z), requires_grad=True)

n_epochs = 200
lr = 1e-1

optimizer = torch.optim.Adam({z}, lr=lr)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_func(z)
    loss.backward()
    optimizer.step()

loss

# The two matrices below should be opposite sign (add up to around 0) where the weak labels are conditionally independent:

z@z.T

torch.Tensor(np.linalg.pinv(cov_O))

# Now, compute mu from the optimal $z$ that was found above

z = z.detach().numpy()


def calculate_mu(z, cov_S, cov_O, exp_Y, exp_O):
    c = 1/cov_S * (1 + z.T@cov_O@z)
    cov_OS = cov_O @ z/np.sqrt(c)
    joint_cov_OS = np.concatenate((-1*cov_OS, cov_OS), axis=1)
    if joint_cov_OS[0,1] > joint_cov_OS[0,0]:
        joint_cov_OS = joint_cov_OS.T[::-1].T
    print("cov_OS:", joint_cov_OS)
    return (joint_cov_OS + (exp_O.reshape(-1,1) @ exp_Y.reshape(1,-1))) / np.tile(exp_Y, (y_dim*(nr_wl-1), 1))


pd.DataFrame(covariance_matrix)

# The covariances of the observed and separator sets agree with the known covariances in the matrix above

mu = calculate_mu(z, cov_S, cov_O, exp_Y, exp_O)

# |   |fp|fn|
# |---|---|---|
# |wl1|0.1|0.2|
# |wl2|0.05|0.4|
# |wl3|0.2|0.3|

# Because we know the fp/fn rates of each of the weak labels, we can compute their probabilities of labeling each value given Y:
#
# |   |P_0|P_1|
# |---|---|---|
# |wl1|0.9|0.2|
# |   |0.1|0.8|
# |wl2|0.95|0.4|
# |   |0.05|0.6|
# |wl3|0.8|0.3|
# |   |0.2|0.7|

# Computed mu agrees with expected values above

# Conditional probability $P(\lambda|Y)$

pd.DataFrame(mu)

# +
# exp_mu = np.zeros((mu.shape[0], 2))
# sum_0 = label_matrix_onehot[y == 0].sum(axis=0)
# sum_1 = label_matrix_onehot[y == 1].sum(axis=0)
# for i in range(nr_wl-1):
#     exp_mu[i*y_dim, 0] = sum_0[i*y_dim]/sum_0[nr_wl+y_dim]
#     exp_mu[i*y_dim+1, 1] = sum_1[i*y_dim+1]/sum_1[nr_wl+y_dim+1]

# +
# exp_mu
# -

# Joint probability $P(Y,\lambda)$

P_Ylam = mu * np.tile(exp_Y, (y_dim*(nr_wl-1), 1))

P_Ylam

# +
# P_Y_lam = P_Ylam / np.tile(exp_O, (y_dim, 1)).T

# +
# P_Y_lam
# -

# # Compute weak label accuracies

weights = np.zeros((nr_wl-1, 1))
for i in range(nr_wl-1):
    weights[i] = P_Ylam[i*y_dim, 0] + P_Ylam[i*y_dim+1, 1]

weights

label_matrix_onehot[:,:-2]
true_accuracies = np.zeros((nr_wl-1, 1))
for i in range(nr_wl-1):
    true_accuracies[i] = (label_matrix_onehot[:,i*y_dim:i*y_dim+y_dim] == label_matrix_onehot[:,-y_dim:]).mean()


# This looks pretty good, similar to what we found above!

true_accuracies

pd.DataFrame(np.concatenate([true_accuracies, weights], axis=1), columns=["True accuracies", "Learned weights"])

# # Compare to Snorkel

# Let's see what Snorkel does with our synthetic data

metrics = ["accuracy","f1"]

label_model = LabelModel(cardinality=y_dim)
label_model.fit(label_matrix[:,:-1], class_balance=[0.5,0.5])
Y_hat, preds = label_model.predict(label_matrix[:,:-1], return_probs=True)

# Snorkel finds similar values to the mu computed above

label_model.mu.detach().numpy()

# Snorkel also finds similar weights

label_model.get_weights()

label_model.score(label_matrix[:,:-1], df["y"], metrics=metrics)

# # Compute probabilistic labels from label model

label_matrix_onehot = label_matrix_onehot[:,:-y_dim]

X_Z = np.zeros((N_total, y_dim))
for i in range(y_dim):
    X_Z[:,i] = np.prod(np.tile(mu[:,i], (N_total, 1)), axis=1, where=(label_matrix_onehot == 1.0))

Z = X_Z.sum(axis=1)

Y_probs = X_Z/np.tile(Z, (y_dim, 1)).T

# # Train final model on probabilistic labels

# For the next step, we want to train a discriminative model on the probabilistic label output. Sklearn logistic regression works only with the hard labels:

# +
reg = LogisticRegression(solver="lbfgs")
reg.fit(np.concatenate([X[:,0].reshape(-1, 1), X[:,1].reshape(-1, 1)], axis=1), np.argmax(np.around(Y_probs),axis=-1))

print(reg.intercept_)
print(reg.coef_)


# -

# So here we implement logistic regression in PyTorch. The regular cross entropy loss also requires hard labels, so we implement an adjusted loss function that deals with probabilistic labels:

# $loss = \sum_y s_y * - y log p$

def cross_entropy_soft_labels(predictions, targets):
    y_dim = targets.shape[1]
    loss = torch.zeros(predictions.shape[0])
    for y in range(y_dim):
        loss_y = F.cross_entropy(predictions, predictions.new_full((predictions.shape[0],), y, dtype=torch.long), reduction="none")
        loss += targets[:,y] * loss_y
        
    return loss.mean()


batch_size = 32
input_dim = 2 # number of features
output_dim = 2 # number of classes
lr = 1e-3
n_epochs = 200

# Convert our data to tensor format and create dataloader to allow for dealing with batches

train = torch.Tensor(df[["x1", "x2"]].values)
# target = torch.LongTensor(np.argmax(np.around(Y_probs),axis=-1))
target = torch.Tensor(Y_probs)
train_tensor = torch.utils.data.TensorDataset(train, target) 
train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


model = LogisticRegression(input_dim, output_dim)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train model for a couple of epochs, using the soft label cross entropy loss implementation.

# +
model.train()

for epoch in range(n_epochs):
    for i, (features, soft_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(features)
        loss = cross_entropy_soft_labels(logits, soft_labels)
#         loss = F.cross_entropy(logits, soft_labels)
        loss.backward()
        optimizer.step()
# -

loss

# See what the model does on our train set. We should actually use a test set here

model.eval()
logits = model(train)

probs = F.softmax(logits, dim=1)

# Overall accuracy probabilistic labels

(np.argmax(np.around(Y_probs),axis=-1) == np.array(df["y"])).sum()/N_total

# Overall accuracy final model output

(np.argmax(np.around(probs.detach().numpy()),axis=-1) == np.array(df["y"])).sum()/N_total

# +
# coefs = [list(model.parameters())[1][1].tolist()] + list(model.parameters())[0][1,:].tolist()

# +
# x_dec = np.linspace(centroid_2[0]-2, centroid_1[0]+2, 1000)

# y_dec = (coefs[0] + coefs[1]*x_dec)/-coefs[2]
# -

# Plot the estimated probabilities

fig = px.scatter(x=X[:,0], y=X[:,1], color=Y_probs[:,1])
fig.update_layout(xaxis=dict(range=[centroid_2[0]-2,centroid_1[0]+2]), yaxis=dict(range=[centroid_2[1]-2,centroid_1[1]+2],scaleanchor="x", scaleratio=1), width=700, height=700, xaxis_title="x1", yaxis_title="x2")
fig.show()

fig = px.scatter(x=X[:,0], y=X[:,1], color=probs[:,1].detach().numpy())
fig.update_layout(xaxis=dict(range=[centroid_2[0]-2,centroid_1[0]+2]), yaxis=dict(range=[centroid_2[1]-2,centroid_1[1]+2],scaleanchor="x", scaleratio=1), width=700, height=700, xaxis_title="x1", yaxis_title="x2")
fig.show()

# # Add active learning

# Add active learning weak label as additional column of abstains to start with

wl_al = np.full_like(df["y"], -1)
# wl_al = np.array(df["y"]).reshape(-1,1)
L = np.concatenate([label_matrix[:,:-1], wl_al.reshape(len(wl_al),1)], axis=1)

y_set = np.unique(L) # array of classes
if y_set[0] == -1:
    y_set = y_set[1:]
y_dim = len(y_set) # number of classes
nr_wl = L.shape[1] # number of weak labels


def calculate_mu(z, cov_S, cov_O, exp_Y, exp_O):
    """Compute mu from optimal z"""
    
    c = 1/cov_S * (1 + torch.mm(torch.mm(z.T, cov_O), z))
    cov_OS = torch.mm(cov_O, z/torch.sqrt(c))
    if cov_OS[0] < 0:
        joint_cov_OS = torch.cat((-1*cov_OS, cov_OS), axis=1)
    else:
        joint_cov_OS = torch.cat((cov_OS, -1*cov_OS), axis=1)
#     if joint_cov_OS[0,1] > joint_cov_OS[0,0]:
#         joint_cov_OS = joint_cov_OS.T[::-1].T
#     print(joint_cov_OS)
    return (joint_cov_OS + torch.Tensor(exp_O.reshape(-1,1) @ exp_Y.reshape(1,-1))) / torch.Tensor(np.tile(exp_Y, (joint_cov_OS.shape[0], 1)))


def calculate_cov(z, cov_S, cov_O):
    c = 1/cov_S * (1 + torch.mm(torch.mm(z.T, cov_O), z))
    cov_OS = torch.mm(cov_O, z/torch.sqrt(c))
    if cov_OS[0] < 0:
        joint_cov_OS = torch.cat((-1*cov_OS, cov_OS), axis=1)
    else:
        joint_cov_OS = torch.cat((cov_OS, -1*cov_OS), axis=1)
    return joint_cov_OS


def loss_probs(mu, t: float = 10000) -> torch.Tensor:
    m_zeros = torch.zeros(mu.shape)
    m_ones = torch.ones(mu.shape)
#     print((m_zeros - mu)[mu < 0])
    return t * (torch.norm((m_zeros - mu)[mu < 0]) ** 2 + torch.norm((mu - m_ones)[mu > 1]) ** 2)


# +
def loss_pk_mu(mu, s: float = 1e1) -> torch.Tensor:
    r"""Loss from prior knowledge"""
    O_dim, y_dim = mu.shape

    m = torch.zeros([O_dim, y_dim])
    m[:-mu.shape[1]] = 1
#     m[6,0] = 0
#     m[7,1] = 0
    
    mu_known = torch.zeros(mu.shape)
    for i, k in enumerate(range(O_dim - y_dim, O_dim)):
        mu_known[k, i] = 1

#     mu_known[6,0] = 0.0014
#     mu_known[7,1] = 0.0028
        
    return s * torch.norm((mu - mu_known)[m.type(torch.BoolTensor)]) ** 2


# -

def loss_pk_cov(cov_OS, cov_AL, s: float = 0.5):
    O_dim, y_dim = cov_OS.shape
    
    cov_OS_known = torch.zeros(cov_OS.shape)
    cov_OS_known[-y_dim:,:] = cov_AL
    
    m = torch.ones([O_dim, y_dim])
    m[:-y_dim] = 0
#     print(cov_OS)
    
    return s * torch.norm((cov_OS - cov_OS_known)[m.type(torch.BoolTensor)]) ** 2


def loss_func(z, cov_S, cov_O, exp_Y, exp_O, mask, cov_AL):
    loss_1 = torch.norm((torch.Tensor(np.linalg.pinv(cov_O)) + z@z.T)[torch.BoolTensor(mask)]) ** 2
    int_mu = calculate_mu(z, cov_S, cov_O, exp_Y, exp_O)
    int_cov = calculate_cov(z, cov_S, cov_O)
#     print("1:", loss_1)
#     print("2:", loss_pk_mu(torch.Tensor(int_mu)))
#     print("3:", loss_probs(torch.Tensor(int_mu)))
#     print("4:", loss_pk_cov(int_cov, cov_AL))
    return loss_1 + loss_pk_cov(int_cov, cov_AL) #+ loss_probs(torch.Tensor(int_mu))


# +
def fit_predict_AL(label_matrix, class_balance):
    """Fit label model and predict training labels"""
    
    y_set = np.unique(label_matrix) # array of classes
    if y_set[0] == -1:
        y_set = y_set[1:]
    y_dim = len(y_set) # number of classes
    nr_wl = label_matrix.shape[1] # number of weak labels
    
    label_matrix_onehot = (y_set == label_matrix[...,None])*1
    lm_sh = label_matrix_onehot.shape
    label_matrix_onehot = label_matrix_onehot.reshape(lm_sh[0],lm_sh[1]*lm_sh[2])
    
    exp_O = label_matrix_onehot.mean(axis=0)
    cov_O = np.cov(label_matrix_onehot.T)
    
    sq_exp = (y_set**2*class_balance).sum()
    exp_Y = (y_set*class_balance).sum()
    cov_S = sq_exp - exp_Y*exp_Y
    
    N_total = label_matrix.shape[0]
    cov_AL = np.diag(label_matrix_onehot[:,-y_dim:].sum(axis=0)/(np.array(class_balance)*N_total)*cov_S)
    cov_AL[0,1] = -cov_AL[0,0]
    cov_AL[1,0] = -cov_AL[1,1]
    
    mask = np.ones(cov_O.shape)
    for i in range(int(cov_O.shape[0]/y_dim)):
        mask[i*y_dim:(i+1)*y_dim,i*y_dim:(i+1)*y_dim] = 0
        for j in range(int(cov_O.shape[0]/y_dim)):
            if (i == 1 and j == 2):
                mask[i*y_dim:(i+1)*y_dim,j*y_dim:(j+1)*y_dim] = 0
                mask[j*y_dim:(j+1)*y_dim,i*y_dim:(i+1)*y_dim] = 0
                
    mask[:,-y_dim:] = 0
    mask[-y_dim:,:] = 0
    
    z = np.random.normal(0, 1, (y_dim*nr_wl, 1)) # random initialization
    z = nn.Parameter(torch.Tensor(z), requires_grad=True)
    n_epochs = 200
    lr = 1e-2
    optimizer = torch.optim.Adam({z}, lr=lr)
    for epoch in range(n_epochs):
       
        optimizer.zero_grad()
        loss = loss_func(z, cov_S, torch.Tensor(cov_O), exp_Y, exp_O, mask, torch.Tensor(cov_AL))
        loss.backward()
        optimizer.step()
        
    
    mu = calculate_mu(z, cov_S, torch.Tensor(cov_O), exp_Y, exp_O)

    mu_eps = min(0.01, 1 / 10 ** np.ceil(np.log10(N_total)))
    mu = mu.clamp(mu_eps, 1 - mu_eps)

#     mu[-y_dim:,:] = mu_eps
#     for i, k in enumerate(range(mu.shape[0] - y_dim, mu.shape[0])):
#         mu[k, i] = 1-mu_eps

    mu = mu.detach().numpy()
    
    X_Z = np.zeros((lm_sh[0], y_dim))
    for i in range(y_dim):
        X_Z[:,i] = np.prod(np.tile(mu[:,i], (N_total, 1)), axis=1, where=(label_matrix_onehot == 1.0))
    
        
    Z = X_Z.sum(axis=1)
    probs = X_Z/np.tile(Z, (y_dim, 1)).T
    
    return mu, np.argmax(np.around(probs),axis=-1), probs
# -

_, Y_probs = fit_predict(L, [0.5, 0.5], np.array([0, 1]))

# +
it = 10
accuracies = []

for i in range(it):
    Y_hat, preds = fit_predict(L, [0.5, 0.5], np.array([0, 1]))
    
    # Find data points the model is least confident about
    abs_diff = np.abs(preds[:,1] - preds[:,0])
    minimum = min(j for i,j in enumerate(abs_diff) if L[i,nr_wl-1] == -1)
    indices = [j for j, v in enumerate(abs_diff) if v == minimum]
    
    # Make really random
    random.seed(random.SystemRandom().random())
    
    # Pick a random point from least confident data points and set to true value for AL weak label in label matrix
    sel_idx = random.choice(indices)
    print("Iteration:", i+1, " Label combination", L[sel_idx,:nr_wl], " True label:",y[sel_idx], "Estimated label:", Y_hat[sel_idx], " selected index:", sel_idx)
    
    L[sel_idx, nr_wl-1] = y[sel_idx]
    before = preds[sel_idx, :]
    
    # Fit label model on refined label matrix
    Y_hat, preds = fit_predict(L, [0.5, 0.5], np.array([0, 1]))
    after = preds[sel_idx]
    print("Before:", before, "After:", after)
    print("")
    
    accuracy = (Y_hat == np.array(df["y"])).sum()/N_total
    accuracies.append(accuracy)
# -
accuracy_prob = (np.argmax(np.around(Y_probs[1]),axis=-1) == np.array(df["y"])).sum()/N_total

# +
x = list(range(len(accuracies)))

fig = go.Figure(data=go.Scatter(x=x,y=accuracies))
fig.add_trace(go.Scatter(x=x, y=np.repeat(accuracy_prob, len(accuracies))))

from IPython.display import HTML, display
display(HTML(fig.to_html()))
# -

fig = px.scatter(x=X[:,0], y=X[:,1], color=preds[:,1])
fig.update_layout(xaxis=dict(range=[centroid_2[0]-2,centroid_1[0]+2]), yaxis=dict(range=[centroid_2[1]-2,centroid_1[1]+2],scaleanchor="x", scaleratio=1), width=700, height=700, xaxis_title="x1", yaxis_title="x2")
fig.show()


target = torch.Tensor(preds)
train_tensor = torch.utils.data.TensorDataset(train, target) 
train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)

model = LogisticRegression(input_dim, output_dim)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# +
model.train()

for epoch in range(n_epochs):
    for i, (features, soft_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(features)
        loss = cross_entropy_soft_labels(logits, soft_labels)
#         loss = F.cross_entropy(logits, soft_labels)
        loss.backward()
        optimizer.step()
# -

model.eval()
logits = model(train)

probs_2 = F.softmax(logits, dim=1)

(np.argmax(np.around(probs_2.detach().numpy()),axis=-1) == np.array(df["y"])).sum()/N_total


