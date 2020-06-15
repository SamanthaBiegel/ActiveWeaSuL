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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# +
N = 10000

x1 = np.random.normal(loc=0, scale=1, size=N)
x2 = np.random.normal(loc=0, scale=1, size=N)


# -

def sigmoid(z):
    return 1/(1 + np.exp(-z))


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
reg = LogisticRegression()
reg.fit(np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1),y)

print(reg.intercept_)
print(reg.coef_)

# +
N_1 = 5000
N_2 = 5000
centroid_1 = np.array([0.5, 0.5])
centroid_2 = np.array([-0.5, -0.5])
centroids = np.concatenate([centroid_1.reshape(1,-1), centroid_2.reshape(1,-1)], axis=0)
scale=0.5

x1_1 = np.random.normal(loc=centroid_1[0], scale=scale, size=N_1).reshape(-1, 1)
x2_1 = np.random.normal(loc=centroid_1[1], scale=scale, size=N_1).reshape(-1, 1)
x1_2 = np.random.normal(loc=centroid_2[0], scale=scale, size=N_2).reshape(-1, 1)
x2_2 = np.random.normal(loc=centroid_2[1], scale=scale, size=N_2).reshape(-1, 1)

X_1 = np.concatenate([x1_1, x2_1], axis=1)
X_2 = np.concatenate([x1_2, x2_2], axis=1)
X = np.concatenate([X_1, X_2])

# +
bp1 = centroids.sum(axis=0)/2

diff = centroids[1,:]-centroids[0,:]

slope = diff[1]/diff[0]
perp_slope = -1/slope
b = bp1[1] - perp_slope*bp1[0]


# +
coef = [b, perp_slope, -1]
coef = [4*co for co in coef]

Z = coef[0] + coef[1]*X[:,0] + coef[2]*X[:,1]

h = sigmoid(Z)

y = np.random.binomial(1, h, N_1+N_2)#.astype("str")
# -

x_dec = np.linspace(centroid_2[0]-2, centroid_1[0]+2, 1000)
y_dec = (- coef[0] - coef[1]*x_dec)/coef[2]

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
reg = LogisticRegression()
reg.fit(np.concatenate([X[:,0].reshape(-1, 1), X[:,1].reshape(-1, 1)], axis=1),y)

print(reg.intercept_)
print(reg.coef_)
# -

wl1 = (x1>-0.3)*1
wl2 = (x2<0.3)*1

print("Accuracy wl1:", (y == wl1).sum()/len(y))

print("Accuracy wl2:", (y == wl2).sum()/len(y))

label_matrix = np.concatenate([wl1.reshape(-1,1), wl2.reshape(-1,1)], axis=1)

label_matrix

# +
x_dec = np.linspace(-5,5, 1000)

fig = px.scatter(x=df["x1"], y=df["x2"], color=df["y"].astype("category"))
fig.add_trace(go.Scatter(x=x_dec, y=y_dec, mode="lines", name="decision boundary"))
fig.add_trace(go.Scatter(x=x_dec, y=np.repeat(0.3, N), mode="lines", name="wl2"))
fig.add_trace(go.Scatter(x=np.repeat(-0.3, N), y=x_dec, mode="lines", name="wl1"))
fig.update_layout(xaxis=dict(range=[-4,4]), yaxis=dict(range=[-4,4],scaleanchor="x", scaleratio=1), width=1000, height=1000, xaxis_title="x1", yaxis_title="x2")
fig.show()
# -

plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(label_matrix[df["y"] == 1]).corr(), annot=True)
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(label_matrix[df["y"] == 0]).corr(), annot=True)


def random_LF(y, fp, fn):
    threshold = np.random.uniform()
    
    if y == 1 and threshold < fn:
        y = 0
        
    elif y == 0 and threshold < fp:
        y = 1
    
    return y


df.loc[:, "wl1"] = [random_LF(y, fp=0.1, fn=0.2) for y in df["y"]]
df.loc[:, "wl2"] = [random_LF(y, fp=0.05, fn=0.4) for y in df["y"]]

print("Accuracy wl1:", (df["y"] == df["wl1"]).sum()/len(y))

print("Accuracy wl2:", (df["y"] == df["wl2"]).sum()/len(y))

label_matrix = np.array(df[["wl1", "wl2"]])

plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(label_matrix[df["y"] == 1]).corr(), annot=True)
plt.figure(figsize=(5,3))
sns.heatmap(pd.DataFrame(label_matrix[df["y"] == 0]).corr(), annot=True)

label_matrix.shape

# +
# def to_binary_indicator(label_matrix):

y_dim = len(np.unique(label_matrix))
    
# -

y_dim

label_matrix_onehot = (np.arange(label_matrix.max()+1) == label_matrix[...,None]).astype(int)

label_matrix_onehot.shape

label_matrix

label_matrix_onehot = label_matrix_onehot.reshape(label_matrix_onehot.shape[0],label_matrix_onehot.shape[1]*label_matrix_onehot.shape[2])

overlap_matrix = (label_matrix_onehot.T @ label_matrix_onehot)/label_matrix_onehot.shape[0]
print(overlap_matrix)

label_matrix.sum(axis=0)

df["wl1"].sum()/len(df["wl1"])

# +
y_dim = len(np.unique(label_matrix))
y_set = np.unique(label_matrix)
nr_wl = label_matrix.shape[1]

mult_matrix = (np.tile(np.unique(label_matrix), nr_wl)).reshape(1,-1).T @ (np.tile(np.unique(label_matrix), nr_wl)).reshape(1,-1)
pre_sum = overlap_matrix * mult_matrix
# -

pre_sum

expectations = np.einsum('ijkl->ik',pre_sum.reshape(2,2,2,2))

expectations

marginal_exp = np.diag(expectations)

covariance_matrix = expectations.copy()

for i in range(len(covariance_matrix)):
    for j in range(len(covariance_matrix)):
        if i != j:
            covariance_matrix[i,j] = expectations[i,j]-marginal_exp[i]*marginal_exp[j]

covariance_matrix






