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

# This notebook essentially implements robust PCA to attempt to recover the dependency structure of weak labellers and true label from the data.  
#
# This can be contrasted against the Metal implementation in this notebook:  
# https://github.com/HazyResearch/metal/blob/cb_deps/tutorials/Learned_Deps.ipynb

#

# # Generate synthetic data

# +
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from torch import nn, optim


# -


def generateSynthetic(coefs, wlrates, n=10000, seed=None):
    """Class to generate synthetic two-covariate logistic-distributed datasets from specified decision boundary.
    
    Args:
    n (int): Number of samples to generate
    coefs (list): Regression coefficients from which to generate data.
                  Format: (intercept, coef for x1, coef for x2)
                  Samples will then be generated with
                  p = sigmoid(z) = sigmoid(coefs[0] + coefs[1]*x1 + coefs[2]*x2)
    wlrates (list of 2-tuples): For each weak learner specify its desired (false positive, false negative) rates
                  
    Output: A dataframe containing covariates, Bernoulli probability, label, and weak labels"""
        
    np.random.seed(seed=seed)
        
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    def softmax(z):
        z_x = np.exp(z - np.max(z))
        return z_x / np.sum(z_x, axis=1, keepdims=True)
    
    x1 = np.random.normal(loc=0, scale=1, size=int(n))
    x2 = np.random.normal(loc=0, scale=1, size=int(n))
    
    z = coefs[0] + coefs[1]*x1 + coefs[2]*x2
    
    p = sigmoid(z)

    y = np.random.binomial(1, p, n)
    
    df = pd.DataFrame({'x1':x1, 'x2':x2, 'p':p, 'y':y})
    
    
    def random_LF(y, fp, fn):
#        if seed==None:
#            threshold = np.random.uniform()
#        else:
#            np.random.seed(seed=seed+1)
#            threshold = np.random.uniform()
        threshold = np.random.uniform()
        if y == 1 and threshold < fn:
            y = 0 
        elif y == 0 and threshold < fp:
            y = 1
        return y

    for idx, wl in enumerate(wlrates):
        df.loc[:, "wl"+str(idx+1)] = df['y'].apply(lambda row: random_LF(row, fp=wl[0], fn=wl[1]))
    
    np.random.seed(seed=None)
    
    return(df)

# Fix the seed to make results reproducible (on the same machine at least).

seed = 5
np.random.seed(seed=seed)

# Set the coefficients for the logistic function data generation
coefs = [1, -2, 3]

# Instead of creating a handful of weak learners by hand, the code below automatically generates any number of weak learners, with randomly set false positive and false negative rates.  
# This is so we can explore how the structure learning varies with number of weak learners.

# +
#wlrates = [(0.1,0.2), (0.05,0.4), (0.2,0.3)]

num_wl = 9

rates = np.random.uniform(0, 1, 2*num_wl)

# Verify how rates are split into false positive and false negative (taking every other element)
#print(rates)
#print(rates[0:2*num_wl:2])
#print(rates[1:2*num_wl:2])

fprates = rates[0:2*num_wl:2]
fnrates = rates[1:2*num_wl:2]

wlrates = []
for i in range(len(fprates)):
    wlrates.append((fprates[i], fnrates[i]))



nrows=10000

df = generateSynthetic(coefs=coefs, wlrates=wlrates, n=nrows, seed=seed)
df


# -

# Now we will induce a dependency between two of the weak labellers.

def makeAandBdependent(inputs):
    """Induce a depedency between a pair of inputs.
    
    Input: A list with two elements, both binary integers
    Output: A list with two elements, both binary integers """
    if inputs[0]+inputs[1] == 0:
        return pd.Series([inputs[0], inputs[1]], dtype='int')
    elif inputs[0]+inputs[1] == 1:
        rand = np.random.uniform(1)
        if rand<0.4:
            return pd.Series([1,0], dtype='int')
        else:
            return pd.Series([1,1], dtype='int')
    elif inputs[0]+inputs[1] == 2:
        return pd.Series([0,1], dtype='int')


# +
### Code to verify by eye that makeAandBdependent() works as expected
#testA = np.random.binomial(n=1, p=0.5, size=20)
#testB = np.random.binomial(n=1, p=0.5, size=20)

#testdf = pd.DataFrame({'input1':testA, 'input2':testB})
#testdf.join(testdf.apply(lambda row: makeAandBdependent(row[['input1','input2']]), axis=1))
# -

df = df.join(df.apply(lambda row: makeAandBdependent(row[['wl8','wl9']]), axis=1))
df.rename(columns={0:'wl8dep', 1:'wl9dep'}, inplace=True)
df

# Checking if we recover the coefficients we generated the data with, as sanity check.

# +
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(solver="lbfgs")
reg.fit(df[['x1','x2']].values, df['y'].values)

print(np.concatenate([reg.intercept_, reg.coef_.flatten()]))
# -

coefs

# Checking a few other basic statistics of the data

df['y'].mean()

print("Accuracy wl1:", (df["y"] == df["wl1"]).sum()/df.shape[0])

print("Accuracy wl2:", (df["y"] == df["wl2"]).sum()/df.shape[0])

print("Accuracy wl3:", (df["y"] == df["wl3"]).sum()/df.shape[0])

df

# # Structure Learning (Robust PCA)

import torch
from torch import nn, optim

# +
inputdf = df.filter(regex='wl*|y') # Get all columns which contain string 'wl' or 'y'
inputdf = inputdf.reindex(sorted(inputdf.columns), axis=1) # sort columns alphabetically so true label is last one
inputdf

#inputdf = df[["wl1", "wl2", "wl3", "y"]].values
#inputdf = df[["wl1dep", "wl2dep", "wl3", "y"]].values
# -

inputdf = inputdf.values

# Choose whether you want to one-hot encode the weak labels and true label.

# +
one_hot = False

if one_hot:
    inputdf = (np.array([0,1]) == inputdf[...,None])*1
    lm_sh = inputdf.shape
    inputdf = inputdf.reshape(lm_sh[0],lm_sh[1]*lm_sh[2])
    
    y_dim = 2
else:
    y_dim = 1
# -

inputdf = np.cov(inputdf.T)

# Also one form of robust PCA seems to require the covariance matrix as input, and another form seems to require the *inverse* covariance matrix (aka the precision matrix), so choose that here.

# +
invert = False

if invert==True:
    inputdf = np.linalg.pinv(inputdf)
# -

# In Section 3.3 of this paper (page 4, top right) they give a bound for the minimum number of weak labellers (sources) required for the structure learning problem to be identifiable, we compute this below for our case.  
# http://proceedings.mlr.press/v97/varma19a.html  
#
# For the 'maximum dependency degree ', d, I used a value of 2, interpreting this to be the degree in the graph theory sense (the total number of nodes connected to a given node). This means that each weak learner would be connected at most to one other weak learner, and Y.

inputdf.shape

# +
c = inputdf[:-y_dim, :-y_dim]

cmin = np.amin(c)
cmax = np.amax(c)

c.shape

# +
a = inputdf[:(inputdf.shape[0]-y_dim), (inputdf.shape[1]-y_dim):]

amin = np.amin(a)
amax = np.amax(a)

a.shape
# -

inputdf

c

a

amin

# And the number of weak labellers we would need to make the structure learning problem we have here identifiable is....

# +
maxdegree = 2

40.96*maxdegree**2*((cmax*amax)/(cmin*amin))**2
# -

# However note that the number above can fluctuate significantly, and in particular if you add weak learners it will change as well, so doesn't look to me like this number estimates anything robustly.

# ## Fitting Robust PCA

# You will see a 'lgrn' parameter below which is commented out, I used it to try to impose the input=S+L constraint in the optimisation, using this parameter as the Lagrange multiplier, but the optimisation would then fail, so removed it for now.

inputdf = inputdf[:-1, :-1]

# +
S = nn.Parameter(torch.rand(inputdf.shape)).float()
L = nn.Parameter(torch.rand(inputdf.shape)).float()

#lgrn = nn.Parameter(torch.rand([]), requires_grad=True).float()
# -

inputtorch = torch.from_numpy(inputdf).float()


# It is not yet entirely clear to me which loss they are using to do the Robust PCA.  
#
# One form of the loss function for robust PCA is described on page 64 (page 78 in the PDF) of Ratner's thesis:  
# https://ajratner.github.io/assets/papers/thesis.pdf  
# And is also given as the loss function in Section 3.3 of this paper
# http://proceedings.mlr.press/v97/varma19a.html  
# This is a constrained optimisation on the covariance matrix, using the trace.  
#
#
# However there is also another form of the loss function, this one is a squared loss on the *inverse* covariance matrix (and apparently is unconstrained optimisation) on slide 17 here:  
# https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AACztdxm-GTWxOkA7PfX2ooaa/Day%201?dl=0&preview=04_Theory_Apps.pdf&subfolder_nav_tracking=1
#
# The form they use in their actual Metal code seems to be the first form, as can be seen in the learn_structure() function in this notebook:  
# https://github.com/HazyResearch/metal/blob/cb_deps/tutorials/Learned_Deps.ipynb

# Are the two forms of loss above actually equivalent in some way? I don't know yet.  
# The second form seems easier to implement in PyTorch since I'm not yet sure how to implement constraints in the optimisation (using a Lagrange multiplier seemed to fail).  
#
# So we could simply use the function in the Metal notebook, which uses CVPX to do the optimisation.

# In terms of the motivation for why the loss looks like that:  
# - The L1 term on S is clearly to induce sparsity, because we assume the underlying graphs will be nowhere near fully-connected.  
# - The nuclear norm term on L is to minimise the rank of L, since the nuclear norm is related to the rank (see Theorem 2.2 from this paper: https://arxiv.org/abs/0706.4138 )

# +
def get_loss(inputcovdf, lambdan=1, gamma=1):
    
#    if invert==False:
#        # Loss given in Section 3.3 of this paper, for covariance matrix
#        # http://proceedings.mlr.press/v97/varma19a.html
#        diffs = 0.5*torch.trace((S-L)*inputcovdf*(S-L)) - torch.trace(S-L)
#        diffs += (gamma*torch.norm(S, p=1) + lambdan*torch.norm(L, p='nuc'))
#    else:
#        # Loss given on slide 17 here, for *inverse* covariance matrix
#        # https://www.dropbox.com/sh/ipxmm6twu4p2qo1/AACztdxm-GTWxOkA7PfX2ooaa/Day%201?dl=0&preview=04_Theory_Apps.pdf&subfolder_nav_tracking=1
#        diffs = torch.norm(inputcovdf - (S+L), p=2) + lambdan*torch.norm(S, p=1) + gamma*torch.norm(L, p='nuc')
    
    
    diffs = torch.norm(inputcovdf - (S+L), p=2).clone()
    diffs = 0.5*torch.trace((S-L)*inputcovdf*(S-L)) - torch.trace(S-L)#.clone()
    diffs += lambdan*(gamma*torch.norm(S, p=1) + torch.norm(L, p='nuc'))
    #diffs += lgrn*torch.norm(inputcovdf-(S+L), p=2)
    
    
    #prefactor = torch.prod(torch.tensor(inputcovdf.shape))

    # I don't know how to constrain the optimization so that S-L is positive definite (symmetric and positive eigenvalues)
    #   and L is positive semi-definite (symmetric and non-negative eigenvalues)
    # https://mathworld.wolfram.com/PositiveDefiniteMatrix.html#:~:text=A%20Hermitian%20(or%20symmetric)%20matrix,part%20has%20all%20positive%20eigenvalues.
    
    # However I can at least make them symmetric, below
    
    # We expect the Sparse matrix with the graph dependency to be symmetric
#    diffs += prefactor*torch.norm(S - torch.transpose(S, 0, 1), p=2)
    
#    diffs += prefactor*torch.norm((S-L) - torch.transpose(S-L, 0, 1), p=2)
#    diffs += prefactor*torch.norm(L - torch.transpose(L, 0, 1), p=2)
    
    return diffs

# +
optimizer = optim.LBFGS([S,L], lr=0.01, max_iter=1e4)
#optimizer = optim.LBFGS([S,L,lgrn], lr=0.01, max_iter=1e4)

gamma = 1e-8
lambdan = 1/np.sqrt(inputtorch.shape[0])

# The closure computes the loss
def closure():
    optimizer.zero_grad()
    loss = get_loss(inputtorch, lambdan=lambdan, gamma=gamma)
    loss.backward()
    #with torch.no_grad():
    #    for param in model.parameters():
    #            param.clamp_(0, 1)
    return loss

# Perform optimizer step
optimizer.step(closure)


Soutput = S.detach().numpy()
Loutput = L.detach().numpy()
# -

# Inspect the outputs to do sanity checks. S should be sparse, numbers overall should not be large (otherwise possible convergence issue)

Soutput

Loutput

inputdf

# I would expect the terms below to be very small since otherwise we have not truly decomposed the input as L+S. It isn't, so something is wrong here (the loss function most likely).  
#
# Either that, or it failed because the data cannot be separated as L+S for L low rank and S sparse. The reason for this is that the low rank component *cannot also be sparse*, as the one of the authors of the Robust PCA paper explains here (the whole video is worth a watch):  
# https://youtu.be/DK8RTamIoB8?t=1490  
#
# The same explanation for when this method fails is given in the first paragraph of Section 3.3 in http://proceedings.mlr.press/v97/varma19a.html

inputdf - (Soutput+Loutput)

# Note that the matrix S isn't truly sparse at this point since no elements are exactly zero. In order to make it truly sparse we need to set a threshold below which we set values to zero.  
#
# For the time being I noticed by inspection that the diagonal elements of S -which should be non-zero- were often orders of magnitude larger than the off-diagonal parts which should be zero, so I use the order of magnitude to set the threshold with respect ot diagonal elements, as will be explained below. This is arbitrary and can/should be changed/improved.
#
# IDEA FOR LATER: Insert the true known dependency structure into the loss function and learn this threshold parameter as well together with S and L.

import math
def orderOfMagnitude(number):
    return math.floor(math.log(abs(number), 10))


orderOfMagnitude(0.09)


def _color_zeros(threshold):
    def _color_zeros_by_val(val):
        #color = 'red' if np.abs(val) < threshold else 'green'
        color = 'red' if orderOfMagnitude(val) < threshold else 'green'
        return 'color: %s' % color
    return _color_zeros_by_val


# Currently setting the threshold as the order of magnitude of the first element of the diagonal in the observable part of the (inverse) covariance matrix.  
#
# Another way to set the scale would be:  
#
# The eigenvalues of a matrix are a natural scale for the matrix. The trace of a matrix is equal to the sum of the eigenvalues. Therefore we use the trace divided by the number of elements in the matrix (to normalise across different matrix sizes) as a natural threshold.  
#
# Any number of other options might work better as well.

def getThreshold(arr):
    # Get the order of magnitude of the largest element in the *observable* part of the cov matrix
    #return orderOfMagnitude(np.amax(arr[:-y_dim, :-y_dim]))
    #Get the order of magnitude of the first element in the *observable* part of the cov matrix
    return orderOfMagnitude(arr[0, 0])


# +
thresh1 = getThreshold(inputdf)

# Setting the threshold with respect to the mean of the diagonal elements

#meanforthresh1 = np.unique(inputdf[np.where(~np.eye(inputdf.shape[0],dtype=bool))]).mean()
#sdforthresh1 = np.unique(inputdf[np.where(~np.eye(inputdf.shape[0],dtype=bool))]).std()
#thresh1 = meanforthresh1 + 2*sdforthresh1

#thresh1 = np.trace(inputdf)/np.prod(inputdf.shape)
# -

pd.DataFrame(inputdf).style.applymap(_color_zeros(thresh1))

# +
thresh2 = getThreshold(Soutput)

#meanforthresh2 = np.unique(Soutput[np.where(~np.eye(Soutput.shape[0],dtype=bool))]).mean()
#sdforthresh2 = np.unique(Soutput[np.where(~np.eye(Soutput.shape[0],dtype=bool))]).std()
#thresh2 = meanforthresh2 + 1*sdforthresh2

#thresh2 = np.trace(Soutput)/np.prod(Soutput.shape)
# -

# As you can see below, the diagonal structure is so far recovered, but not the dependency between two of the weak labellers.

pd.DataFrame(Soutput).style.applymap(_color_zeros(thresh2))

# There are three main parameters to adjust to try to get good recovery of structure from this algorithm:  
# - The threshold, which controls how much structure we expect the matrix to have  
# - lambdan, which controls the sparsity we expect the matrix to have  
# - gamma, which controls how big of a part in the completion we expect the low-rank part of the sum (L) to play

# These three parameters need to be explored to find configurations which work reliably across the settings we expect to deploy this method on.

# The dependence on number of weak labellers, type of dependencies between labellers, false positive and negative rates, and the number of data points would also need to be explored.


