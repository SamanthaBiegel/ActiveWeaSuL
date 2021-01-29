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

# +
coefs = [1, -2, 3]
wlrates = [(0.1,0.2), (0.05,0.4), (0.2,0.3)]
nrows=10000

df = generateSynthetic(coefs=coefs, wlrates=wlrates, n=nrows, seed=5)
df

# +
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(solver="lbfgs")
reg.fit(df[['x1','x2']].values, df['y'].values)

print(np.concatenate([reg.intercept_, reg.coef_.flatten()]))
# -

coefs

df['y'].mean()

# Recall the form of the logistic function which determines the Bernoulli probability which we are sampling from:  
# $p(x ; \mathbf{\beta}) = \frac{1}{1 + e^{\mathbf{\beta}_0 + \mathbf{\beta}\mathbf{X}}} = \frac{1}{1 + e^{\mathbf{\beta}_0}e^{\mathbf{\beta}\mathbf{X}}} \; .$  
#   
# Here you can see that if the intercept term, $\mathbf{\beta}_0$, is large and positive this suppresses the probabilities, and if it is large and negative it pushes the probabilities towards $1$. For fixed coefficients $\mathbf{\beta}$, we can therefore use the intercept term to manipulate the class balance as desired.
#
# You can verify this empirically above by seeing how the class balance shifts as you modify the intercept term.

# +
intercepts = np.linspace(-15, 10, num=35)

intrcptvsclassbal = pd.DataFrame({'intercept':intercepts, 'classbal':np.empty_like(intercepts)})

for idx,i in enumerate(intercepts):
    intrcptvsclassbal.loc[idx, 'intercept'] = i
    
    tempdf = generateSynthetic(coefs=[i, -2, 3], wlrates=wlrates, n=nrows, seed=5)
    intrcptvsclassbal.loc[idx, 'classbal'] = tempdf['y'].mean()
# -

sns.lineplot(x=intrcptvsclassbal['intercept'], y=intrcptvsclassbal['classbal'])

intrcptvsclassbal

print("Accuracy wl1:", (df["y"] == df["wl1"]).sum()/df.shape[0])

print("Accuracy wl2:", (df["y"] == df["wl2"]).sum()/df.shape[0])

print("Accuracy wl3:", (df["y"] == df["wl3"]).sum()/df.shape[0])

df

df['y'].mean()

# +
L = df[["wl1", "wl2", "wl3", "y"]].values
L = L + 1 #Metal requires labels in {1,...,k}, zero is the label for abstains

k = 2 #number of output states, counting abstain state, if present

n,m = L.shape
# -

L

L.shape

# Break down label matrix into a tensor where each slice on the first index is a matrix and corresponds to a value of $y$, with entries equal to 1 if that value of $y$ is present, and 0 otherwise

LY = np.array([np.where(L == y, 1, 0) for y in range(1, k + 1)])

L

L.shape

# LY here is a tensor $L_{y,r,c}$ where $y$ indexes the label set, $r$ the rows (datapoints), and $c$ the columns (labelling function, and target label at column 3)

LY

LY.shape

# # Intermezzo: Tensor products and contractions

testtensor1 = np.array([i for i in range(1,4) for _ in range(1,10)])
testtensor1 = np.reshape(testtensor1, (3,3,3))
testtensor1

testtensor2 = np.array([i for i in range(4,7) for _ in range(1,10)])
testtensor2 = np.reshape(testtensor2, (3,3,3))
testtensor2

np.dot(testtensor1[0,:,:],testtensor2[0,:,:])

np.einsum('ij,jk->ik', testtensor1[0,:,:], testtensor2[0,:,:])

np.einsum('ij,jk->i', testtensor1[0,:,:], testtensor2[0,:,:])

np.einsum('abc,dbe->acde', testtensor1, testtensor2)

# +
#For more on this see:
#https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/

outputtens = np.empty((3,3,3,3))
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                total = 0
                for m in range(0,3):
                    total += testtensor1[i,m,j]*testtensor2[k,m,l]
                outputtens[i,j,k,l] = total
# -

outputtens

np.array_equal(outputtens, np.einsum('abc,dbe->acde', testtensor1, testtensor2))

# # Class Imbalance Estimation

# Taking the tensor product $L \otimes L \otimes L$ (where $L$ here is `LY` in the code) we get the tensor  
# $(L \otimes L \otimes L)_{a,b,c,d,e,f,g,h,i}$.  
#   
# However contracting the indices for the rows for each of these tensors together we get the inner product of all the observations, for all combinations of labelling function and all output values $y$. This is  
#   
# $O'_{i,y',j,y'',k,y'''} = \sum_r (L \otimes L \otimes L)_{i,r,y',j,r,y'',k,r,y'''}$

# Trivially, we can reshape this tensor and divide it by the number of observations, $n$, to then interpret its entries as plug-in estimators for the joint probability of all the labelling functions and their possible output labels.  
#
# (Note that the joint probability $P(\lambda_i = y', \lambda_j=y'', \lambda_k=y''')$ is the probability that $\lambda_i=y'$ AND $\lambda_j=y''$ AND $\lambda_k=y'''$ ... which is exactly what we got in the tensor product and contraction over all rows, and then dividing by $n$.)  
#
# $O_{i,j,k,y',y'',y'''} = O'_{i,r,y',j,r,y'',k,r,y'''}/n = O' = \mathbb{E}[\psi(\lambda_i)_{y'}\psi(\lambda_j)_{y''}\psi(\lambda_k)_{y'''}] = P(\lambda_i = y', \lambda_j=y'', \lambda_k=y''')$

# We therefore have that  
# $O_{i,j,k,y',y'',y'''} = P(\lambda_i = y', \lambda_j=y'', \lambda_k=y''') = \sum_Y P(\lambda_i = y', \lambda_j=y'', \lambda_k=y''', Y) = \sum_y P(\lambda_i = y', \lambda_j=y'', \lambda_k=y''' | Y=y) P(Y=y)$
#
# Using the conditional independence assumption for the labelling functions, $P(\lambda_i = y', \lambda_j=y'', \lambda_k=y''' | Y) = P(\lambda_i = y' | Y) P(\lambda_j=y'' | Y) P( \lambda_k=y''' | Y)$  
#
# we then have  
# $O_{i,j,k,y',y'',y'''} = \sum_y P(\lambda_i = y' | Y=y) P(\lambda_j=y'' | Y=y) P( \lambda_k=y''' | Y=y) P(Y=y)$

# Therefore we expect the $O$ tensor we computed to factorise into the following tensor product, where each factor corresponds to one of the probability factors just above.  
# $O_{i,j,k,y',y'',y'''} = \sum_y C_{i,y',y} C_{j,y'',y} C_{k,y''',y} P(Y=y)$

# But the tensor decomposition gives us only the product of three tensors, not the factor of $P(Y=Y)$, so how do we get that?  
# Well we argue that the factor of $P(Y=y)$ was evenly absorbed into each of the tensor factors as a factor of $P(Y=y)^(1/3)$, as $C'_{i,y',y}=C_{i,y',y}P(Y=y)^\frac{1}{3}$  
#
# And thus  
# $O_{i,j,k,y',y'',y'''} = \sum_y C'_{i,y',y} C'_{j,y'',y} C'_{k,y''',y}$

        # Form the three-way overlaps matrix (m,m,m,k,k,k), m weak labellers and k output states
O = np.einsum("abc,dbe,fbg->cegadf", LY, LY, LY) / n

O.shape

# +
from itertools import product
import torch

m = O.shape[0] #number of labelling functions
k = O.shape[3] #number of output states

# Compute mask
mask = torch.ones((m, m, m, k, k, k)).bool()
for ii, jj, kk in product(range(m), repeat=3):
    if len(set((ii, jj, kk))) < 3:
        mask[ii, jj, kk, :, :, :] = 0

# +
from torch import nn, optim

Q = nn.Parameter(torch.rand(m, k, k)).float()
Otorch = torch.from_numpy(O).float()

#pois_loss = nn.PoissonNLLLoss()

def get_loss(O, Q, mask):
    # Main constraint: match empirical three-way overlaps matrix
    # (entries O_{ijk} for i != j != k)
    diffs = torch.norm((O - torch.einsum("aby,cdy,efy->acebdf", [Q, Q, Q]))[mask])**2
    #diffs = pois_loss(torch.einsum("aby,cdy,efy->acebdf", [Q, Q, Q])[mask], O[mask])
    # Error types below to be multiplied by factor of m*k*k (dims of Q)
    #  so they are larger and comparable to making a mistake of that size in each element of Q
    prefactor = torch.prod(torch.tensor(Q.shape))
    # Constrain conditional probabilities between 0 and 1
    diffs += prefactor*(torch.norm(torch.where(Q>1,1-Q,torch.zeros_like(Q)))**2 + torch.norm(torch.where(Q<0, Q, torch.zeros_like(Q)))**2) 
    # Constrain class probabilities, P(Y), to add to 1
    classprobs = torch.mean(torch.sum(Q, axis=1)**3, axis=0)
    diffs += prefactor*torch.norm(1-torch.sum(classprobs))**2
    # Constrain each class probability to be in [0,1]
    diffs += prefactor*torch.norm(torch.where(classprobs<0, classprobs, torch.zeros_like(classprobs)))**2
    diffs += prefactor*torch.norm(torch.where(classprobs>1, 1-classprobs, torch.zeros_like(classprobs)))**2
    return diffs


# +
optimizer = optim.LBFGS([Q], lr=0.01, max_iter=1e5)

# The closure computes the loss
def closure():
    optimizer.zero_grad()
    loss = get_loss(Otorch, Q, mask)
    loss.backward()
    #with torch.no_grad():
    #    for param in model.parameters():
    #            param.clamp_(0, 1)
    return loss

# Perform optimizer step
optimizer.step(closure)

# Recover the class balance
# Note that the columns are not necessarily ordered correctly at this
# point, since there's a column-wise symmetry remaining
q = Q.detach().numpy()
p_y = np.mean(q.sum(axis=1) ** 3, axis=0)

# Recover the estimated cond probs: Q = C(P^{1/3}) --> C = Q(P^{-1/3})
cps = q @ np.diag(1 / p_y ** (1 / 3))

q
# -

q.sum(axis=1)

q.sum(axis=1) ** 3

np.mean(q.sum(axis=1) ** 3, axis=0)

np.sum(np.mean(q.sum(axis=1) ** 3, axis=0))

p_y[1]

cps


# All of the above is exactly the code in the ClassBalanceModel() class of the Metal package. Below we check how well it estimates the class balance as the true class balance changes, modifying only the get_loss() function, to incorporate the constraints we added.

# +
# https://github.com/HazyResearch/metal/blob/cb_deps/metal/label_model/class_balance.py

class ClassBalanceModel_WBAA(nn.Module):
    """A model for learning the class balance, P(Y=y), given a subset of LFs
    which are *conditionally independent*, i.e. \lambda_i \perp \lambda_j | Y,
    for  i != j.
    Learns the model using a tensor factorization approach.
    Note: This approach can also be used for estimation of the LabelModel, may
    want to later refactor and expand this class.
    """

    def __init__(self, k, abstains=True, config=None):
        super().__init__()
        self.config = config
        self.k = k  # The cardinality of the true label, Y \in {1,...,k}

        # Labeling functions output labels in range {k_0,...,k}, and have
        # cardinality k_lf
        # If abstains=False, k_0 = 1 ==> k_lf = k
        # If abstains=True, k_0 = 0 ==> k_lf = k + 1
        self.abstains = abstains
        self.k_0 = 0 if self.abstains else 1
        self.k_lf = k + 1 if self.abstains else k

        # Estimated quantities (np.array)
        self.cond_probs = None
        self.class_balance = None

    def _get_overlaps_tensor(self, L):
        """Transforms the input label matrix to a three-way overlaps tensor.
        Args:
            L: (np.array) An n x m array of LF output labels, in {0,...,k} if
                self.abstains, else in {1,...,k}, generated by m conditionally
                independent LFs on n data points
        Outputs:
            O: (torch.Tensor) A (m, m, m, k, k, k) tensor of the label-specific
            empirical overlap rates; that is,
                O[i,j,k,y1,y2,y3] = P(\lf_i = y1, \lf_j = y2, \lf_k = y3)
            where this quantity is computed empirically by this function, based
            on the label matrix L.
        """
        n, m = L.shape

        # Convert from a (n,m) matrix of ints to a (k_lf, n, m) indicator tensor
        LY = np.array([np.where(L == y, 1, 0) for y in range(self.k_0, self.k + 1)])

        # Form the three-way overlaps matrix (m,m,m,k,k,k)
        O = np.einsum("abc,dbe,fbg->cegadf", LY, LY, LY) / n
        return torch.from_numpy(O).float()

    def get_mask(self, m):
        """Get the mask for the three-way overlaps matrix O, which is 0 when
        indices i,j,k are not unique"""
        mask = torch.ones((m, m, m, self.k_lf, self.k_lf, self.k_lf)).byte()
        for i, j, k in product(range(m), repeat=3):
            if len(set((i, j, k))) < 3:
                mask[i, j, k, :, :, :] = 0
        return mask

    #pois_loss = nn.PoissonNLLLoss()
    
    @staticmethod
    def get_loss(O, Q, mask):
        # Main constraint: match empirical three-way overlaps matrix
        # (entries O_{ijk} for i != j != k)
        #diffs = torch.norm((O - torch.einsum("aby,cdy,efy->acebdf", [Q, Q, Q]))[mask])**2
        #epsilon = 1e-5
        lambda1 = O[mask] #+ epsilon
        lambda2 = torch.einsum("aby,cdy,efy->acebdf", [Q, Q, Q])[mask] #+ epsilon
        # Additive smoothing
        epsilon = 1e-6
        lambda1 = torch.where(lambda1<epsilon, torch.zeros_like(lambda1)+epsilon, lambda1)
        lambda2 = torch.where(lambda2<epsilon, torch.zeros_like(lambda2)+epsilon, lambda2)
        #Generalised Kullback-Leibler loss: likelihood ratio of two Poissons
        # clone() required to avert this issue: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        diffs = torch.norm(lambda1*torch.log(lambda1/lambda2) - lambda1 + lambda2).clone()
        #diffs = pois_loss(torch.einsum("aby,cdy,efy->acebdf", [Q, Q, Q])[mask], O[mask])
        # Error types below to be multiplied by factor of m*k*k (dims of Q)
        #  so they are larger and comparable to making a mistake of that size in each element of Q
        prefactor = torch.prod(torch.tensor(Q.shape))
        # Constrain conditional probabilities between 0 and 1
        diffs += prefactor*(torch.norm(torch.where(Q>1,1-Q,torch.zeros_like(Q)))**2 \
                            + torch.norm(torch.where(Q<0, Q, torch.zeros_like(Q)))**2) 
        # Constrain class probabilities, P(Y), to add to 1
        classprobs = torch.mean(torch.sum(Q, axis=1)**3, axis=0)
        diffs += prefactor*torch.norm(1-torch.sum(classprobs))**2
        # Constrain each class probability to be in [0,1]
        diffs += prefactor*torch.norm(torch.where(classprobs<0, classprobs, torch.zeros_like(classprobs)))**2
        diffs += prefactor*torch.norm(torch.where(classprobs>1, 1-classprobs, torch.zeros_like(classprobs)))**2
        return diffs

    def train_model(self, L=None, O=None, lr=1, max_iter=1000, verbose=False):
        # Get overlaps tensor if L provided else use O directly (e.g. for tests)
        if O is not None:
            pass
        elif L is not None:
            O = self._get_overlaps_tensor(L)
        else:
            raise ValueError("L or O required as input.")
        self.m = O.shape[0]

        # Compute mask
        self.mask = self.get_mask(self.m)

        # Initialize parameters
        self.Q = nn.Parameter(torch.rand(self.m, self.k_lf, self.k)).float()

        # Use L-BFGS here
        # Seems to be a tricky problem for simple 1st order approaches, and
        # small enough for quasi-Newton... L-BFGS seems to work well here
        optimizer = optim.LBFGS([self.Q], lr=lr, max_iter=max_iter)
        #optimizer = optim.RMSprop([self.Q], lr=lr)

        # The closure computes the loss
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(O, self.Q, self.mask)
            loss.backward()
            if verbose:
                print(f"Loss: {loss.detach():.8f}")
            return loss

        # Perform optimizer step
        optimizer.step(closure)

        # Recover the class balance
        # Note that the columns are not necessarily ordered correctly at this
        # point, since there's a column-wise symmetry remaining
        q = self.Q.detach().numpy()
        p_y = np.mean(q.sum(axis=1) ** 3, axis=0)

        # Resolve remaining col-wise symmetry
        # We do this by first estimating the conditional probabilities (accs.)
        # P(\lambda_i = y' | Y = y) of the labeling functions, *then leveraging
        # the assumption that they are better than random* to resolve col-wise
        # symmetries here
        # Note we then store both the estimated conditional probs, and the class
        # balance

        # Recover the estimated cond probs: Q = C(P^{1/3}) --> C = Q(P^{-1/3})
        cps = q @ np.diag(1 / p_y ** (1 / 3))

        # Note: For assessing the order, we only care about the non-abstains
        if self.k_lf > self.k:
            cps_na = cps[:, 1:, :]
        else:
            cps_na = cps

        # Re-order cps and p_y using assumption and store np.array values
        # Note: We take the *most common* ordering
        vals, counts = np.unique(cps_na.argmax(axis=2), axis=0, return_counts=True)
        col_order = vals[counts.argmax()]
        self.class_balance = p_y[col_order]
        self.cond_probs = cps[:, :, col_order]


# -

# Uncomment the two lines below and comment the "cbmodel =" to see what the impact of the constraints I added to the optimisation (the get_loss() function) is on the plot below

# +
from tqdm import tqdm

#from metal.label_model.class_balance import ClassBalanceModel

intercepts = np.linspace(-15, 10, num=35)
classbalestmdf = pd.DataFrame()

for idx,i in enumerate(tqdm(intercepts)):
    classbalestmdf.loc[idx, 'intercept'] = i
    
    tempdf = generateSynthetic(coefs=[i, -2, 3], wlrates=wlrates, n=10000, seed=5)
    classbalestmdf.loc[idx, 'classbal'] = tempdf['y'].mean()
    
    tempL = tempdf[["wl1", "wl2", "wl3", "y"]].values
    tempL = tempL + 1
    
    #cbmodel = ClassBalanceModel(2, abstains=False)
    cbmodel = ClassBalanceModel_WBAA(2, abstains=False)
    cbmodel.train_model(L=tempL, lr=0.1, max_iter=1e3)
    classbalestmdf.loc[idx, 'estimated'] = cbmodel.class_balance[1]
# -

classbalestmdf

# # The above behaviour looks very much like the all-or-nothing phenomenon to me!! Not at all sure it's the same thing though.
# See this paper:
# https://arxiv.org/abs/2007.11138

plt.plot(classbalestmdf['classbal'].values, classbalestmdf['estimated'].values)
plt.plot([0, 1], [0, 1], color='r', linestyle='dashed', linewidth=1)
#plt.ylim(0,1)
plt.xlabel("True class balance")
plt.ylabel("Estimated class balance")
plt.show()

# - Change to Poisson loss?  
# - Any more constraints we could add?
# - Perhaps extract class balance separately and multiply each conditional probability matrix by (p_y_true/p_y) to correct conditional probabilities for mis-estimated class balance at high imbalances?  
# - Why not just compute the class balance directly from the label since we need that for this procedure to work anyways? However note that incorrect class balances will badly skew the conditional probabilities too since to get those we need to factor out the class balance factor from the tensor product!
# - Or input class balance observed as parameter into the optimisation, and introduce a term like (p_y-p_y_true)^2 into the optimisation?

0.02*n

# Class balance estimation does not seem to work for values less than ~0.02, does this mean we need at least ~200 positive labels in the dataset, or is the requirement fractional (i.e. require at least 2% of events)?


