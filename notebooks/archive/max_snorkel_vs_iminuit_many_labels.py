# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# Make synthetic dataset.
#I, J, K, R = 2, 2, 2, 2  # dimensions and rank
#X = tt.rand_ktensor((I, J, K), rank=R).full()
#X += np.random.rand(I, J, K)  # add noise

#I, J, K, R = 5, 4, 3, 2  # dimensions and rank
#X = tt.rand_ktensor((I, J, K), rank=R).full()
# -

# Set our RNG for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

import snorkel

y = np.random.binomial(n=1, p=0.001, size=100000)


def noise(x, fp=0.01, fn=0.1, abstain=0.1):
    v = np.random.uniform()
    if v <= abstain:
        return 2
    u = np.random.uniform()
    if x==0:
        if u <= (fp/(1.0-abstain)):
            return 1
    else:
        if u <= (fn/(1.0-abstain)):
            return 0
    return x


# +
# create 3 weak label vectors with a lot of noise
w1 = np.asarray([noise(l, 0.05, 0.4) for l in y])

w2 = np.asarray([noise(l, 0.05, 0.3) for l in y])

w3 = np.asarray([noise(l, 0.10, 0.1) for l in y])

w4 = np.random.binomial(n=1, p=0.05, size=100000)

w5 = np.asarray([noise(l, 0.15, 0.4) for l in y])

# -

wls = [w1, w2, w3, w4, w5]

# +
from scipy import special
import math

n_sources = len(wls)
N = 100000
R = 2
dim = [len(np.unique(w)) for w in wls]
dim_minus_one = [d-1 for d in dim]

w = [0.333] * (sum(dim_minus_one) * R + R-1)
w[0] = 0.999

def logpoisson(w):
    f = nexp(w)
    f = f.ravel()
    y = X.ravel()
    # Poisson binned likelihood sum
    ll = f - special.xlogy(y, f) + special.gammaln(y + 1)
    return ll

def nll(w):
    ll = logpoisson(w)
    return sum(ll)

def calc_factors(w):
    Y = list(w[:R-1])
    pY = Y
    pY.append(1.0 - sum(pY))
    pYpow = [math.pow(p, 1./n_sources) for p in pY]
    cprob = w[R-1:]
    factors = [np.zeros((R, dim[j])) for j in range(n_sources)]
    
    for i in range(n_sources):
        for j in range(R):
            for k in range(dim_minus_one[i]):
                factors[i][j][k] = cprob[sum(dim_minus_one[:i])*R + j*dim_minus_one[i] + k]
            pend = 1 - sum(factors[i][j])
            if pend <= 0: 
                pend = 1e-9
            factors[i][j][dim_minus_one[i]] = pend
            factors[i][j] *= pYpow[j]
    factors = [f.T for f in factors]
    return factors

def probmat(w):
    factors = calc_factors(w)
    E = reconstruct(factors)
    return E

def nexp(w):
    E = probmat(w)
    E *= N
    return E

def cond_probs(w):
    cprob = w[R-1:]
    factors = [np.zeros((R, dim[j])) for j in range(n_sources)]    
    for i in range(n_sources):
        for j in range(R):
            for k in range(dim_minus_one[i]):
                factors[i][j][k] = cprob[sum(dim_minus_one[:i])*R + j*dim_minus_one[i] + k]
            pend = 1 - sum(factors[i][j])
            if pend <= 0: 
                pend = 1e-9
            factors[i][j][dim_minus_one[i]] = pend
    factors = [f.T for f in factors]
    return factors

def reconstruct(factors, rank=None):
    assert len(factors) >= 2
    rank = rank if rank is not None else factors[0].shape[1]
    shapes = [f.shape[0] for f in factors]
    shapes.append(rank)
    shapes = tuple(shapes)
    R1s = np.zeros(shapes)

    for i in range(rank):
        stubi = [f[:, i] for f in factors]
        colsi = [s.reshape(-1, 1).astype(np.float32) for s in stubi]
        a, b = colsi[0], colsi[1]
        Ti = a * b.T
        for j in range(2, len(colsi)):
            cij = colsi[j]
            Tij = np.tensordot(Ti, cij, axes=0)
            cmd1 = 'Ti = Tij[' + ''.join([':, '] * (j+1)) + ' 0]'
            d = locals()
            exec(cmd1, d)
            Ti = d['Ti']
            #T = Tj[:, :, :, 0]

        #R1s[:, :, :, i] = rank_one_tensor(a[:, i], b[:, i], c[:, i])
        cmd2 = 'R1s[' + ''.join([':, '] * len(factors)) + ' i] = Ti'
        d = locals()
        exec(cmd2) #, {"R1s": R1s, "Ti": Ti, "i": i})
        R1s = d['R1s']

    naxis = len(factors)
    E = R1s.sum(axis=naxis)
    return E

def r1s(factors, rank=None):
    assert len(factors) >= 2
    rank = rank if rank is not None else factors[0].shape[1]
    shapes = [f.shape[0] for f in factors]
    shapes.append(rank)
    shapes = tuple(shapes)
    R1s = np.zeros(shapes)

    for i in range(rank):
        stubi = [f[:, i] for f in factors]
        colsi = [s.reshape(-1, 1).astype(np.float32) for s in stubi]
        a, b = colsi[0], colsi[1]
        Ti = a * b.T
        for j in range(2, len(colsi)):
            cij = colsi[j]
            Tij = np.tensordot(Ti, cij, axes=0)
            cmd1 = 'Ti = Tij[' + ''.join([':, '] * (j+1)) + ' 0]'
            d = locals()
            exec(cmd1, d)
            Ti = d['Ti']
            #T = Tj[:, :, :, 0]

        #R1s[:, :, :, i] = rank_one_tensor(a[:, i], b[:, i], c[:, i])
        cmd2 = 'R1s[' + ''.join([':, '] * len(factors)) + ' i] = Ti'
        d = locals()
        exec(cmd2) #, {"R1s": R1s, "Ti": Ti, "i": i})
        R1s = d['R1s']
    return R1s

def cfunc(w):
    lamd = 1e7
    chi2 = 0
    # sum_Y p(Y) = 1
    x_sub = w[:R-1]
    delta = 1 - sum(x_sub)
    if delta < 0:
        chi2 += lamd * delta**2
    # sum_lambda p(lambda|Y) = 1
    x = w[R-1:]
    for i in range(n_sources):
        for j in range(R):
            j_start = sum(dim_minus_one[:i]) * R + j * dim_minus_one[i]
            j_stop  = sum(dim_minus_one[:i]) * R + (j+1) * dim_minus_one[i]
            x_sub = x[j_start: j_stop]
            delta = 1 - sum(x_sub)
            if delta < 0:
                chi2 += lamd * delta**2
    return chi2

def func(w):
    return nll(w) + cfunc(w)


# +
# create the count-tensor
# note: prob(1,2,3) = X / 100k

tdim = tuple(dim)
X = np.zeros(tdim)
for ws in zip(*wls):
    X[ws] += 1
# -

len(w)

# fit starting values
w = [0.999, 0.9, 0.05, 0.05, 0.7, 0.65, 0.05, 0.05, 0.5, 0.6, 0.05, 0.05, 0.9, 0.9, 0.1, 0.9, 0.05, 0.05, 0.7]

nll(w)

cfunc(w)

from iminuit import Minuit

p_names = ['p{index}'.format(index=i) for i in range(len(w))]
kws = {}
for i in range(len(w)):
    kws['limit_p{index}'.format(index=i)] = (0,1)
    kws['error_p{index}'.format(index=i)] = 0.01

for co in range(100):
    for i in range(len(w)):
        kws['p{index}'.format(index=i)] = w[i]

    m = Minuit(func, use_array_call=True, forced_parameters=p_names, errordef = 0.5, **kws) 

    res1, res2 = m.migrad()

    ps = m.get_param_states()
    p = [ps[i]['value'] for i in range(len(ps))]
    w = p

    ok = res1.has_accurate_covar and res1.has_covariance and not res1.has_made_posdef_covar and res1.has_posdef_covar and not res1.has_reached_call_limit and res1.has_valid_parameters and not res1.hesse_failed
    if ok:
        break


res1

m.matrix(correlation=True)

ps = m.get_param_states()
m.get_param_states()

len(ps)

nll(p)

cfunc(p)

factors = cond_probs(p)

a, b, c, d, e = factors

# eigenvalues
p[0], 1-p[0]

a

b

c

d

e

p

E = nexp(p)

Xr = X.ravel()


def mysqrt(x):
    if x>0:
        return math.sqrt(x)
    return 1


errX = np.asarray([mysqrt(x) for x in Xr])

normres = (X.ravel() - E.ravel()) / errX

# %matplotlib inline

import pandas as pd

df = pd.DataFrame({'norm': normres})

df['norm'].hist(bins=40)

df['norm'].std()

df['norm'].mean()

m.draw_profile('p0')

a, fa = m.profile('p0')







import pandas as pd

df = pd.DataFrame({'wl1':w1, 'wl2':w2, 'wl3':w3, 'wl4':w4, 'wl5':w5})

mask = df==2

df[mask] = -1

df.corr()

from snorkel.labeling import LabelModel, MajorityLabelVoter, RandomVoter

label_model = LabelModel(cardinality=2, verbose=True)

label_model.fit(L_train=df.values, n_epochs=1000, lr=0.001, log_freq=100, seed=123)


preds_queried = label_model.predict(L=df.values, return_probs=True)


preds_queried[1]

sum(preds_queried[1][:,1])

sum(preds_queried[0])

preds_queried[0]

p

factors = calc_factors(p)
R1s = r1s(factors)


R1s.shape

sump0 = R1s[0,0,0,0,0,0]

sump1 = R1s[0,0,0,0,0,1]

E = R1s.sum(axis=len(factors))

sump = E[0,0,0,0,0]

sump1 / sump

sump0 = R1s[1,1,1,1,1,0]

sump1 = R1s[1,1,1,1,1,1]

sump = E[1,1,1,1,1]

sump1/sump

wls

# +
p1s = []

for ws in zip(*wls):
    #print (ws)
    p = R1s
    for idx in ws:
        p = p[idx]
    #print (R1s[ws, 0])
    p1s.append(p[1]/(p[0]+p[1]))
    #break
# -

sum(p1s)

100000. * (p[0])

label_model.get_conditional_probs()

factors = cond_probs(p)

a

b

c

d

e


