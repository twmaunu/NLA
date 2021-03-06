# Example script to run the logistic regression experiment

import sys, getopt
from mirrorLangevinMC import *
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import sqrtm
from scipy.linalg import null_space
from scipy.special import gamma, logit
from tqdm import tqdm

# Choose number of repetitions for experiment and step size
step = 0.1
reps = 1 

# create directory
import os
path = os.getcwd() + "/LR_" + str(step) 
try:
    os.mkdir(path)
except:
    print("Path already created, overwriting contents")


# Example: logistic regression

# Parameters
# dimension of parameter
d = 2 
# number of samples 
N = 500
# burn-in
bi = 500
#initial parameter
beta0 = np.ones(shape=(d,))

# note: due to faster iterations, create 10x as many samples for ULA and TULA
sm = 10


###################################################
# generate data
# number of pairs (xi,yi)
n = 100
Sigma = 100 * np.eye(d)
beta = np.ones((d,))
SigmaX = np.diag([10, .1])
X = np.random.multivariate_normal(np.zeros(d, ), SigmaX, n).T
Y = np.random.binomial(1, p = ilogit(beta.T.dot(X)))

###################################################
# define the necessary functions for the Langevin Algorithms 
def V(b):
  return (0.5 * b.T.dot(np.linalg.pinv(Sigma)).dot(b) - 
          np.sum(Y * b.T.dot(X)) + 
          np.sum(np.log(1 + np.exp(b.T.dot(X)))))
def grad_V(b):
  return (np.linalg.pinv(Sigma).dot(b) - 
          np.sum(np.outer(np.ones(d, ), Y) * X, axis = 1) + 
          np.sum(np.outer(np.ones(d, ), 
                          np.power(1 + np.exp(-b.T.dot(X)), -1)) * X, axis = 1))
def H_V(b):
  Z = X * (np.outer(np.ones(d, ), (np.exp(-b.T.dot(X)) * 
                                   np.power(1 + np.exp(-b.T.dot(X)), -2))))
  return (np.linalg.pinv(Sigma) + Z.dot(X.T))


###################################################
# store means, covariances, samples, and times for later plotting
mLs = np.zeros((d, reps))
mNs = np.zeros((d, reps))
mTs = np.zeros((d, reps))

CLs = np.zeros((d, d, reps))
CNs = np.zeros((d, d, reps))
CTs = np.zeros((d, d, reps))

YLs = np.zeros((d, sm * N, reps))
YNs = np.zeros((d, N, reps))
YTs = np.zeros((d, sm * N, reps))

tLs = np.zeros((sm * N, reps))
tNs = np.zeros((N, reps))
tTs = np.zeros((sm * N, reps))

# initialize optimization object
ML = mirrorLangevinMC(V, grad_V, grad_V, H_V, inv_grad_V = True)

for r in tqdm(range(reps)):
    mLs[:, r], CLs[:, :, r], YLs[:, :, r], tLs[:, r] = ML.ULA(beta0, 0.1, sm * N, burn_in = bi, quiet = True)
    mNs[:, r], CNs[:, :, r], YNs[:, :, r], tNs[:, r] = ML.NLA(beta0, 0.001, N, burn_in = bi, quiet = True)
    mTs[:, r], CTs[:, :, r], YTs[:, :, r], tTs[:, r] = ML.TULA(beta0, 0.001, sm * N, gamma = 0.01, burn_in = bi, quiet = True)


####################################
# save outputs
np.save('LR_' + str(step) + '/mLs.npy', mLs)
np.save('LR_' + str(step) + '/mNs.npy', mNs)
np.save('LR_' + str(step) + '/mTs.npy', mTs)

np.save('LR_' + str(step) + '/CLs.npy', CLs)
np.save('LR_' + str(step) + '/CNs.npy', CNs)
np.save('LR_' + str(step) + '/CTs.npy', CTs)

np.save('LR_' + str(step) + '/YLs.npy', YLs)
np.save('LR_' + str(step) + '/YNs.npy', YNs)
np.save('LR_' + str(step) + '/YTs.npy', YTs)

np.save('LR_' + str(step) + '/beta.npy', beta)

np.save('LR_' + str(step) + '/tLs.npy', tLs)
np.save('LR_' + str(step) + '/tNs.npy', tNs)
np.save('LR_' + str(step) + '/tTs.npy', tTs)

np.save('LR_' + str(step) + '/X.npy', X)
np.save('LR_' + str(step) + '/Y.npy', Y)

np.save('LR_' + str(step) + '/Sigma.npy', Sigma)

