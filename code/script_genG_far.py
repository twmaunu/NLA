# Example script to run the generalized Gaussian experiment

import sys, getopt
sys.path.append('../code/')
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

# parse and create directory
import os
path = os.getcwd() + "/genGaussian_" + str(step) 
try:
    os.mkdir(path)
except:
    print("Path already created, overwriting contents")

# Parameters
# dimension
d = 2 
# number of samples
N = 2000
# burn-in
bi = 0

# note: due to faster iterations, create 10x as many samples for ULA and TULA
sm = 10

###################################################
# generalized Gaussian power
p = 3/2

D = np.eye(d)  #inv(sqrtm(Sigma))
phi = scale_V(V_p(p), D)
grad_phi = scale_grad_V(grad_V_p(p), D)
grad_phi_star = scale_grad_V_star(grad_V_p_star(p), D)
H_phi = scale_Hess_V(Hess_V_p(p, d), D)
def u(x):
   return ((p * (np.power(x, (p-1)))) / 2)

c = 0.001
offset = np.ones(d, )
def V(x):
    return np.linalg.norm(x) + (c / 2) * np.power(np.linalg.norm(x - offset), 2) 
def grad_V(x):
    return x / np.linalg.norm(x) + c * (x - offset) 
def grad_V_star(x):
    return (x - x / np.linalg.norm(x)) / c
def H_V(x):
    return np.eye(np.size(x)) / np.linalg.norm(x) - np.outer(x, x) / np.power(np.linalg.norm(x), 3) + c * np.eye(np.size(x)) 

###################################################
# different initialization for each algorithm
x0N = np.random.normal(size=(d,)) 
x0L = np.random.normal(size=(d,)) 
x0M = np.random.normal(size=(d,)) 
x0T = np.random.normal(size=(d,)) 
x0N = 1000 * x0N / np.linalg.norm(x0N) 
x0M = 1000 * x0M / np.linalg.norm(x0M) 
x0L = 1000 * x0L / np.linalg.norm(x0L) 
x0T = 1000 * x0T / np.linalg.norm(x0T) 


# store means
mLs = np.zeros((d, reps))
mNs = np.zeros((d, reps))
mTs = np.zeros((d, reps))
mMs = np.zeros((d, reps))

# store covariances 
CLs = np.zeros((d, d, reps))
CNs = np.zeros((d, d, reps))
CTs = np.zeros((d, d, reps))
CMs = np.zeros((d, d, reps))

# store samples 
YLs = np.zeros((d, sm*N, reps))
YNs = np.zeros((d, N, reps))
YTs = np.zeros((d, sm*N, reps))
YMs = np.zeros((d, N, reps))

# store times
tLs = np.zeros((sm*N, reps))
tNs = np.zeros((N, reps))
tTs = np.zeros((sm*N, reps))
tMs = np.zeros((N, reps))

ML = mirrorLangevinMC(V, grad_V, grad_V_star, H_V)
ML.phi = lambda x: phi(x)
ML.grad_phi = lambda x: grad_phi(x)
ML.grad_phi_star = lambda x: grad_phi_star(x)
ML.H_phi = lambda x: H_phi(x)

for r in tqdm(range(reps)):
    mLs[:, r], CLs[:, :, r], YLs[:, :, r], tLs[:,r] = ML.ULA(x0L, step, sm*N, burn_in = bi, quiet = True)
    mNs[:, r], CNs[:, :, r], YNs[:, :, r], tNs[:,r] = ML.NLA(x0N, step, N, burn_in = bi, quiet = True)
    mTs[:, r], CTs[:, :, r], YTs[:, :, r], tTs[:,r] = ML.TULA(x0T, step, sm*N, gamma = 0.1, burn_in = bi, quiet = True)
    mMs[:, r], CMs[:, :, r], YMs[:, :, r], tMs[:,r] = ML.MLA(x0M, step, N, burn_in = bi, quiet = True)



#########
# save outputs
np.save("genGaussian_" + str(step) + '/mLs.npy', mLs)
np.save("genGaussian_" + str(step) + '/mNs.npy', mNs)
np.save("genGaussian_" + str(step) + '/mMs.npy', mMs)
np.save("genGaussian_" + str(step) + '/mTs.npy', mTs)

np.save("genGaussian_" + str(step) + '/CLs.npy', CLs)
np.save("genGaussian_" + str(step) + '/CNs.npy', CNs)
np.save("genGaussian_" + str(step) + '/CTs.npy', CTs)
np.save("genGaussian_" + str(step) + '/CMs.npy', CMs)

np.save("genGaussian_" + str(step) + '/YLs.npy', YLs)
np.save("genGaussian_" + str(step) + '/YNs.npy', YNs)
np.save("genGaussian_" + str(step) + '/YTs.npy', YTs)
np.save("genGaussian_" + str(step) + '/YMs.npy', YMs)

np.save("genGaussian_" + str(step) + '/tLs.npy', tLs)
np.save("genGaussian_" + str(step) + '/tNs.npy', tNs)
np.save("genGaussian_" + str(step) + '/tTs.npy', tTs)
np.save("genGaussian_" + str(step) + '/tMs.npy', tMs)

np.save("genGaussian_" + str(step) + '/x0N.npy', x0N)
np.save("genGaussian_" + str(step) + '/x0M.npy', x0M)
np.save("genGaussian_" + str(step) + '/x0L.npy', x0L)
np.save("genGaussian_" + str(step) + '/x0T.npy', x0T)



#np.save(str(step) + '/Sigma.npy', Sigma)


