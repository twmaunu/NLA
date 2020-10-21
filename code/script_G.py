# Example script to run the ill-conditioned Gaussian experiment

import sys, getopt
sys.path.append('../code/')
from mirrorLangevinMC import *
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import sqrtm
from scipy.linalg import null_space
from scipy.special import gamma, logit
from tqdm import tqdm

import os
path = os.getcwd() + "/Gaussian_" + str(step) 
try:
    os.mkdir(path)
except:
    print("Path already created, overwriting contents")

# Choose number of repetitions for experiment and step size
step = 0.1
reps = 1 

# Parameters
d = 100 
# number of samples 
N = 2000
# burn-in
bi = 1000
# power for generalized Gaussian (2 is Gaussian)
p = 2
# initialization
x0 = np.ones(shape=(d,))

###################################################
# define the necessary functions for the Langevin Algorithms 
Sigma = np.diag(np.linspace(1, 101, d)) 
D = inv(sqrtm(Sigma))
V = scale_V(V_p(p), D)
grad_V = scale_grad_V(grad_V_p(p), D)
grad_V_star = scale_grad_V_star(grad_V_p_star(p), D)
H_V = scale_Hess_V(Hess_V_p(p, d), D)
def u(x):
    return ((p * (np.power(x, (p-1)))) / 2)

###################################################
# store means, covariances, samples, and times for later plotting
mLs = np.zeros((d, reps))
mNs = np.zeros((d, reps))
mTs = np.zeros((d, reps))

CLs = np.zeros((d, d, reps))
CNs = np.zeros((d, d, reps))
CTs = np.zeros((d, d, reps))

YLs = np.zeros((d, sm*N, reps))
YNs = np.zeros((d, N, reps))
YTs = np.zeros((d, sm*N, reps))

tLs = np.zeros((sm*N, reps))
tNs = np.zeros((N, reps))
tTs = np.zeros((sm*N, reps))



ML = mirrorLangevinMC(V, grad_V, grad_V_star, H_V)

for r in tqdm(range(reps)):
    mLs[:, r], CLs[:, :, r], YLs[:, :, r], tLs[:,r] = ML.ULA(x0, step, sm*N, burn_in = bi, quiet = True)
    mNs[:, r], CNs[:, :, r], YNs[:, :, r], tNs[:,r] = ML.NLA(x0, step, N, burn_in = bi, quiet = True)
    mTs[:, r], CTs[:, :, r], YTs[:, :, r], tTs[:,r] = ML.TULA(x0, step, sm*N, gamma = 0.1, burn_in = bi, quiet = True)


#########
# save outputs
np.save("Gaussian_" + str(step) + '/mLs.npy', mLs)
np.save("Gaussian_" + str(step) + '/mNs.npy', mNs)
np.save("Gaussian_" + str(step) + '/mTs.npy', mTs)

np.save("Gaussian_" + str(step) + '/CLs.npy', CLs)
np.save("Gaussian_" + str(step) + '/CNs.npy', CNs)
np.save("Gaussian_" + str(step) + '/CTs.npy', CTs)

np.save("Gaussian_" + str(step) + '/YLs.npy', YLs)
np.save("Gaussian_" + str(step) + '/YNs.npy', YNs)
np.save("Gaussian_" + str(step) + '/YTs.npy', YTs)

np.save("Gaussian_" + str(step) + '/tLs.npy', tLs)
np.save("Gaussian_" + str(step) + '/tNs.npy', tNs)
np.save("Gaussian_" + str(step) + '/tTs.npy', tTs)

np.save("Gaussian_" + str(step) + '/Sigma.npy', Sigma)

