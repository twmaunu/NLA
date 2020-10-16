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

# parse and create directory
step = float(sys.argv[1])
import os
path = os.getcwd() + "/" + str(step) 
try:
    os.mkdir(path)
except:
    print("Path already created, overwriting contents")

# Parameters
d = 100 
N = 2000
bi = 1000
p = 2
x0 = np.ones(shape=(d,))
reps = 20 
sm = 10 

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


Sigma = np.diag(np.linspace(1, 101, d)) 

D = inv(sqrtm(Sigma))
V = scale_V(V_p(p), D)
grad_V = scale_grad_V(grad_V_p(p), D)
grad_V_star = scale_grad_V_star(grad_V_p_star(p), D)
H_V = scale_Hess_V(Hess_V_p(p, d), D)
def u(x):
    return ((p * (np.power(x, (p-1)))) / 2)


mod1 = mirrorLangevinMC(V, grad_V, grad_V_star, H_V)

for r in tqdm(range(reps)):
    mLs[:, r], CLs[:, :, r], YLs[:, :, r], tLs[:,r] = mod1.ULA(x0, step, sm*N, burn_in = bi, quiet = True)
    mNs[:, r], CNs[:, :, r], YNs[:, :, r], tNs[:,r] = mod1.NLA(x0, step, N, burn_in = bi, quiet = True)
    mTs[:, r], CTs[:, :, r], YTs[:, :, r], tTs[:,r] = mod1.TULA(x0, step, sm*N, gamma = 0.1, burn_in = bi, quiet = True)


#########
# save outputs
np.save(str(step) + '/mLs.npy', mLs)
np.save(str(step) + '/mNs.npy', mNs)
np.save(str(step) + '/mTs.npy', mTs)
np.save(str(step) + '/CLs.npy', CLs)
np.save(str(step) + '/CNs.npy', CNs)
np.save(str(step) + '/CTs.npy', CTs)
np.save(str(step) + '/YLs.npy', YLs)
np.save(str(step) + '/YNs.npy', YNs)
np.save(str(step) + '/YTs.npy', YTs)
np.save(str(step) + '/tLs.npy', tLs)
np.save(str(step) + '/tNs.npy', tNs)
np.save(str(step) + '/tTs.npy', tTs)
np.save(str(step) + '/mUNs.npy', mUNs)
np.save(str(step) + '/CUNs.npy', CUNs)
np.save(str(step) + '/YUNs.npy', YUNs)
np.save(str(step) + '/tUNs.npy', tUNs)
np.save(str(step) + '/Sigma.npy', Sigma)


