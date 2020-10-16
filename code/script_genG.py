# Example script to run the generalized Gaussian experiment

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
d = 2 
N = 2000
bi = 0
p = 2


x0N = np.random.normal(size=(d,)) 
x0L = np.random.normal(size=(d,)) 
x0M = np.random.normal(size=(d,)) 
x0T = np.random.normal(size=(d,)) 
x0N = 1000 * x0N / np.linalg.norm(x0N) 
x0M = 1000 * x0M / np.linalg.norm(x0M) 
x0L = 1000 * x0L / np.linalg.norm(x0L) 
x0T = 1000 * x0T / np.linalg.norm(x0T) 


reps = 50 
sm = 10 

mLs = np.zeros((d, reps))
mNs = np.zeros((d, reps))
mTs = np.zeros((d, reps))
mMs = np.zeros((d, reps))

CLs = np.zeros((d, d, reps))
CNs = np.zeros((d, d, reps))
CTs = np.zeros((d, d, reps))
CMs = np.zeros((d, d, reps))

YLs = np.zeros((d, sm*N, reps))
YNs = np.zeros((d, N, reps))
YTs = np.zeros((d, sm*N, reps))
YMs = np.zeros((d, N, reps))

tLs = np.zeros((sm*N, reps))
tNs = np.zeros((N, reps))
tTs = np.zeros((sm*N, reps))


#Sigma = np.diag(np.linspace(1, 101, d)) 
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


mod1 = mirrorLangevinMC(V, grad_V, grad_V_star, H_V, inv_grad = True)

mod1.phi = lambda x: phi(x)
mod1.grad_phi = lambda x: grad_phi(x)
mod1.grad_phi_star = lambda x: grad_phi_star(x)
mod1.H_phi = lambda x: H_phi(x)

for r in tqdm(range(reps)):
    mLs[:, r], CLs[:, :, r], YLs[:, :, r], tLs[:,r] = mod1.ULA(x0L, step, sm*N, burn_in = bi, quiet = True)
    mNs[:, r], CNs[:, :, r], YNs[:, :, r], tNs[:,r] = mod1.NLA(x0N, step, N, burn_in = bi, quiet = True)
    mTs[:, r], CTs[:, :, r], YTs[:, :, r], tTs[:,r] = mod1.TULA(x0T, step, sm*N, gamma = 0.1, burn_in = bi, quiet = True)
    mMs[:, r], CMs[:, :, r], YMs[:, :, r] = mod1.MLA(x0M, step, N, burn_in = bi, quiet = True)



#########
# save outputs
np.save(str(step) + '/mLs.npy', mLs)
np.save(str(step) + '/mNs.npy', mNs)
np.save(str(step) + '/mMs.npy', mMs)
np.save(str(step) + '/mTs.npy', mTs)
np.save(str(step) + '/CLs.npy', CLs)
np.save(str(step) + '/CNs.npy', CNs)
np.save(str(step) + '/CTs.npy', CTs)
np.save(str(step) + '/CMs.npy', CMs)
np.save(str(step) + '/YLs.npy', YLs)
np.save(str(step) + '/YNs.npy', YNs)
np.save(str(step) + '/YTs.npy', YTs)
np.save(str(step) + '/YMs.npy', YMs)
np.save(str(step) + '/tLs.npy', tLs)
np.save(str(step) + '/tNs.npy', tNs)
np.save(str(step) + '/tTs.npy', tTs)

np.save(str(step) + '/x0N.npy', x0N)
np.save(str(step) + '/x0M.npy', x0M)
np.save(str(step) + '/x0L.npy', x0L)
np.save(str(step) + '/x0T.npy', x0T)



#np.save(str(step) + '/Sigma.npy', Sigma)


