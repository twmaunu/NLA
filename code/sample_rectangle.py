

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import sqrtm
from scipy.linalg import null_space




# why do we need these imports?
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import ot

# Generate the rectangular plot in Figure 3

# Parameters
nsamp = 300
d = 2
maxiter = 200
h0 = 0.00001
c = 1/10000
a=0.01

# Target two dimensional unit cube
def V1(x):
    return c*(- np.log(1+x) - np.log(1-x))

def grad_V1(x):
  return c*(-1/(x+1) + 1/(1-x))

def grad_V_star1(x):
  y = x
  y[x>0] = - c/x[x>0] + np.sqrt(np.power(c/x[x>0], 2) + 1)
  y[x==0] = 0
  y[x<0] = - c/x[x<0] - np.sqrt(np.power(c/x[x<0], 2) + 1)
  return y

def V2(x):
    return c*(- np.log(a+x) - np.log(a-x))

def grad_V2(x):
  return c*(-1/(x+a) + 1/(a-x))

def grad_V_star2(x):
  y = x
  y[x>0] = - c/x[x>0] + np.sqrt(np.power(c/x[x>0], 2) + a**2)
  y[x==0] = 0
  y[x<0] = - c/x[x<0] - np.sqrt(np.power(c/x[x<0], 2) + a**2)
  return y

def H_V1(x):
  # print(np.sqrt(c*(1/(np.power(1+x, 2)) + 1/(np.power(1-x, 2)))))
  return np.sqrt(c*(1/(np.power(1+x, 2)) + 1/(np.power(1-x, 2))))

def H_V2(x):
  # print(np.sqrt(c*(1/(np.power(1+x, 2)) + 1/(np.power(1-x, 2)))))
  return np.sqrt(c*(1/(np.power(a+x, 2)) + 1/(np.power(a-x, 2))))

# initialize with zeros
# init = np.zeros(shape=(nsamp,2))
# print(init.shape)

# # do the problem in 1d at the same time, see if we get different results
X_nla1 = np.zeros(shape=(1,nsamp))
X_nla2 = np.zeros(shape=(1,nsamp))

for i in range(maxiter):
  step = h0
  # update the 1d examples first, see if they work
  # lets couple these guys
  # print("printing hess 1 from 1d")
  X_nla1 = grad_V_star1(grad_V1(X_nla1) - 
                      step * grad_V1(X_nla1) + 
                      math.sqrt(2 * step) * 
                      H_V1(X_nla1)*(np.random.normal(size = (1,nsamp))))
  # print("printing hess 2 from 1d")
  X_nla2 = grad_V_star2(grad_V2(X_nla2) - 
                      step * grad_V2(X_nla2) + 
                      math.sqrt(2 * step) * 
                      H_V2(X_nla2)*(np.random.normal(size = (1,nsamp))))

# sample with random walk 
nsamp = 300
d = 2
maxiter = 200
h0 = 0.00001
h1 = 0.01
sns.set_palette("muted")

X_pla = np.zeros((2,nsamp))
X_mala = np.zeros((2,nsamp,2))

for i in range(maxiter):
    X_malan = X_mala + np.sqrt(h1)*np.random.normal(size=(2, nsamp,2))
    X_pla = X_pla + np.sqrt(h0)*np.random.normal(size=(2, nsamp))
    X_mala[0,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)] = X_malan[0,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)]
    X_mala[1,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)] = X_malan[1,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)]
    X_pla[0,X_pla[0,:]>1] = 1
    X_pla[0,X_pla[0,:]<-1] = -1
    X_pla[1,X_pla[1,:]>a] = a
    X_pla[1,X_pla[1,:]<-a] = -a

fig = plt.figure(figsize=(3.5,7))
ax = fig.add_subplot(111)
plt.xlim([-0.013, 0.013])
plt.ylim([-1.15,1.15])
ax.scatter(X_pla[1,:], X_pla[0,:])
ax.scatter(X_nla2[0], X_nla1[0], c='darkorange')
ax.grid(True)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
plt.savefig('rect.pdf', bbox_inches='tight')




# Generate Figure 8

nsamp = 1000
d = 2
maxiter = 100
step_nla = 0.000001
c = 1./10000
a = 0.01
eps = 0.01
sigma = np.matrix([[1,0],[0,a**2]])
sigma_half = np.matrix([[1,0],[0,a]])
step_pla = 0.00001
step_mala = 0.01
rep = 20

# generate uniform random sample of size n from 2d
# ellipse with covariance [[1, 0], [0, a^2]]
#
# returns samples of shape (2, n)
# def ellipse_samp(n, d = 2):
#   # this is hacky:
#   i = 0
#   res = []
#   while i < n:
#     x = np.array([2*np.random.rand() - 1, 2*np.random.rand() - 1])
#     if np.linalg.norm(x) <= 1.:
#       x[1] = x[1]/a
#       res.append(x)
#       i += 1

#   return np.array(res).transpose()

def rect_samp(n, d = 2):
  # this is hacky:
  x = np.zeros((d,n))
  x[0,:] = 2*np.random.rand(n) - 1
  x[1,:] = 2*a*np.random.rand(n) - a

  return x

# compute sinkhorn distance between mu and nu
# mu and nu are assume to be of shape (2, nsamp)
def sinkhorn(mu, nu, eps):
  d, n = mu.shape
  
  res = 0.

  a, b = np.ones((n,))/n, np.ones((n,))/n
  
  # cost matrix
    
  C = np.array([[np.linalg.norm(mu[:,i] - nu[:,j])**2
    for i in range(0, n)] for j in range(0, n)])
    
  m = C.max()
  C = C/m
    
  pi_eps = np.array(ot.sinkhorn(a, b, C, eps))

  C = C*m
  res += ((np.trace(np.matmul(np.transpose(C), pi_eps)))**0.5)
  
  return res
# Target two dimensional ellipse
# def V(x):
#   return c*(- np.log(1+np.linalg.norm(x)) - np.log(1-np.linalg.norm(x)))

# def grad_V(x):
#   x_sigma2 = x[0]**2 + a**2 * x[1]**2
#   sigma_x = sigma.dot(x)
#   return c*(2/(1-x_sigma2)) * sigma_x

# def grad_V_star(y):
#   k = np.linalg.norm(1/c * np.linalg.inv(sigma_half).dot(np.transpose(y)))
#   x_sigma = (-1 + np.sqrt(1 + k**2))/k
#   return 1/(2*c)*(1 - x_sigma**2)* np.linalg.inv(sigma).dot(np.transpose(y))

# def H_V(x):
#   # print(x.shape)
#   # Austin: is this missing a factor of c?
#   hess = np.zeros((2,2))
#   x_sigma2 = x[0]**2 + a**2 * x[1]**2
#   k1 = 4/np.power((x_sigma2 - 1), 2)
#   k2 = 2/(1-x_sigma2)
#   hess[0,0] = k1*np.power(x[0],2) + k2
#   hess[1,0] = k1*(a**2)*x[0]*x[1]
#   hess[0,1] = hess[1,0]
#   hess[1,1] = k1*(a**4)*(x[1]**2) + k2*(a**2)
#   return (c**0.5)*sc.linalg.sqrtm(hess)


def V1(x):
  return c*(- np.log(1+x) - np.log(1-x))

def grad_V1(x):
  return c*(-1/(x+1) + 1/(1-x))

def grad_V_star1(x):
  y = x
  y[x>0] = - c/x[x>0] + np.sqrt(np.power(c/x[x>0], 2) + 1)
  y[x==0] = 0
  y[x<0] = - c/x[x<0] - np.sqrt(np.power(c/x[x<0], 2) + 1)
  return y

def V2(x):
  return c*(- np.log(a+x) - np.log(a-x))

def grad_V2(x):
  return c*(-1/(x+a) + 1/(a-x))

def grad_V_star2(x):
  y = x
  y[x>0] = - c/x[x>0] + np.sqrt(np.power(c/x[x>0], 2) + a**2)
  y[x==0] = 0
  y[x<0] = - c/x[x<0] - np.sqrt(np.power(c/x[x<0], 2) + a**2)
  return y

def H_V1(x):
  # print(np.sqrt(c*(1/(np.power(1+x, 2)) + 1/(np.power(1-x, 2)))))
  return np.sqrt(c*(1/(np.power(1+x, 2)) + 1/(np.power(1-x, 2))))

def H_V2(x):
  # print(np.sqrt(c*(1/(np.power(1+x, 2)) + 1/(np.power(1-x, 2)))))
  return np.sqrt(c*(1/(np.power(a+x, 2)) + 1/(np.power(a-x, 2))))


# initialize with zeros
X_nla = np.zeros(shape=(2,nsamp,rep))
X_nla_loss = np.zeros(shape=(maxiter,rep))
# initialization of projected ULA
X_pla = np.zeros(shape=(2,nsamp,rep))
X_pla_loss = np.zeros(shape=(maxiter,rep))

X_mala = np.zeros(shape=(2,nsamp,rep))
X_mala_loss = np.zeros(shape=(maxiter,rep))

bias = sinkhorn(rect_samp(nsamp), rect_samp(nsamp), eps)
print("bias = " + str(bias))

# for k in range(rep):
#     for i in range(maxiter):
#       X_pla[:,:,k] = X_pla[:,:,k] + np.sqrt(step_pla)*np.random.normal(size=(2, nsamp))
#       for j in range(nsamp):
#         step = h0
#         h2 = H_V(X_nla[:,j,k])
#         noise = np.random.normal(size=(2,1))
#         X_nla[:,j,k] = np.transpose(grad_V_star(grad_V(X_nla[:,j,k]) - 
#                           step * grad_V(X_nla[:,j,k]) + 
#                           ((2 * step)**0.5) *
#                           np.transpose(np.matmul(h2,noise))))

#         # projection for ULA
#         xnorm = np.sqrt(X_pla[0,j,k]**2 + a**2 * X_pla[1,j,k]**2)
#         if xnorm > 1:
#           X_pla[:,j,k] = X_pla[:,j,k]/xnorm

for i in range(maxiter):
  X_nla[0,:,:] = grad_V_star1(grad_V1(X_nla[0,:,:]) - 
                    step_nla * grad_V1(X_nla[0,:,:]) + 
                    math.sqrt(2 * step_nla) * 
                    H_V1(X_nla[0,:,:])*(np.random.normal(size = (1,nsamp,rep))))
  # print("printing hess 2 from 1d")
  X_nla[1,:,:] = grad_V_star2(grad_V2(X_nla[1,:,:]) - 
                      step_nla * grad_V2(X_nla[1,:,:]) + 
                      math.sqrt(2 * step_nla) * 
                      H_V2(X_nla[1,:,:])*(np.random.normal(size = (1,nsamp,rep))))

  X_malan = X_mala + np.sqrt(step_mala)*np.random.normal(size=(2, nsamp,rep))
  X_pla = X_pla + np.sqrt(step_pla)*np.random.normal(size=(2, nsamp,rep))
  X_mala[0,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)] = X_malan[0,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)]
  X_mala[1,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)] = X_malan[1,np.logical_and(np.abs(X_malan[0,:,:])<1, np.abs(X_malan[1,:,:])<a)]
  X_pla[0,X_pla[0,:,:]>1] = 1
  X_pla[0,X_pla[0,:,:]<-1] = -1
  X_pla[1,X_pla[1,:,:]>a] = a
  X_pla[1,X_pla[1,:,:]<-a] = -a

  for k in range(rep):
    # calculated losses
    unif_samp = rect_samp(nsamp)
    X_nla_loss[i,k] = sinkhorn(unif_samp, X_nla[:,:,k], eps) - bias
    X_pla_loss[i,k] = sinkhorn(unif_samp, X_pla[:,:,k], eps) - bias
    X_mala_loss[i,k] = sinkhorn(unif_samp, X_mala[:,:,k], eps) - bias
  print("finished iteration "+str(i))







np.save("X_nla_loss2", X_nla_loss)
np.save("X_pla_loss2", X_pla_loss)
np.save("X_mala_loss2", X_mala_loss)



