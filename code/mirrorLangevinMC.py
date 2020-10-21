import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.linalg import sqrtm
from scipy.linalg import null_space
from scipy.special import gamma, logit
from tqdm import tqdm
from time import time

# Mirror Langevin class, can initialize with the desired functions to run the descent
class mirrorLangevinMC:
  # initialize the class with the necessary functions to run the diffusion 
    def __init__(self, V, grad_V, grad_V_star, H_V,
                inv_grad_V = False, inv_grad_phi = False):
        self.V = V
        self.grad_V = grad_V
        self.H_V = H_V
        self.grad_V_star = grad_V_star
        self.inv_grad_V = inv_grad_V
        self.inv_grad_phi = inv_grad_phi
    
    #######################################################
    # If gradient of Legendre transform has no closed form, then invert numerically 

    # invert the gradient of the mirror map with damped newton
    def grad_V_inv_opt(self, x, warm):
        y = warm.copy()
        newton_step = np.linalg.lstsq(self.H_V(y), x - self.grad_V(y), rcond=0)[0]
        # damped newton ascent
        for i in range(10):
        #while np.linalg.norm(newton_step) > 1e-3:
            newton_step = np.linalg.lstsq(self.H_V(y), x - self.grad_V(y), rcond=0)[0]
            y = y + .1 * newton_step
        return y
    
    def grad_phi_inv_opt(self, x, warm):
        y = warm.copy()
        newton_step = np.linalg.lstsq(self.H_phi(y), x - self.grad_phi(y), rcond=0)[0]
        # damped newton ascent
        for i in range(10):
        #while np.linalg.norm(newton_step) > 1e-3:
            newton_step = np.linalg.lstsq(self.H_phi(y), x - self.grad_phi(y), rcond=0)[0]
            y = y + .1 * newton_step
        return y
    
    #######################################################
    # DESCENT FUNCTIONS
    
    # Newton Langevin Algorithm
    def NLA(self, x0, step, N, burn_in = 10**3, quiet = False):
        X = x0
        d = x0.size
        m1 = np.zeros(d)
        m2 = np.zeros((d, d))
        Y = np.zeros((d, N))
        for i in tqdm(range(burn_in), disable = quiet):
            if self.inv_grad_V:
                X = self.grad_V_inv_opt(self.grad_V(X) - 
                                        step * self.grad_V(X) + 
                                        np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_V(X)),
                                        X)
            else:
                X = self.grad_V_star(self.grad_V(X) - 
                                    step * self.grad_V(X) + 
                                    np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_V(X)))
        
        ts = np.zeros(N)
        for i in tqdm(range(N), disable = quiet):
            m1 += X / N
            m2 += np.outer(X, X) / N
            if self.inv_grad_V:
                X = self.grad_V_inv_opt(self.grad_V(X) - 
                                        step * self.grad_V(X) + 
                                        np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_V(X)),
                                        X)
            else:
                X = self.grad_V_star(self.grad_V(X) - 
                                    step * self.grad_V(X) + 
                                    np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_V(X)))
            Y[:, i] = X 
            ts[i] = time()
        return m1, m2, Y, ts 
    

    # Mirror Langevin Algorithm
    def MLA(self, x0, step, N, burn_in = 10**3, quiet = False):
        X = x0
        d = x0.size
        m1 = np.zeros(d)
        m2 = np.zeros((d, d))
        Y = np.zeros((d, N))
        for i in tqdm(range(burn_in), disable = quiet):
          if self.inv_grad_phi:
            X = self.grad_phi_inv_opt(self.grad_phi(X) -
                    step * self.grad_V(X) +
                    np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_phi(X)),
                    X)
          else:
            X = self.grad_phi_star(self.grad_phi(X) -
                    step * self.grad_V(X) +
                    np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_phi(X)))

        ts = np.zeros(N)
        for i in tqdm(range(N), disable = quiet):
          m1 += X / N
          m2 += np.outer(X, X) / N
          if self.inv_grad_phi:
            X = self.grad_phi_inv_opt(self.grad_phi(X) -
                    step * self.grad_V(X) +
                    np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_phi(X)),
                    X)
          else:
            X = self.grad_phi_star(self.grad_phi(X) -
                    step * self.grad_V(X) +
                    np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), self.H_phi(X)))
          Y[:, i] = X
          ts[i] = time()
        return m1, m2, Y, ts
    
    # Unadjusted Langevin Algorithm
    def ULA(self, x0, step, N, burn_in = 10**3, quiet = False):
        X = x0
        d = x0.size
        m1 = np.zeros(d)
        m2 = np.zeros((d, d))
        Y = np.zeros((d, N))
        for i in tqdm(range(burn_in), disable = quiet):
            grad = self.grad_V(X)
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
        ts = np.zeros(N)    
        for i in tqdm(range(N), disable = quiet):
            m1 += X / N
            m2 += np.outer(X, X) / N
            grad = self.grad_V(X)
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
            Y[:, i] = X 
            ts[i] = time()
        return m1, m2, Y, ts 

    # Tamed Unadjusted Langevin Algorithm 
    def TULA(self, x0, step, N, burn_in = 10**3, gamma = 0.1, quiet = False):
        X = x0
        d = x0.size
        m1 = np.zeros(d)
        m2 = np.zeros((d, d))
        Y = np.zeros((d, N))
        for i in tqdm(range(burn_in), disable = quiet):
            grad = self.grad_V(X)
            grad = grad / (1 + gamma * np.linalg.norm(grad))
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
        ts = np.zeros(N)    
        for i in tqdm(range(N), disable = quiet):
            m1 += X / N
            m2 += np.outer(X, X) / N
            grad = self.grad_V(X)
            grad = grad / (1 + step * np.linalg.norm(grad))
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
            Y[:, i] = X
            ts[i] = time()
        return m1, m2, Y, ts
    

##############################################################################
# Euclidean mirror map example
# input: vector
# output: scalar, vector, matrix
def phi2(x):
  return math.pow(np.linalg.norm(x), 2) / 2
def grad_phi2(x):
  return x
def H_phi2(x):
  return np.eye(x.size)

##############################################################################
# vector version
def V_p(p):
  return lambda x: np.power(np.linalg.norm(x), p)
def grad_V_p(p):
  return lambda x: p * np.power(np.linalg.norm(x), (p - 2)) * x
def grad_V_p_star(p):
  return lambda y: (np.power(np.linalg.norm(y) / p,  (1 / (p - 1))) * y / np.linalg.norm(y))
def Hess_V_p(p, d):
  return lambda x: (p * np.power(np.linalg.norm(x), (p - 2)) *
    (np.eye(d) + (p - 2) / (np.power(np.linalg.norm(x), 2)) * np.outer(x, x)))
def p_density(p, d):
  return lambda x: (p * gamma(d / 2) * math.exp(-np.power(np.linalg.norm(x), p)) /
    (2 * (math.pow(math.pi, (d / 2))) * gamma(d / p)))

# coordinate version
def V1_p(p):
  return lambda x: sum(np.power(np.abs(x), p))
def grad_V1_p(p):
  return lambda x: p * np.power(np.abs(x), p-2) * x
def grad_V1_p_star(p):
  return lambda y: np.power(np.abs(y) / p, 1/(p-1)) * np.sign(y)
def Hess_V1_p(p, d):
  return lambda x: p * np.diag((p-2) * np.power(np.abs(x), p-4) * np.power(x, 2) + np.power(np.abs(x), p-2))
def p_density(p, d):
  return lambda x: ((p / (2 * math.gamma(1/p))) ** d) * math.exp(-V_p(p)(x))


def scale_V(V, A):
  return lambda x: V(A.dot(x))
def scale_grad_V(grad_V, A):
  return lambda x: A.T.dot(grad_V(A.dot(x)))
def scale_grad_V_star(grad_V_star, A):
  return lambda y: np.linalg.pinv(A).dot(grad_V_star(np.linalg.pinv(A).dot(y)))
def scale_Hess_V(Hess_V, A):
  return lambda x: A.T.dot(Hess_V(A.dot(x))).dot(A)


##############################################################################
# Other useful functions

# inverse logit for logistic regression example
def ilogit(z):
  return np.exp(z) / (1 + np.exp(z))


# calculate l2 distance between running average and true mean 
def mean_dist_true(m_true, Y):
    d, n = Y.shape
    running_mean = Y[:, 0]
    conv = np.zeros(n)
    conv[0] = np.power(np.linalg.norm(running_mean - m_true), 2)
    for i in range(n-1):
        running_mean = (i+1) * running_mean / (i+2) + (1) * Y[:, i] / (i+2)
        conv[i+1] = np.power(np.linalg.norm(running_mean - m_true), 2)
    return conv

# calculate l2 distance between running covariance and true covariance 
def cov_dist_true(C_true, Y):
    d, n = Y.shape
    C_running = np.outer(Y[:, 0], Y[:, 0])
    conv = np.zeros(n)
    conv[0] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    for i in range(n-1):
        C_running = (i+1) * C_running / (i+2) + (1) * np.outer(Y[:, i], Y[:, i]) / (i+2)
        conv[i+1] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    return conv

# calculate l2 distance between running scatter matrix and true scatter matrix 
def scat_dist_true(C_true, Y, u):
    d, n = Y.shape
    C_running = np.outer(Y[:, 0], Y[:, 0]) * u(Y[:, 0].dot(Y[:, 0]))
    conv = np.zeros(n)
    conv[0] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    for i in tqdm(range(n-1)):
        C_running = calc_scatter(Y[:, 0:(i+2)], u, C_true)
        conv[i+1] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    return conv
# calculate the scatter matrix for a given u function
# see
def calc_scatter(X, u, Sigma):
    if len(X.shape) == 1:
        d = X.size
        n = 1
    else:
        d, n = X.shape
    S = Sigma
    Y = X.copy()
    for i in range(20):
        Y = X * np.outer(np.ones((d,)), u(np.sum(X*(np.linalg.pinv(S).dot(X)), axis = 0)))
        S = Y.dot(X.T) / n   
    return S


