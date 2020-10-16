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
from time import time

# Mirror Langevin class, can initialize with the desired functions to run the descent
# currently, only the 1-d and multivariate Gaussian versions work
class mirrorLangevinMC:
  # initialize the class with the necessary functions to run the diffusion 
  # (don't actually need V or phi for anything)
    def __init__(self, V, grad_V, grad_V_star, H_V,
                inv_grad_V = False, inv_grad_phi = False):
        self.V = V
        self.grad_V = grad_V
        self.H_V = H_V
        self.grad_V_star = grad_V_star
        self.inv_grad_V = inv_grad_V
        self.inv_grad_phi = inv_grad_phi
#         self.phi = phi
#         self.grad_phi = grad_phi
#         self.H_phi = H_phi
#         self.grad_phi_star = grad_phi_star
#         self.true_samples = true_samples
#         self.inv_grad_phi = inv_grad_phi

  # invert the gradient of the mirror map with damped newton
    def grad_V_inv_opt(self, x, warm):
        y = warm.copy()
        newton_step = np.linalg.lstsq(self.H_V(y), x - self.grad_V(y), rcond=0)[0]
        #print(y)
        #print(newton_step)
        # damped newton ascent
        for i in range(10):
        #while np.linalg.norm(newton_step) > 1e-3:
            #print(np.linalg.norm(newton_step))
            #y = y + .1 * np.linalg.pinv(self.H_phi(y)).dot(x - self.grad_phi(y))
            newton_step = np.linalg.lstsq(self.H_V(y), x - self.grad_V(y), rcond=0)[0]
            y = y + .1 * newton_step
        return y
    
    def grad_phi_inv_opt(self, x, warm):
        y = warm.copy()
        newton_step = np.linalg.lstsq(self.H_phi(y), x - self.grad_phi(y), rcond=0)[0]
        #print(y)
        #print(newton_step)
        # damped newton ascent
        for i in range(10):
        #while np.linalg.norm(newton_step) > 1e-3:
            #print(np.linalg.norm(newton_step))
            #y = y + .1 * np.linalg.pinv(self.H_phi(y)).dot(x - self.grad_phi(y))
            newton_step = np.linalg.lstsq(self.H_phi(y), x - self.grad_phi(y), rcond=0)[0]
            y = y + .1 * newton_step
        return y
  

  #######################################################
  # DESCENT FUNCTIONS
  
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
        return m1, m2, Y

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
    
    def UNLA(self, x0, v0, step, N, burn_in = 10**3, L = 1, quiet = False):
        X = x0
        d = x0.size
        V = v0
        m1 = np.zeros(d)
        m2 = np.zeros((d, d))
        Y = np.zeros((d, N))
        C = np.zeros((2*d,2*d))
        M = np.zeros(2*d)
        ts = np.zeros(N)    
        for i in tqdm(range(burn_in), disable = quiet):
            grad = self.grad_V(X)
            mvi = np.exp(-2*step) * V - (1 / (2 * L)) * (1 - np.exp(-2*step)) * grad
            mxi = X + V * (1 - np.exp(-2*step))/2 - grad * (step - (1 - np.exp(-2*step))/2)/(2 * L) 
            
            Cxi = np.eye(d) * (step - np.exp(-4*step)/4 - 3/4 + np.exp(-2*step)) / L
            Cvi = np.eye(d) * (1 - np.exp(-4*step)) / L
            Cxvi = np.eye(d) * (1 + np.exp(-4*step) - 2 * np.exp(-2*step)) / (2*L)
            
            M[:d] = mxi
            M[d:] = mvi
            C[:d,:d] = Cxi
            C[d:,d:] = Cvi
            C[:d,d:] = Cxvi
            C[d:,:d] = Cxvi
            Z = np.random.multivariate_normal(M, C)

            X = Z[:d]
            V = Z[d:]
        
        for i in tqdm(range(N), disable = quiet):
            grad = self.grad_V(X)
            mvi = np.exp(-2*step) * V - (1 / (2 * L)) * (1 - np.exp(-2*step)) * grad
            mxi = X + V * (1 - np.exp(-2*step))/2 - grad * (step - (1 - np.exp(-2*step))/2)/(2 * L) 
            
            Cxi = np.eye(d) * (step - np.exp(-4*step)/4 - 3/4 + np.exp(-2*step)) / L
            Cvi = np.eye(d) * (1 - np.exp(-4*step)) / L
            Cxvi = np.eye(d) * (1 + np.exp(-4*step) - 2 * np.exp(-2*step)) / (2*L)
            
            M[:d] = mxi
            M[d:] = mvi
            C[:d,:d] = Cxi
            C[d:,d:] = Cvi
            C[:d,d:] = Cxvi
            C[d:,:d] = Cxvi
            Z = np.random.multivariate_normal(M, C)

            X = Z[:d]
            V = Z[d:]

            m1 += X / N
            m2 += np.outer(X, X) / N
            Y[:, i] = X
            ts[i] = time()
                    
        return m1, m2, Y, ts


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
    
    def HAMCMC(self, x0, step, N, Q, burn_in = 10**3, quiet = False):
        X = x0
        d = x0.size
        m1 = np.zeros(d)
        m2 = np.zeros((d, d))
        Y = np.zeros((d, N))
        H = np.eye(d)
        # First, Q=2M+1 ULA steps 
        XQ = np.zeros((d,Q))
        XQ[:, 0] = X
        idxM = int((Q-1)/2)
        for i in tqdm(range(Q-1), disable = quiet):
            Z = np.random.normal(size = (d,))
            grad = self.grad_V(X)
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
            XQ[:, i + 1] = X
        ts = np.zeros(N)
        H = self.compute_H(XQ)
        for i in tqdm(range(burn_in), disable = quiet):
            # step
            grad = H.dot(self.grad_V(XQ[:,idxM]))
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
            
            #update XQ
            XQ[:,0:-1] = XQ[:,1:]
            XQ[:,Q-1] = X
            #update H
            H = self.compute_H(XQ)
        for i in tqdm(range(N), disable = quiet):
            m1 += X / N
            m2 += np.outer(X, X) / N
            # step
            grad = H.dot(self.grad_V(XQ[:,idxM]))
            X = X - step * grad + np.sqrt(2*step)*np.random.multivariate_normal(np.zeros((d,)), np.eye(d))
            
            #update XQ
            XQ[:,0:-1] = XQ[:,1:]
            XQ[:,Q-1] = X
            #update H
            H = self.compute_H(XQ)
            Y[:, i] = X
            ts[i] = time()
        return m1, m2, Y, ts

    def init_scat_dist(self, Sigma, u):
        self.Sigma = Sigma
        self.u = u 
        self.use_scat_dist = True





##########################
# Euclidean mirror map example
# input: vector
# output: scalar, vector, matrix
def phi2(x):
  return math.pow(np.linalg.norm(x), 2) / 2
def grad_phi2(x):
  return x
def H_phi2(x):
  return np.eye(x.size)

#########################
# Other useful functions
def ilogit(z):
  return np.exp(z) / (1 + np.exp(z))
    
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


def k(beta, h):
  return lambda x, y: np.power((1 + 1 / h * np.power(np.linalg.norm(x - y), 2)), beta)
def kx(beta, h):
  return lambda x, y: (2 * 1 / h  * beta * np.power((1 + 1 / h * np.power(np.linalg.norm(x - y), 2)), beta - 1) * 
          (x - y))
def ky(beta, h):
  return lambda x, y: (2 * 1 / h * beta * np.power((1 + 1 / h * np.power(np.linalg.norm(x - y), 2)), beta - 1) * 
          (y - x))
def kxy(beta, h):
  return lambda x, y: (4 * (1 / h ** 2)  * beta * (beta - 1) * np.power((1 + 1 / h * np.power(np.linalg.norm(x - y), 2)), beta - 2) * 
          np.outer(x - y, y - x) - 
          2 * 1 / h * beta * np.power((1 + 1 / h * np.power(np.linalg.norm(x - y), 2)), beta - 2) * 
          np.eye(x.size))  


def mean_dist_true(m_true, Y):
    d, n = Y.shape
    running_mean = Y[:, 0]
    conv = np.zeros(n)
    conv[0] = np.power(np.linalg.norm(running_mean - m_true), 2)
    for i in range(n-1):
        running_mean = (i+1) * running_mean / (i+2) + (1) * Y[:, i] / (i+2)
        conv[i+1] = np.power(np.linalg.norm(running_mean - m_true), 2)
    return conv

def cov_dist_true(C_true, Y):
    d, n = Y.shape
    C_running = np.outer(Y[:, 0], Y[:, 0])
    conv = np.zeros(n)
    conv[0] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    for i in range(n-1):
        C_running = (i+1) * C_running / (i+2) + (1) * np.outer(Y[:, i], Y[:, i]) / (i+2)
        conv[i+1] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    return conv

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

def scat_dist_true(C_true, Y, u):
    d, n = Y.shape
    C_running = np.outer(Y[:, 0], Y[:, 0]) * u(Y[:, 0].dot(Y[:, 0]))
    conv = np.zeros(n)
    conv[0] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    for i in tqdm(range(n-1)):
        C_running = calc_scatter(Y[:, 0:(i+2)], u, C_true)
        conv[i+1] = np.power(np.linalg.norm(C_running - C_true), 1) / np.power(np.linalg.norm(C_true), 1)
    return conv
