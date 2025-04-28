import numpy as np
import cvxopt 
from scipy import linalg
import matplotlib.pyplot as plt
from torch import FloatTensor
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

from dataclasses import dataclass

@dataclass()
class OneClassSVMClassifier(object):
    X: FloatTensor
    nu: float

    def __post_init__(self):
        self.gamma = self.find_best_gamma()
        self.decision = None

    def fit(self):
        K = self.rbf_kernel(self.X, self.X)
        self.alpha_support, idx_support = self.qp_solver(K)
        self.rho = self.compute_rho(K, self.alpha_support, idx_support)
        self.X_support = self.X[idx_support]
        G = self.rbf_kernel(self.X, self.X_support)
        self.decision = G.dot(self.alpha_support) - self.rho
        return self.decision, np.sign(self.decision)

    def predict(self, X_test):
        K_test = self.rbf_kernel(X_test, self.X_support)  # Kernel between test and train
        scores = K_test.dot(self.alpha_support) - self.rho
        return scores
    
    
    def find_best_gamma(self):
        pairwise_sq_dists = squareform(pdist(self.X, 'sqeuclidean'))  
        median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  
        return 1/median_dist 
    
    def rbf_kernel(self, X1, X2):
        return rbf_kernel(X1, X2, gamma=self.gamma)

    def _rbf_metric(self, x, y):
        return np.exp(-self.gamma * linalg.norm(x - y, 2)**2)
    
    def compute_rho(self, K, alpha_support, idx_support):
        index = int(np.argmin(alpha_support))
        K_support = K[idx_support][:, idx_support]
        rho = alpha_support.dot(K_support[index])
        return rho

    def qp_solver(self, K): 
        n = len(K)
        ones = np.ones(shape=(n,1))
        zeros = np.zeros(shape=(n,1))
        P = cvxopt.matrix(K)
        q = cvxopt.matrix(zeros)
        G = cvxopt.matrix(np.vstack((-np.identity(n), np.identity(n))))
        C = 1.0/(self.nu*n)
        h = cvxopt.matrix(np.vstack((zeros, C*ones)))
        A = cvxopt.matrix(ones.T)
        b = cvxopt.matrix(1.0)
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')
        alpha =np.ravel(solution['x'])
        idx_support = np.where(np.abs(alpha) > 1e-4)[0]
        alpha_support = alpha[idx_support] * self.nu * len(K)
        return alpha_support, idx_support