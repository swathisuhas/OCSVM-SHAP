import numpy as np
import cvxopt 
from scipy import linalg
import matplotlib.pyplot as plt
from torch import FloatTensor
from scipy.spatial.distance import pdist, squareform
from sklearn.svm import OneClassSVM

from dataclasses import dataclass, field
from typing import Optional

class OneClassSVMModel:
    def __init__(self, nu, gamma):
        self.nu = nu
        self.gamma = gamma
        self.decision = None
    
    def rbf_kernel(self, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._rbf_metric(X1[i], X2[j])
        return K

    def _rbf_metric(self, x, y):
        return np.exp(-self.gamma * linalg.norm(x - y, 2)**2)

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
        print(alpha_support)
        print(idx_support)
        return alpha_support, idx_support

    def compute_rho(self, K, alpha_support, idx_support):
        index = int(np.argmin(alpha_support))
        K_support = K[idx_support][:, idx_support]
        rho = alpha_support.dot(K_support[index])
        print(rho)
        return rho

    def fit(self, X):
        K = self.rbf_kernel(X, X)
        print(K)
        alpha_support, idx_support = self.qp_solver(K)
        rho = self.compute_rho(K, alpha_support, idx_support)
        X_support = X[idx_support]
        G = self.rbf_kernel(X, X_support)
        self.decision = G.dot(alpha_support) - rho
        return self.decision, np.sign(self.decision)

@dataclass()
class OneClassSVMClassifier(object):
    X: FloatTensor
    nu: float
    model: Optional['OneClassSVMModel'] = field(init=False, default=None)

    def __post_init__(self):
        gamma = self.find_best_gamma()
        self.model = OneClassSVMModel(nu=self.nu, gamma=gamma)

    def fit(self):
        return self.model.fit(self.X)

    def plot(self, x1, x2, y1, y2):
        return self.model.plot_ocsvm(self.X.numpy(), x1, x2, y1, y2)
    
    def find_best_gamma(self):
        pairwise_sq_dists = squareform(pdist(self.X, 'sqeuclidean'))  # Squared Euclidean distances
        median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  # Ignore zero distances
        return 1/median_dist 