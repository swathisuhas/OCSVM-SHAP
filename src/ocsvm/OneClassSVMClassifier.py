import numpy as np
import cvxopt 
from scipy import linalg
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Optional

class OneClassSVMModel:
    def __init__(self, nu=0.1, gamma=0.3):
        self.nu = nu
        self.gamma = gamma
        self.rho = None
        self.mu_support = None
        self.idx_support = None
    
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
        
    def fast_rbf_kernel(self, X1, X2):
        return pairwise_distances(X1, X2, metric=self._rbf_metric)
    
    def qp(self, P, q, A, b, C):   # quadratic programming problem solver
        # Gram matrix
        n = P.shape[0]
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        G = cvxopt.matrix(np.concatenate(
            [np.diag(np.ones(n) * -1), np.diag(np.ones(n))], axis=0))
        h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

        # Solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')
        return np.ravel(solution['x'])
    
    def ocsvm_solver(self, K):  # nu default is 0.1
        n = len(K)
        P = K
        q = np.zeros(n)
        A = np.matrix(np.ones(n))
        b = 1.
        C = 1. / (self.nu * n)
        mu = self.qp(P, q, A, b, C)
        self.idx_support = np.where(np.abs(mu) > 1e-5)[0] # if mu is greater than 1e-5 then it is considered a support vector
        self.mu_support = mu[self.idx_support]
        return self.mu_support, self.idx_support
    
    def compute_rho(self, K):
        index = int(np.argmin(self.mu_support))
        K_support = K[self.idx_support][:, self.idx_support] # rho = sum(mu_i.(K(x_i, x_j)) K between that support vectorn and all support vectors before it
        self.rho = self.mu_support.dot(K_support[index])
        return self.rho

    def fit(self, X):
        K = self.rbf_kernel(X, X)
        self.mu_support, self.idx_support = self.ocsvm_solver(K)
        self.rho = self.compute_rho(K)
        # self.plot_ocsvm(X)
        decision, y_pred = self.decision_function(X)
        return decision, y_pred

    def decision_function(self, X):
        X_support = X[self.idx_support]
        G = self.rbf_kernel(X, X_support)
        # Compute decision function
        decision = G.dot(self.mu_support) - self.rho
        y_pred = np.sign(decision)
        return decision, y_pred
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
        
    def plot_ocsvm(self, X):
        # Compute decision function on a grid
        X1, X2 = np.mgrid[-1:5.1:0.2, -1:5.1:0.2]
        na, nb = X1.shape
        X_test = np.c_[np.reshape(X1, (na * nb, 1)),
                    np.reshape(X2, (na * nb, 1))]

        # Compute dot products
        X_support = X[self.idx_support]
        G = self.rbf_kernel(X_test, X_support)
        # Compute decision function
        decision = G.dot(self.mu_support) - self.rho # rho is needed only for decion boundary not for finding mus)

        # Compute predict label
        y_pred = np.sign(decision)

        # Plot decision boundary
        plt.plot(X[:,0], X[:, 1], 'ob', linewidth=2)
        Z = np.reshape(decision, (na, nb))
        plt.contourf(X1, X2, Z, 20, cmap=plt.cm.gray)
        cs = plt.contour(X1, X2, Z, [0], colors='y', linewidths=2, zorder=10)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim([-1, 5])
        plt.ylim([-1, 5])



@dataclass()
class OneClassSVMClassifier(object):
    nu: float = 0.1
    gamma: float = 0.3
    model: Optional['OneClassSVMModel'] = field(init=False, default=None)

    def __post_init__(self):
        self.model = OneClassSVMModel(nu=self.nu, gamma=self.gamma)

    def fit(self, X):
        return self.model.fit(X)

    def plot(self, X):
        return self.model.plot_ocsvm(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def decision(self, X):
        return self.model.decision_function(X)