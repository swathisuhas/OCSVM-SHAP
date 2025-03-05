import numpy as np
import cvxopt 
from scipy import linalg
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from torch import FloatTensor


from dataclasses import dataclass, field
from typing import Optional

class OneClassSVMModel:
    def __init__(self, nu, gamma):
        self.nu = nu
        self.gamma = gamma
        self.rho = None
        self.alpha_support = None
        self.idx_support = None
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

    def fit(self, X):
        K = self.rbf_kernel(X, X)
        self.alpha_support, self.idx_support = ocsvm_solver(K, self.nu)
        self.rho = compute_rho(K, self.alpha_support, self.idx_support)
        X_support = X[self.idx_support]
        G = self.rbf_kernel(X, X_support)
        self.decision = G.dot(self.alpha_support) - self.rho
        return self.decision, np.sign(self.decision)
        
    def plot_ocsvm(self, X, x1, x2, y1, y2):
        # Compute decision function on a grid
        X1, X2 = np.mgrid[x1:x2+0.1:0.2, y1:y2+0.1:0.2]
        na, nb = X1.shape
        X_test = np.c_[np.reshape(X1, (na * nb, 1)),
                    np.reshape(X2, (na * nb, 1))]

        # Compute dot products
        X_support = X[self.idx_support]
        G = self.rbf_kernel(X_test, X_support)
        # Compute decision function
        decision = G.dot(self.alpha_support) - self.rho # rho is needed only for decion boundary not for finding alphas)

        # Compute predict label
        y_pred = np.sign(decision)

        # Plot decision boundary
        plt.plot(X[:,0], X[:, 1], 'ob', linewidth=2)
        Z = np.reshape(decision, (na, nb))
        plt.contourf(X1, X2, Z, 20, cmap=plt.cm.gray)
        cs = plt.contour(X1, X2, Z, [0], colors='y', linewidths=2, zorder=10)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim([x1, x2])
        plt.ylim([y1, y2])


@dataclass()
class OneClassSVMClassifier(object):
    X: FloatTensor
    nu: float
    gamma: float
    model: Optional['OneClassSVMModel'] = field(init=False, default=None)
    num_inducing_points: int = field(default=None)

    def __post_init__(self):
        self.model = OneClassSVMModel(nu=self.nu, gamma=self.gamma)

    def fit(self):
        return self.model.fit(self.X)

    def plot(self, x1, x2, y1, y2):
        return self.model.plot_ocsvm(self.X.numpy(), x1, x2, y1, y2)
    
def qp(P, q, A, b, C):   # quadratic programming problem solver
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

def ocsvm_solver(K, nu):  # nu default is 0.1
    n = len(K)
    P = K
    q = np.zeros(n)
    A = np.matrix(np.ones(n))
    b = 1.
    C = 1. / (nu * n)
    alpha = qp(P, q, A, b, C)
    idx_support = np.where(np.abs(alpha) > 1e-5)[0] # if alpha is greater than 1e-5 then it is considered a support vector
    alpha_support = alpha[idx_support] * nu * len(K)  # multipling with nu * len(K) to match values from sklearn
    return alpha_support, idx_support

def compute_rho(K, alpha_support, idx_support):
    index = int(np.argmin(alpha_support))
    K_support = K[idx_support][:, idx_support] 
    rho = alpha_support.dot(K_support[index])
    return rho