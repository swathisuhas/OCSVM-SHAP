import numpy as np
import cvxopt 
from scipy import linalg
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from torch import FloatTensor

from dataclasses import dataclass, field
from typing import Optional

class OneClassSMMModel:
    def __init__(self, nu, gamma_x, gamma_d):
        self.nu = nu
        self.gamma_x = gamma_x  # Gamma for the instance-level kernel
        self.gamma_d = gamma_d  # Gamma for the distribution-level kernel
        self.alpha = None
        self.rho = None
        self.idx_support = None

    def rbf_kernel(self, X1, X2, gamma):
        """Compute the RBF kernel."""
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = np.exp(-gamma * linalg.norm(X1[i] - X2[j], 2)**2)
        return K

    def compute_mean_embedding(self, datasets):
        """Compute the kernel mean embedding for each dataset."""
        embeddings = []
        for D in datasets:
            K = self.rbf_kernel(D, D, self.gamma_x)
            embeddings.append(K.mean(axis=0))
        return np.array(embeddings)

    def distribution_kernel(self, embeddings1, embeddings2):
        """Compute the RBF kernel between distribution embeddings."""
        return self.rbf_kernel(embeddings1, embeddings2, self.gamma_d)
    
    def qp(self, P, q, A, b, C):
        """Solve the quadratic programming problem."""
        n = P.shape[0]
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        G = cvxopt.matrix(np.concatenate([
            -np.eye(n),
            np.eye(n)
        ]))
        h = cvxopt.matrix(np.concatenate([
            np.zeros(n),
            C * np.ones(n)
        ]))

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])
    
    def fit(self, datasets):
        """Fit the OCSMM model."""
        # Compute mean embeddings for all datasets
        embeddings = self.compute_mean_embedding(datasets)

        # Compute the distribution-level kernel matrix
        K = self.distribution_kernel(embeddings, embeddings)

        # Solve the quadratic programming problem
        n = len(K)
        P = K
        q = np.zeros(n)
        A = np.ones((1, n))
        b = np.array([1.0])
        C = 1.0 / (self.nu * n)

        self.alpha = self.qp(P, q, A, b, C)
        self.idx_support = np.where(self.alpha > 1e-5)[0]
        self.alpha_support = self.alpha[self.idx_support]
        self.embeddings_support = embeddings[self.idx_support]

        # Compute rho
        support_kernel = K[self.idx_support][:, self.idx_support]
        self.rho = np.min(support_kernel @ self.alpha_support)
        return self.alpha, self.rho

    def decision_function(self, datasets):
        """Compute the decision function for new datasets."""
        embeddings = self.compute_mean_embedding(datasets)
        K = self.distribution_kernel(embeddings, self.embeddings_support)
        return K @ self.alpha_support - self.rho

@dataclass()
class OneClassSMMClassifier:
    datasets: list[FloatTensor]
    nu: float
    gamma_x: float
    gamma_d: float
    model: Optional[OneClassSMMModel] = field(init=False, default=None)

    def __post_init__(self):
        self.model = OneClassSMMModel(nu=self.nu, gamma_x=self.gamma_x, gamma_d=self.gamma_d)

    def fit(self):
        numpy_datasets = [d.numpy() for d in self.datasets]
        return self.model.fit(numpy_datasets)

    def decision_function(self, new_datasets):
        numpy_datasets = [d.numpy() for d in new_datasets]
        return self.model.decision_function(numpy_datasets)

    def plot_decision_boundary(self, datasets, x1, x2, y1, y2):
        """Plot the decision boundary for 2D datasets."""
        embeddings = self.model.compute_mean_embedding([d.numpy() for d in datasets])
        X1, X2 = np.mgrid[x1:x2+0.1:0.2, y1:y2+0.1:0.2]
        X_test = np.c_[X1.ravel(), X2.ravel()]
        K = self.model.distribution_kernel(X_test, self.model.embeddings_support)
        decision = K @ self.model.alpha_support - self.model.rho

        # Reshape decision function for plotting
        Z = decision.reshape(X1.shape)
        plt.contourf(X1, X2, Z, 20, cmap=plt.cm.gray)
        plt.colorbar()
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c='blue', edgecolors='k')
        plt.xlabel("Embedding 1")
        plt.ylabel("Embedding 2")
        plt.title("OCSMM Decision Boundary")
        plt.show()
