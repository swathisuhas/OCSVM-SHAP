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
        self.embeddings = None
        self.K = None
        self.C = None

    def rbf_kernel(self, X1, X2, gamma):
        """Compute the RBF kernel."""
        pairwise_dists = pairwise_distances(X1, X2, metric='sqeuclidean')
        return np.exp(-gamma * pairwise_dists)

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
        G = cvxopt.matrix(np.concatenate(
            [np.diag(np.ones(n) * -1), np.diag(np.ones(n))], axis=0))
        h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))


        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.ravel(solution['x'])
    
    def fit(self, datasets):
        """Fit the OCSMM model."""
        # Compute mean embeddings for all datasets
        self.embeddings = self.compute_mean_embedding(datasets)

        # Compute the distribution-level kernel matrix
        self.K = self.distribution_kernel(self.embeddings, self.embeddings)

        # Solve the quadratic programming problem
        n = len(self.K)
        P = self.K
        q = np.zeros(n)
        A = np.matrix(np.ones(n))
        b = 1.
        C = 1. / (self.nu * n)

        self.alpha = self.qp(P, q, A, b, C)
        self.idx_support = np.where(self.alpha > 1e-7)[0]
        self.alpha_support = self.alpha[self.idx_support] * self.nu * len(self.K) # maybe multiply with nu* len(K)
        self.embeddings_support = self.embeddings[self.idx_support]

        # Compute rho
        index = int(np.argmin(self.alpha_support))
        support_kernel = self.K[self.idx_support][:, self.idx_support]
        self.rho = self.alpha_support.dot(support_kernel[index])
        # self.rho = np.min(support_kernel @ self.alpha_support)
        return self.alpha_support, self.rho

    def decision_function(self, datasets):
        """Compute the decision function for new datasets."""
        embeddings = self.compute_mean_embedding(datasets)
        G = self.distribution_kernel(self.embeddings, self.embeddings_support)
        return G.dot(self.alpha_support) - self.rho


@dataclass()
class OneClassSMMClassifier:
    datasets: list[FloatTensor]
    nu: float
    gamma_x: float
    gamma_d: float
    model: Optional[OneClassSMMModel] = field(init=False, default=None)
    num_inducing_points: int = field(default=None)

    def __post_init__(self):
        self.model = OneClassSMMModel(nu=self.nu, gamma_x=self.gamma_x, gamma_d=self.gamma_d)

    def fit(self):
        numpy_datasets = [d.numpy() for d in self.datasets]
        return self.model.fit(numpy_datasets)

    def decision_function(self, new_datasets):
        numpy_datasets = [d.numpy() for d in new_datasets]
        decision = self.model.decision_function(numpy_datasets)
        return decision, np.sign(decision)
    