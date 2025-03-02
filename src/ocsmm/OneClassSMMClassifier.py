import numpy as np
import cvxopt 
from torch import FloatTensor
from scipy import linalg
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.distance import pdist, squareform
import torch

class OneClassSMMModel:
    def __init__(self, nu, gamma_x, gamma_d):
        self.nu = nu
        self.gamma_x = gamma_x  # Gamma for the instance-level kernel
        self.gamma_d = gamma_d  # Gamma for the distribution-level kernel
        self.alpha_support = None
        self.rho = None
        self.idx_support = None
        self.decision = None
    
    def rbf_kernel(self, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = 0
        for i in range(n1):
            for j in range(n2):
                K = K + self._rbf_metric(X1[i], X2[j])
        return K
    
    def _rbf_metric(self, x, y):
        return np.exp(-self.gamma_x * linalg.norm(x - y, 2)**2)
    
    def compute_mmd_squared(self, X, Y):
        """MMD^2 = K_XX + K_YY - 2*K_XY"""
        n, m = len(X), len(Y)
        K_XX = self.rbf_kernel(X, X) 
        K_YY = self.rbf_kernel(Y, Y) 
        K_XY = self.rbf_kernel(X, Y)  

        mmd2 = (K_XX / (n * n)) + (K_YY / (m * m)) - (2 * K_XY / (n * m))
        return mmd2 # is a real value 

    def compute_kappa_matrix(self, datasets_1, datasets_2):
        """Compute the Kappa value which is exp(-gamma_d(mu_p - mu_q)^2)"""
        num_sets_1, num_sets_2 = len(datasets_1), len(datasets_2)
        kappa_matrix = np.zeros((num_sets_1, num_sets_2))

        for i, group_1 in enumerate(datasets_1):
            for j, group_2 in enumerate(datasets_2):
                mmd2 = self.compute_mmd_squared(group_1, group_2)
                kappa_matrix[i, j] = np.exp(-self.gamma_d * mmd2)
        return kappa_matrix
        
    
    def qp(self, P, q, A, b, C):   
    
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
    
    def ocsvm_solver(self, kappa):  
        n = len(kappa)
        P = kappa
        q = np.zeros(n)
        A = np.matrix(np.ones(n))
        b = 1.
        C = 1. / (self.nu * n)
        alpha = self.qp(P, q, A, b, C)
        self.idx_support = np.where(np.abs(alpha) > 1e-5)[0] 
        self.alpha_support = alpha[self.idx_support] #* self.nu * n  # multipling with nu * len(K) to match values from sklearn
        return self.alpha_support, self.idx_support
    
    def compute_rho(self, kappa):
        index = int(np.argmin(self.alpha_support))
        Kappa_support = kappa[self.idx_support][:, self.idx_support] 
        rho = self.alpha_support.dot(Kappa_support[index])
        return rho
    
    def fit(self, datasets):
        kappa = self.compute_kappa_matrix(datasets, datasets)
        self.alpha_support, self.idx_support = self.ocsvm_solver(kappa)
        self.rho = self.compute_rho(kappa)
        print(self.idx_support)
        print(self.alpha_support)
        print(self.rho)
        datasets_support = datasets[self.idx_support] #[datasets[i] for i in self.idx_support]
        G = self.compute_kappa_matrix(datasets, datasets_support)
        self.decision = G.dot(self.alpha_support) - self.rho
        return self.decision, np.sign(self.decision)
    
@dataclass()
class OneClassSMMClassifier:
    datasets: list[FloatTensor]
    nu: FloatTensor
    gamma_x: FloatTensor = field(init=False, default=torch.tensor(0.1).float())
    gamma_d: FloatTensor = field(init=False, default=torch.tensor(0.1).float())
    model: Optional[OneClassSMMModel] = field(init=False, default=None)

    def __post_init__(self):
        self.gamma_x = self.find_best_gamma_x()
        self.gamma_d = self.find_best_gamma_d()
        self.model = OneClassSMMModel(nu=self.nu, gamma_x=self.gamma_x, gamma_d=self.gamma_d)

    def fit(self):
        return self.model.fit(self.datasets)
    
    def find_best_gamma_x(self):
        gamma_x_values = []
        for group in self.datasets: 
            pairwise_sq_dists = squareform(pdist(group, 'sqeuclidean'))  # Squared Euclidean distances
            median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  # Ignore zero distances
            gamma_x_values.append( 1 / (2 * median_dist))
        return np.median(gamma_x_values) 
    
    def find_best_gamma_d(self):
        group_distances = []
        for i in range(len(self.datasets)):
            for j in range(i + 1, len(self.datasets)):
                dist = self.mmd_distance(self.datasets[i], self.datasets[j], self.gamma_x)
                group_distances.append(dist)
    

    def mmd_distance(S1, S2, gamma):
        """Compute Maximum Mean Discrepancy (MMD) between two groups S1 and S2."""
        K_xx = np.mean(np.exp(-gamma * squareform(pdist(S1, 'sqeuclidean'))))  # Self-similarity S1
        K_yy = np.mean(np.exp(-gamma * squareform(pdist(S2, 'sqeuclidean'))))  # Self-similarity S2
        K_xy = np.mean(np.exp(-gamma * np.linalg.norm(S1[:, None] - S2, axis=2)**2))  # Between S1 and S2
        return K_xx + K_yy - 2 * K_xy  # MMD formula