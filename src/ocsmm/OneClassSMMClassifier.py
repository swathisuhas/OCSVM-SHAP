import numpy as np
import cvxopt 
from scipy.spatial.distance import cdist
from torch import FloatTensor
from dataclasses import dataclass, field
from typing import Optional

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
        """Compute the RBF kernel."""
        pairwise_sq_dists = cdist(X1, X2, 'sqeuclidean')  # Compute ||xi - xj||²
        return np.exp(-self.gamma_x * pairwise_sq_dists)
    
    def compute_mmd_squared(self, X, Y):
        n, m = len(X), len(Y)
        K_XX = self.rbf_kernel(X, X)  # k(xi, xj)
        K_YY = self.rbf_kernel(Y, Y)  # k(x'i, x'j)
        K_XY = self.rbf_kernel(X, Y)  # k(xi, x'j)
        K_YX = self.rbf_kernel(Y, X)  # k(x'i, xj) (same as K_XY.T)

        mmd2 = (K_XX.sum() / (n * n)) + (K_YY.sum() / (m * m)) - (2 * K_XY.sum() / (n * m))
        return mmd2
    

    def compute_kappa_matrix(self, datasets_1, datasets_2):
        """Compute the Kappa value which is exp(-gamma_d(mu_p - mu_q)^2)"""
        num_sets_1, num_sets_2 = len(datasets_1), len(datasets_2)
        kappa_matrix = np.zeros((num_sets_1, num_sets_2))

        for i, data_1 in enumerate(datasets_1):
            for j, data_2 in enumerate(datasets_2):
                mmd2 = self.compute_mmd_squared(data_1, data_2)
                kappa_matrix[i, j] = np.exp(-self.gamma_d * mmd2)
        return kappa_matrix
        
    
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
    
    def ocsvm_solver(self, kappa):  # nu default is 0.1
        n = len(kappa)
        P = kappa
        q = np.zeros(n)
        A = np.matrix(np.ones(n))
        b = 1.
        C = 1. / (self.nu * n)
        alpha = self.qp(P, q, A, b, C)
        self.idx_support = np.where(np.abs(alpha) > 1e-5)[0] # if alpha is greater than 1e-5 then it is considered a support vector
        self.alpha_support = alpha[self.idx_support] * self.nu * n  # multipling with nu * len(K) to match values from sklearn
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
        datasets_support = [datasets[i] for i in self.idx_support]
        G = self.compute_kappa_matrix(datasets, datasets_support)
        self.decision = G.dot(self.alpha_support) - self.rho
        return self.decision, np.sign(self.decision)

    
@dataclass()
class OneClassSMMClassifier:
    datasets: list[FloatTensor]
    nu: float
    gamma_x: float
    gamma_d: float
    model: Optional[OneClassSMMModel] = field(init=False, default=None)
    num_inducing_points: int = field(default=None)

    def __post_init__(self):
        self.num_inducing_points = len(self.datasets)
        self.model = OneClassSMMModel(nu=self.nu, gamma_x=self.gamma_x, gamma_d=self.gamma_d)

    def fit(self):
        return self.model.fit(self.datasets)