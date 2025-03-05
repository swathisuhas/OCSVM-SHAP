import numpy as np
import cvxopt 
from torch import FloatTensor
from scipy import linalg
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.distance import pdist, squareform
import torch
from src.ocsvm.OneClassSVMClassifier import ocsvm_solver, compute_rho

class OneClassSMMModel:
    def __init__(self, nu, gamma_x, gamma_d):
        self.nu = nu
        self.gamma_x = gamma_x  # Gamma for the instance-level kernel
        self.gamma_d = gamma_d  # Gamma for the distribution-level kernel
        self.alpha_support = None
        self.rho = None
        self.idx_support = None
        self.decision = None

    def compute_kappa_matrix(self, datasets_1, datasets_2):
        num_sets_1, num_sets_2 = len(datasets_1), len(datasets_2)
        kappa_matrix = np.zeros((num_sets_1, num_sets_2))
        for i, group_1 in enumerate(datasets_1):
            for j, group_2 in enumerate(datasets_2):
                mmd2 = compute_mmd_squared(group_1, group_2, self.gamma_x)
                kappa_matrix[i, j] = np.exp(-self.gamma_d * mmd2)
        return kappa_matrix
    
    def fit(self, datasets):
        kappa = self.compute_kappa_matrix(datasets, datasets) # correct
        self.alpha_support, self.idx_support = ocsvm_solver(kappa, self.nu)
        #print(kappa)
        print(self.alpha_support)
        print(self.idx_support)
        self.rho = compute_rho(kappa, self.alpha_support, self.idx_support)
        print(self.rho)
        datasets_support = [datasets[i] for i in self.idx_support] 
        G = self.compute_kappa_matrix(datasets, datasets_support) # simillarity between all groups and the support groups
        self.decision = G.dot(self.alpha_support) - self.rho
        print(self.decision+self.rho)
        # self.decision = (self.decision - np.mean(self.decision)) / np.std(self.decision)
        return self.decision, np.sign(self.decision)  # decision is wrong
    
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
        print(self.gamma_x)
        print(self.gamma_d)
        self.model = OneClassSMMModel(nu=self.nu, gamma_x=self.gamma_x, gamma_d=self.gamma_d)

    def fit(self):
        return self.model.fit(self.datasets)
    
    def find_best_gamma_x(self):
        gamma_x_values = []
        for group in self.datasets: 
            pairwise_sq_dists = squareform(pdist(group, 'sqeuclidean'))  # Squared Euclidean distances
            median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  # Ignore zero distances
            gamma_x_values.append( 1 / (median_dist))
        return np.median(gamma_x_values) 
    
    def find_best_gamma_d(self):
        group_distances = []
        for i in range(len(self.datasets)):
            for j in range(i + 1, len(self.datasets)):
                dist = compute_mmd_squared(self.datasets[i], self.datasets[j], self.gamma_x)
                group_distances.append(dist)
        return  1 / (np.median(group_distances))
    
def compute_mmd_squared(X, Y, gamma):
    """MMD^2 = K_XX + K_YY - 2*K_XY"""
    n, m = len(X), len(Y)
    K_XX = rbf_kernel(X, X, gamma) 
    K_YY = rbf_kernel(Y, Y, gamma) 
    K_XY = rbf_kernel(X, Y, gamma)  

    mmd2 = (K_XX / (n * n)) + (K_YY / (m * m)) - (2 * K_XY / (n * m))
    return mmd2 # is a real value 
    

def rbf_kernel(X1, X2, gamma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = 0
        for i in range(n1):
            for j in range(n2):
                K = K + rbf_metric(X1[i], X2[j], gamma)
        return K 

def rbf_metric(x, y, gamma):
    return np.exp(-gamma * linalg.norm(x - y, 2)**2)