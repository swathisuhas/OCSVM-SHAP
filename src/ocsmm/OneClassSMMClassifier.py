import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from torch import FloatTensor
from src.ocsvm.OneClassSVMClassifier import OneClassSVMClassifier

from dataclasses import dataclass, field
from typing import Optional

class OneClassSMMModel:
    def __init__(self, nu, gamma_x, gamma_d):
        self.nu = nu
        self.gamma_x = gamma_x  # Gamma for the instance-level kernel
        self.gamma_d = gamma_d  # Gamma for the distribution-level kernel
        self.ocsvm = None  # One-Class SVM model
        self.embeddings = None

    def rbf_kernel(self, X1, X2, gamma):
        """Compute the RBF kernel."""
        pairwise_dists = pairwise_distances(X1, X2, metric='sqeuclidean')
        return np.exp(-gamma * pairwise_dists)

    def compute_mean_embedding(self, datasets):
        """Compute Kernel Mean Embedding (KME) for each dataset."""
        embeddings = []
        for D in datasets:
            K = self.rbf_kernel(D, D, self.gamma_x)  # Compute instance-level RBF kernel
            embeddings.append(K.mean(axis=0))  # Compute mean embedding
        return np.array(embeddings)

    def fit(self, datasets):
        """Fit the One-Class SMM model using OCSVM on embeddings."""
        # Compute KME for all datasets
        self.embeddings = self.compute_mean_embedding(datasets)

        # Train One-Class SVM on the embeddings
        self.ocsvm = OneClassSVMClassifier(
            X=FloatTensor(self.embeddings),
            nu=self.nu,
            gamma=self.gamma_d
        )
        return self.ocsvm.fit() 

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
    