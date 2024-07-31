from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from joblib import Parallel, delayed
from torch import FloatTensor, BoolTensor, Tensor
from tqdm import tqdm

from src.ocsvm.OneClassSVMClassifier import OneClassSVMClassifier
from src.utils.shapley_procedure.preparing_weights_and_coalitions import compute_weights_and_coalitions

@dataclass(kw_only=True)
class OCSVMSHAP(object):
    """Run the SHAP algorithm to explain the output of a OneClassSVMClassifier model."""
    X: FloatTensor
    classifier: OneClassSVMClassifier
    scale: FloatTensor = field(init=False, default=None)
    inducing_points = None
    
    mean_stochastic_value_function_evaluations: Tensor = field(init=False)
    conditional_mean_projections: FloatTensor | Tensor = field(init=False)
    coalitions: BoolTensor = field(init=False)
    weights: FloatTensor = field(init=False)
    
    cme_regularisation: FloatTensor = field(init=False, default=torch.tensor(1e-4).float())
    num_cpus: int = field(init=False, default=6)
    
    def __post_init__(self):
        # self.kernel_lengthscales = 1.0 / self.classifier.gamma
        # self.classifier.fit()
        self.rho = self.classifier.model.rho
        self.mu_support = self.classifier.model.mu_support
        self.idx_support = self.classifier.model.idx_support
        self.support_vectors = self.X[self.idx_support]
        self.decision = self.classifier.decision()[0]
        self.inducing_points = self.classifier.inducing_points
    
    def fit_ocsvmshap(self, X: FloatTensor, num_coalitions: int) -> None:
        # X = self._scaled_by_lengthscales(X)
        self.weights, self.coalitions = compute_weights_and_coalitions(num_features=X.shape[1], num_coalitions=num_coalitions)
        self.conditional_mean_projections = self._compute_conditional_mean_projections(X)
        self.mean_stochastic_value_function_evaluations = torch.cat([
            torch.ones((1, X.shape[0])) * self.decision.mean(),
            torch.einsum(
                'ijk,j->ik', self.conditional_mean_projections, torch.tensor(self.decision).float()
            )
        ])
    
    def return_deterministic_shapley_values(self) -> FloatTensor:
        return _solve_weighted_least_square_regression(SHAP_weights=self.weights,
                                                            coalitions=self.coalitions,
                                                            regression_target=self.mean_stochastic_value_function_evaluations
                                                            ) # * self.scale  
    # there is no need to stnadardize the data in ocsvm -> there is no target variable
    
    def _compute_conditional_mean_projections(self, X):
        minus_first_coalitions = self.coalitions[1:]  # remove the first row of 0s.
        compute_conditional_mean_projections = lambda S: self._compute_conditional_mean_projection(S.bool(), X)
        return torch.stack(
            Parallel(n_jobs=self.num_cpus)(
                delayed(compute_conditional_mean_projections)(S.bool())
                for S in tqdm(minus_first_coalitions)
            )
        )
    
    def _compute_value_function_at_coalition(self, S: BoolTensor, X: FloatTensor):
        if S.sum() == 0:  # no active feature
            return (torch.ones((1, X.shape[0])) * torch.tensor(self.rho)).squeeze()
        
        conditional_mean_projection = self._compute_conditional_mean_projection(S, X)
        return conditional_mean_projection.T @ torch.tensor(self.mu_support)
    
    # def _compute_conditional_mean_projection(self, S: BoolTensor, X: FloatTensor):
    #     k_inducingXS_XS = self._rbf_kernel(self.support_vectors[:, S], X[:, S])
    #     K_SS = self._rbf_kernel(self.support_vectors[:, S])
    #     return np.linalg.solve(K_SS + self.classifier.nu * np.eye(K_SS.shape[0]), k_inducingXS_XS.T)
    
    # def _compute_conditional_mean_projection(self, S: BoolTensor, X: FloatTensor):
    #     """ compute the expression k_S(x, X)(K_SS + lambda I)^{-1} that can be reused multiple times
    #     """
    #     k_inducingXS_XS = self.classifier.model.rbf_kernel(self.inducing_points[:, S], X[:, S])
    #     return (torch.from_numpy(self.classifier.model.rbf_kernel(self.inducing_points[:, S]))).add_diag(
    #         self.model.num_inducing_points * self.cme_regularisation).inv_matmul(
    #         k_inducingXS_XS.evaluate()).detach()
    

    def _compute_conditional_mean_projection(self, S: BoolTensor, X: FloatTensor):
        k_inducingXS_XS = self.classifier.model.rbf_kernel(self.inducing_points[:, S], X[:, S])
        # Compute K_SS
        K_SS = self.classifier.model.rbf_kernel(self.inducing_points[:, S], self.inducing_points[:, S])
        # Add diagonal regularization term
        regularization_term = self.classifier.num_inducing_points * self.cme_regularisation
        K_SS_regularized = np.add(K_SS, regularization_term * np.eye(K_SS.shape[0]))
        # Convert to torch.Tensor
        K_SS_regularized = K_SS_regularized.float()
        k_inducingXS_XS = torch.from_numpy(k_inducingXS_XS).float()
        # Compute the inverse of the regularized K_SS
        K_SS_inv = torch.inverse(K_SS_regularized)
        # Perform matrix multiplication
        conditional_mean_projection = K_SS_inv.matmul(k_inducingXS_XS)
        return conditional_mean_projection.detach()


    # def _rbf_kernel(self, X1, X2=None):
    #     if X2 is None:
    #         X2 = X1
    #     return pairwise_distances(X1, X2, metric=lambda x, y: np.exp(-self.classifier.gamma * np.linalg.norm(x - y)**2))
    
   
    # def _scaled_by_lengthscales(self, X: torch.FloatTensor) -> FloatTensor:
    #     return X / self.kernel_lengthscales

def _solve_weighted_least_square_regression(SHAP_weights: FloatTensor,
                                            coalitions: BoolTensor,
                                            regression_target: FloatTensor | Tensor,
                                            ) -> FloatTensor:
    weighted_regression_target = regression_target * SHAP_weights
    ZtWvx = coalitions.t() @ weighted_regression_target
    L = torch.linalg.cholesky(coalitions.t() @ (coalitions * SHAP_weights))

    return torch.cholesky_solve(ZtWvx, L).detach()
