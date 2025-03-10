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
    mean_stochastic_value_function_evaluations: Tensor = field(init=False)
    conditional_mean_projections: FloatTensor | Tensor = field(init=False)
    coalitions: BoolTensor = field(init=False)
    weights: FloatTensor = field(init=False)
    cme_regularisation: FloatTensor = field(init=False, default=torch.tensor(1e-4).float())
    num_cpus: int = field(init=False, default=6)
    
    def __post_init__(self):
        self.decision = self.classifier.model.decision
    
    def fit_ocsvmshap(self, X: FloatTensor, num_coalitions: int) -> None:
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
                                                            )
    
    def _compute_conditional_mean_projections(self, X):
        minus_first_coalitions = self.coalitions[1:]  # remove the first row of 0s.
        compute_conditional_mean_projections = lambda S: self._compute_conditional_mean_projection(S.bool(), X)
        return torch.stack(
            Parallel(n_jobs=self.num_cpus)(
                delayed(compute_conditional_mean_projections)(S.bool())
                for S in tqdm(minus_first_coalitions)
            )
        )

    def _compute_conditional_mean_projection(self, S: BoolTensor, X: FloatTensor):
        """ compute the expression k_S(x, X)(K_SS + lambda I)^{-1} that can be reused multiple times
        """
        K_SS = self.classifier.model.rbf_kernel(self.X[:, S], self.X[:, S])
        regularization_term = len(X) * self.cme_regularisation
        K_SS_regularized = np.add(K_SS, regularization_term * np.eye(K_SS.shape[0]))
        K_SS_regularized = K_SS_regularized.float()
        K_SS = torch.from_numpy(K_SS).float()
        K_SS_inv = torch.inverse(K_SS_regularized)
        conditional_mean_projection = K_SS_inv.matmul(K_SS)
        return conditional_mean_projection.detach()


def _solve_weighted_least_square_regression(SHAP_weights: FloatTensor,
                                            coalitions: BoolTensor,
                                            regression_target: FloatTensor | Tensor,
                                            ) -> FloatTensor:
    weighted_regression_target = regression_target * SHAP_weights
    ZtWvx = coalitions.t() @ weighted_regression_target
    L = torch.linalg.cholesky(coalitions.t() @ (coalitions * SHAP_weights))

    return torch.cholesky_solve(ZtWvx, L).detach()
