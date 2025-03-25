from dataclasses import dataclass, field
import torch
from joblib import Parallel, delayed
from torch import FloatTensor, BoolTensor, Tensor
from tqdm import tqdm
from typing import List

from src.ocsmm.OneClassSMMClassifier import OneClassSMMClassifier
from src.utils.shapley_procedure.preparing_weights_and_coalitions import compute_weights_and_coalitions

@dataclass(kw_only=True)
class OCSMMSHAP(object):
    """Run the SHAP algorithm to explain the output of a OneClassSMMClassifier model."""
    X: List[FloatTensor]  
    classifier: OneClassSMMClassifier
    decision: FloatTensor

    mean_stochastic_value_function_evaluations: Tensor = field(init=False)
    conditional_mean_projections: FloatTensor | Tensor = field(init=False)
    coalitions: BoolTensor = field(init=False)
    weights: FloatTensor = field(init=False)

    cme_regularisation: FloatTensor = field(init=False, default=torch.tensor(1e-4).float())
    num_cpus: int = field(init=False, default=6)

    def fit_ocsmmshap(self, X: List[FloatTensor], num_coalitions: int) -> None:
        num_groups = len(X) 
        num_features = X[0].shape[1]
        self.weights, self.coalitions = compute_weights_and_coalitions(num_features=num_features, num_coalitions=num_coalitions)
        self.conditional_mean_projections = self._compute_conditional_mean_projections(X)
        decision_tensor = torch.tensor(self.decision, dtype=torch.float32) 
        first_term = decision_tensor.mean().expand(1, num_groups)
        second_term = torch.einsum(
            'ijk,j->ik', 
            self.conditional_mean_projections,
            decision_tensor
        ) 
        self.mean_stochastic_value_function_evaluations = torch.cat([first_term, second_term])
 

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

    def _compute_conditional_mean_projection(self, S: BoolTensor, X: List[FloatTensor]):
        X_filtered = [group[:, S] for group in X] 
        K_SS = self.classifier.kappa_matrix(X_filtered, X_filtered, self.classifier.gamma)
        K_SS = torch.tensor(K_SS).float()
        regularization_term = len(self.X) * self.cme_regularisation
        K_SS_regularized = K_SS + regularization_term * torch.eye(K_SS.shape[0]).float()
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