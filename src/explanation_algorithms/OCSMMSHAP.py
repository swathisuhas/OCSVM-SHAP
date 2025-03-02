from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from joblib import Parallel, delayed
from torch import FloatTensor, BoolTensor, Tensor
from tqdm import tqdm
from typing import List

from src.ocsmm.OneClassSMMClassifier import OneClassSMMClassifier
from src.utils.shapley_procedure.preparing_weights_and_coalitions import compute_weights_and_coalitions
from src.utils.kernels.inducing_points import compute_inducing_points


@dataclass(kw_only=True)
class OCSMMSHAP(object):
    """Run the SHAP algorithm to explain the output of a OneClassSMMClassifier model."""
    X: List[FloatTensor]  # Each element is a dataset representing a group
    classifier: OneClassSMMClassifier
    inducing_points: List[FloatTensor] = field(init=False)

    mean_stochastic_value_function_evaluations: Tensor = field(init=False)
    conditional_mean_projections: FloatTensor | Tensor = field(init=False)
    coalitions: BoolTensor = field(init=False)
    weights: FloatTensor = field(init=False)

    cme_regularisation: FloatTensor = field(init=False, default=torch.tensor(1e-4).float())
    num_cpus: int = field(init=False, default=6)
    
    def __post_init__(self):
        self.classifier.fit()
        self.rho = self.classifier.model.rho
        self.mu_support = self.classifier.model.alpha_support
        self.idx_support = self.classifier.model.idx_support
        self.support_vectors = [self.X[i] for i in self.idx_support]
        self.decision = self.classifier.model.decision
        self.inducing_points = self.X 

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
        minus_first_coalitions = self.coalitions[1:]  

        def compute_projection(S):
            """Compute projections for all groups given a coalition mask S."""
            return self._compute_conditional_mean_projection(S.bool(), X)
        projections_list = Parallel(n_jobs=self.num_cpus)(
            delayed(compute_projection)(S) for S in tqdm(minus_first_coalitions)
        )
        return torch.stack(projections_list)

    def _compute_conditional_mean_projection(self, S: BoolTensor, X: List[FloatTensor]):
        """ compute the expression k_S(x, X)(K_SS + lambda I)^{-1} that can be reused multiple times
        """
        S = S.bool()  

        X_filtered = [group[:, S] for group in X] 
        inducing_filtered = [self.inducing_points[i][:, S] for i in range(len(self.inducing_points))]
        k_inducingXS_XS = self.classifier.model.compute_kappa_matrix(inducing_filtered, X_filtered)
        K_SS = self.classifier.model.compute_kappa_matrix(inducing_filtered, inducing_filtered)

        k_inducingXS_XS = torch.tensor(k_inducingXS_XS).float()
        K_SS = torch.tensor(K_SS).float()

        regularization_term = self.classifier.num_inducing_points * self.cme_regularisation
        K_SS_regularized = K_SS + regularization_term * torch.eye(K_SS.shape[0]).float()

        K_SS_inv = torch.inverse(K_SS_regularized)
        conditional_mean_projection = K_SS_inv.matmul(k_inducingXS_XS)
        return conditional_mean_projection.detach()


def _solve_weighted_least_square_regression(SHAP_weights: FloatTensor,
                                            coalitions: BoolTensor,
                                            regression_target: FloatTensor | Tensor,
                                            ) -> FloatTensor:
    weighted_regression_target = regression_target * SHAP_weights
    ZtWvx = coalitions.t() @ weighted_regression_target
    L = torch.linalg.cholesky(coalitions.t() @ (coalitions * SHAP_weights))

    return torch.cholesky_solve(ZtWvx, L).detach()
