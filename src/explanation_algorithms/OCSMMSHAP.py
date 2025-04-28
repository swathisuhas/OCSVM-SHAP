from dataclasses import dataclass, field
import torch
from joblib import Parallel, delayed
from torch import FloatTensor, BoolTensor, Tensor
from tqdm import tqdm
from typing import List
import warnings

from src.ocsmm.OneClassSMMClassifier import OneClassSMMClassifier
from src.utils.shapley_procedure.preparing_weights_and_coalitions import compute_weights_and_coalitions

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


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
    num_cpus: int = field(init=True, default=150)

    def fit_ocsmmshap(self, X: List[FloatTensor], num_coalitions: int) -> None:
        num_groups = len(X)
        num_features = X[0].shape[1]
        self.weights, self.coalitions = compute_weights_and_coalitions(num_features=num_features, num_coalitions=num_coalitions)

        minus_first_coalitions = self.coalitions[1:]
        decision_tensor = torch.tensor(self.decision).float()

        def compute_value(S):
            proj = self._compute_conditional_mean_projection(S.bool(), X)
            return torch.matmul(proj, decision_tensor)

        value_function_evals = Parallel(n_jobs=self.num_cpus)( 
            delayed(compute_value)(S) for S in tqdm(minus_first_coalitions, desc="Calculating projections")
        )

        # Clear intermediate variables after use
        del decision_tensor
        del minus_first_coalitions

        empty_value = torch.ones((num_groups,)) * self.decision.mean()
        value_function_evals.insert(0, empty_value)

        self.mean_stochastic_value_function_evaluations = torch.stack(value_function_evals)

        # Clear intermediate results from memory
        del value_function_evals

       

    def return_deterministic_shapley_values(self) -> FloatTensor:
        return _solve_weighted_least_square_regression(SHAP_weights=self.weights,
                                                            coalitions=self.coalitions,
                                                            regression_target=self.mean_stochastic_value_function_evaluations
                                                            )
    
    def _compute_conditional_mean_projections(self, X):
        minus_first_coalitions = self.coalitions[1:]  # remove the first row of 0s.
        projections = []

        for S in tqdm(minus_first_coalitions, desc="Computing conditional mean projections"):
            proj = self._compute_conditional_mean_projection(S.bool(), X)
            projections.append(proj)

        return projections


    def _compute_conditional_mean_projection(self, S: BoolTensor, X: List[FloatTensor]):
        X_filtered = [group[:, S] for group in X] 
        K_SS = self.classifier.kappa_matrix(X_filtered, X_filtered)
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