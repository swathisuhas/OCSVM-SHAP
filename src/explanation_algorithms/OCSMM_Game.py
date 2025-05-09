import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Callable
from shapiq.games.base import Game
from src.ocsmm.new_OCSMMClassifier import OneClassSMMClassifier

class OCSMM_Game_for_SHAPIQ(Game): # Inherit from shapiq.games.Game
    """
    Game class for explaining OCSMM scores (h = K.alpha) using KernelSHAPIQ,
    compatible with the shapiq.games.Game interface.
    The value function v(P_k, S) is based on the formula:
    alpha^T @ ( (kappa_S_norm(P,P_k) broadcast_mul K_SbarSbar_norm) @ 
                (K_SS_norm + lambda I)^-1 @ kappa_S_norm(P,P_k) )
    """
    def __init__(self,
                 ocsmm_classifier: "OneClassSMMClassifier",
                 group_P_to_explain_data: np.ndarray, # Raw data for P_k
                 lambda_reg_in_formula: float = 1e-4,
                 # shapiq.games.Game specific args
                 normalize_by_base_game: bool = True, # If True, base __call__ subtracts normalization_value
                 game_normalization_value: float = 0.0, # v(empty) before normalization
                 verbose_base_game: bool = False,
                 **kwargs # To catch any other args for Game base
                ):
        
        self.classifier = ocsmm_classifier
        _n_players = self.classifier.datasets[0].shape[1]

        # Initialize the shapiq.Game base class
        # Pass game_normalization_value as the v(empty) that value_function will produce.
        # If value_function already produces v(empty)=0 and you want that, set game_normalization_value=0.
        super().__init__(n_players=_n_players, 
                         normalize=normalize_by_base_game, 
                         normalization_value=game_normalization_value,
                         verbose=verbose_base_game,
                         **kwargs)
        
        # Device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.classifier.alpha, 'device') and isinstance(self.classifier.alpha, torch.Tensor):
            self.device = self.classifier.alpha.device # If alpha is already a tensor
        
        if isinstance(group_P_to_explain_data, np.ndarray):
            self.P_k_explain_data_tensor = torch.from_numpy(group_P_to_explain_data).float().to(self.device)
        else: 
            self.P_k_explain_data_tensor = group_P_to_explain_data.float().to(self.device)
            
        self.lambda_formula = lambda_reg_in_formula
        
        self.P_train_data_tensors = [
            torch.from_numpy(d).float().to(self.device) for d in self.classifier.datasets
        ]
        self.m_train_groups = len(self.P_train_data_tensors)
        
        if isinstance(self.classifier.alpha, np.ndarray):
            self.alpha_ocsmm = torch.from_numpy(self.classifier.alpha).float().to(self.device)
        else: 
            self.alpha_ocsmm = self.classifier.alpha.float().to(self.device)
            
        self.gamma_rbf = self.classifier.gamma
        
        # If you set normalize=True and game_normalization_value=0.0 (the default for v(empty) from your formula),
        # then the base Game.__call__ will do values - 0.0, which is fine.
        # If your v(empty) from value_function is non-zero and you want base Game to center it to 0,
        # you'd pass that non-zero v(empty) as game_normalization_value.
        # Your current value_function returns 0.0 for empty set.
        if self.normalize and self.normalization_value != 0.0:
            print(f"Warning: OCSMM_Game created with normalize=True, but its value_function provides v(empty)=0. "
                  f"Base Game will subtract normalization_value={self.normalization_value}. "
                  f"Ensure this is intended or set game_normalization_value=0.0.")


    def _pointwise_rbf_kernel_S(self, X1_S: torch.Tensor, X2_S: torch.Tensor) -> torch.Tensor:
        # (Same as your version)
        if X1_S.shape[1] == 0 or X2_S.shape[1] == 0:
            return torch.ones((X1_S.shape[0], X2_S.shape[0]), dtype=torch.float64, device=self.device)
        sq_dist_S = torch.cdist(X1_S.double(), X2_S.double(), p=2)**2
        return torch.exp(-self.gamma_rbf * sq_dist_S)

    def _inter_group_kappa_S(self, group_A_tensor: torch.Tensor, group_B_tensor: torch.Tensor,
                               feature_indices_S: np.ndarray) -> float:
        # (Same as your version)
        if len(feature_indices_S) == 0: 
            return 1.0
        P_A_S = group_A_tensor[:, feature_indices_S]
        P_B_S = group_B_tensor[:, feature_indices_S]
        pointwise_k_S_matrix = self._pointwise_rbf_kernel_S(P_A_S, P_B_S)
        return float(pointwise_k_S_matrix.mean().item())

    def _get_gram_matrix_S(self, list_of_group_tensors_P: list[torch.Tensor],
                               feature_indices_S: np.ndarray,
                               Y_sets_P_tensors: list[torch.Tensor] | None = None,
                               apply_normalization: bool = True) -> torch.Tensor:
        # (Same as your version, with minor stability tweak for sqrt)
        X_sets_tensors = list_of_group_tensors_P
        Y_sets_tensors = X_sets_tensors if Y_sets_P_tensors is None else Y_sets_P_tensors
        g1, g2 = len(X_sets_tensors), len(Y_sets_tensors)
        
        K_unnormalized_np = np.empty((g1, g2), dtype=np.float64)
        for i in range(g1):
            for j in range(g2):
                K_unnormalized_np[i, j] = self._inter_group_kappa_S(X_sets_tensors[i], Y_sets_tensors[j], feature_indices_S)
        
        K_unnormalized = torch.from_numpy(K_unnormalized_np).to(dtype=torch.float32, device=self.device)

        if not apply_normalization:
            return K_unnormalized

        diag_K_SS_unnormalized_np = np.array(
            [self._inter_group_kappa_S(X_sets_tensors[i], X_sets_tensors[i], feature_indices_S) for i in range(g1)],
            dtype=np.float64
        )
        diag_K_SS_unnormalized = torch.from_numpy(diag_K_SS_unnormalized_np).to(dtype=torch.float32, device=self.device)
        # Ensure argument to sqrt is non-negative and add epsilon for stability
        norm1 = torch.sqrt(torch.clamp(diag_K_SS_unnormalized, min=0) + 1e-9).unsqueeze(1)
        
        if X_sets_tensors is Y_sets_tensors:
            norm2 = norm1.T
        else:
            diag_K_Y_Y_unnormalized_np = np.array(
                [self._inter_group_kappa_S(Y_sets_tensors[j], Y_sets_tensors[j], feature_indices_S) for j in range(g2)],
                dtype=np.float64
            )
            diag_K_Y_Y_unnormalized = torch.from_numpy(diag_K_Y_Y_unnormalized_np).to(dtype=torch.float32, device=self.device)
            norm2 = torch.sqrt(torch.clamp(diag_K_Y_Y_unnormalized, min=0) + 1e-9).unsqueeze(0)
            
        denominator = norm1 @ norm2
        denominator = denominator + 1e-9 
        K_normalized = K_unnormalized / denominator
        return K_normalized

    # This is the core method required by shapiq.games.Game
    def value_function(self, coalitions_batch_np_bool: np.ndarray) -> np.ndarray:
        """
        Computes the raw game values (before normalization by shapiq.Game base class).
        This function implements the formula for v_hat(P_k, S).
        """
        game_values_list = []
        all_feature_indices = np.arange(self.n_players)
        P_k_explain_list = [self.P_k_explain_data_tensor]

        # Use self.verbose from base class if tqdm is desired
        iterator = coalitions_batch_np_bool
        if self.verbose: # self.verbose is from shapiq.games.Game
            iterator = tqdm(coalitions_batch_np_bool, desc="OCSMM Game Valuation", leave=False)

        for s_bool_vector_np in iterator:
            s_indices = np.where(s_bool_vector_np)[0]
            s_bar_indices = np.setdiff1d(all_feature_indices, s_indices)

            K_SS_train_norm = self._get_gram_matrix_S(self.P_train_data_tensors, s_indices)
            K_SbarSbar_train_norm = self._get_gram_matrix_S(self.P_train_data_tensors, s_bar_indices)
            kappa_mat_S_TrainPkS_norm = self._get_gram_matrix_S(self.P_train_data_tensors, s_indices,
                                                                Y_sets_P_tensors=P_k_explain_list)
            kappa_vec_S_PkS_norm = kappa_mat_S_TrainPkS_norm.squeeze(-1)

            if len(s_indices) == self.n_players:
                score = self.alpha_ocsmm @ kappa_vec_S_PkS_norm
                game_values_list.append(score.item())
                continue
            
            if len(s_indices) == 0:
                # This is v(emptyset) before base Game class normalization.
                # Your formula implies a complex value, but for SHAP baseline, 0.0 is often used.
                # The base Game class will subtract self.normalization_value from this.
                # If you want final v(empty)=0, and this returns 0, set self.normalization_value=0.
                game_values_list.append(0.0) 
                continue
            
            current_lambda = self.lambda_formula if self.lambda_formula > 1e-9 else 1e-9
            try:
                K_SS_reg_inv = torch.linalg.inv(
                    K_SS_train_norm + current_lambda * torch.eye(self.m_train_groups, device=self.device)
                )
            except torch.linalg.LinAlgError:
                 K_SS_reg_inv = torch.linalg.pinv(
                    K_SS_train_norm + current_lambda * torch.eye(self.m_train_groups, device=self.device)
                )
            
            Term_A = kappa_vec_S_PkS_norm.unsqueeze(1) * K_SbarSbar_train_norm
            Term_B = Term_A @ K_SS_reg_inv
            ValVec_for_alpha = Term_B @ kappa_vec_S_PkS_norm
            score_s = self.alpha_ocsmm @ ValVec_for_alpha
            
            game_values_list.append(score_s.item())

        return np.array(game_values_list, dtype=float)

    # __call__ is inherited from shapiq.games.Game and will call self.value_function
    # and then apply normalization: return self.value_function(coalitions) - self.normalization_value