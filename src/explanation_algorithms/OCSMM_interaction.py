import numpy as np
from scipy.special import bernoulli, binom
from itertools import product # combinations is imported in powerset
from typing import Callable, List, Dict, Tuple 
import torch
from torch import BoolTensor, FloatTensor
# from dataclasses import dataclass, field # Not strictly needed for this version
from tqdm import tqdm
from joblib import Parallel, delayed

def powerset(s, min_size=0, max_size=None):
    from itertools import combinations # Keep import local to function as per original
    if max_size is None:
        max_size = len(s)
    return [set(c) for r in range(min_size, max_size + 1) for c in combinations(s, r)]

def solve_regression(X: np.ndarray, y: np.ndarray, kernel_weights: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0: # No features to regress on
        return np.array([])
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    try:
        # Ensure kernel_weights is 1D for broadcasting
        if kernel_weights.ndim > 1:
            kernel_weights = kernel_weights.squeeze()
            
        WX = kernel_weights[:, np.newaxis] * X
        phi = np.linalg.solve(X.T @ WX, WX.T @ y)
    except np.linalg.LinAlgError:
        W_sqrt = np.sqrt(kernel_weights)
        # Avoid modifying input X and y by creating new variables for weighted versions
        X_w = W_sqrt[:, np.newaxis] * X
        y_w = W_sqrt * y
        phi = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
    except Exception as e: # Catch other potential errors e.g. X is empty after slicing
        # print(f"Error in solve_regression: {e}. X shape: {X.shape}, y shape: {y.shape}")
        # Fallback or re-raise depending on desired behavior.
        # For now, if X is not empty but solve fails, try lstsq as a general fallback.
        # If X was empty, it should have been caught above.
        try:
            W_sqrt = np.sqrt(kernel_weights)
            X_w = W_sqrt[:, np.newaxis] * X
            y_w = W_sqrt * y
            phi = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
        except Exception as e2:
            # print(f"Fallback lstsq also failed in solve_regression: {e2}")
            # Return empty or nan array matching expected phi dimension
            return np.full(X.shape[1], np.nan) 
    return phi


class KernelSHAPIQ:
    def __init__(self, n: int, max_order: int):
        self.n = n
        self.max_order = max_order
        self._big_M = 1_000_000_000
        self.num_cpus = 30 # Default, can be overridden
        self._bernoulli_numbers = bernoulli(n) 
        self._grand_coalition_set = set(range(n))
        # self.cme_regularisation = torch.tensor(1e-4).float() # Original had this, not directly used by core KernelSHAP-IQ here
        
        # interaction_lookup: maps sorted interaction tuple to a unique global index
        # This lookup includes interactions of ALL orders from 0 to max_order.
        self.interaction_lookup = self._build_interaction_lookup()

    def _build_interaction_lookup(self) -> Dict[Tuple[int, ...], int]:
        lookup = {}
        idx = 0
        for order in range(0, self.max_order + 1): # Crucially, includes order 0
            for interaction_set in powerset(self._grand_coalition_set, min_size=order, max_size=order):
                interaction_tuple = tuple(sorted(list(interaction_set)))
                lookup[interaction_tuple] = idx
                idx += 1
        return lookup

    def _init_kernel_weights(self, interaction_size_ell: int) -> np.ndarray: # ell is the order being estimated
        # This computes µ_ℓ(|T|) from the KernelSHAP-IQ paper
        weight_vector = np.zeros(shape=self.n + 1) # Indexed by coalition_size |T|
        
        # KernelSHAP-IQ µ_ℓ(|T|) for ℓ >= 1 (interaction_size_ell is ℓ)
        # For ℓ=0, the paper doesn't define a specific µ_0. phi_0 is usually v(emptyset).
        # If interaction_size_ell is 0, this formula might not be what's intended for phi_0 via WLS.
        # We handle phi_0 directly, so this kernel is mainly for ℓ >= 1.
        if interaction_size_ell == 0: # Not typically used for WLS in iterative KernelSHAP-IQ
            # For completeness, if one *were* to use WLS for phi_0 with this framework:
            # A common choice might be high weights on empty/grand, low/uniform on others.
            # Or Shapley kernel for |S|=0 which means averaging over all players.
            # However, for phi_0 = v(emptyset), no WLS is needed.
            # Let's make it big_M for non-empty/grand if ℓ=0 to reflect it's not standard.
            for coalition_size_t in range(self.n + 1):
                if coalition_size_t == 0 or coalition_size_t == self.n :
                    weight_vector[coalition_size_t] = 1.0 # Or some other high val if used in WLS
                else:
                    weight_vector[coalition_size_t] = self._big_M # Effectively ignore these
            return weight_vector


        for coalition_size_t in range(self.n + 1): # coalition_size_t is |T|
            # Denominator for µ_ℓ(|T|)
            # (n - 2ℓ + 1) * C(n - 2ℓ, |T| - ℓ)
            if coalition_size_t < interaction_size_ell or \
               coalition_size_t > self.n - interaction_size_ell:
                weight_vector[coalition_size_t] = self._big_M
            else:
                term_n_minus_2ell_plus_1 = self.n - 2 * interaction_size_ell + 1
                if term_n_minus_2ell_plus_1 <= 0: # Binomial C(k,r) undefined if k < 0 or k < r
                    weight_vector[coalition_size_t] = self._big_M
                    continue
                
                try: # binom can fail if n-2l < |T|-l or |T|-l < 0
                    term_binom = binom(self.n - 2 * interaction_size_ell, 
                                       coalition_size_t - interaction_size_ell)
                except ValueError:
                    weight_vector[coalition_size_t] = self._big_M
                    continue

                denominator = term_n_minus_2ell_plus_1 * term_binom
                                
                if denominator == 0 or np.isinf(denominator) or np.isnan(denominator):
                    weight_vector[coalition_size_t] = self._big_M
                else:
                    weight_vector[coalition_size_t] = 1.0 / denominator
        return weight_vector
    
    def _bernoulli_weights_coefficient(self, intersection_size_t_s: int, interaction_size_s: int) -> float:
        # Calculates λ(|S|, |T∩S|) = λ(s, t_s)
        # s = interaction_size_s = |S|
        # t_s = intersection_size_t_s = |T∩S|
        # Formula: Sum_{j=0}^{t_s} (-1)^(t_s - j) * C(t_s, j) * B_{s-j}
        # (B_k are standard Bernoulli numbers, B_1 = -1/2)
        
        val = 0.0
        for j in range(intersection_size_t_s + 1):
            term_binom = binom(intersection_size_t_s, j)
            bernoulli_index = interaction_size_s - j
            
            b_val = 0.0
            if bernoulli_index < 0:
                b_val = 0.0 
            elif bernoulli_index == 1: # B_1 = -0.5
                b_val = self._bernoulli_numbers[1] 
            elif bernoulli_index < len(self._bernoulli_numbers):
                b_val = self._bernoulli_numbers[bernoulli_index]
            # else: b_val remains 0 (for bernoulli_index >= len(self._bernoulli_numbers), e.g. B_m for m odd > 1 if not stored)
            
            val += ((-1)**(intersection_size_t_s - j)) * term_binom * b_val
        return val

    def _get_bernoulli_weights_table(self) -> np.ndarray:
        # Precomputes table for λ(|S|, |T∩S|)
        # Indexed by |S|, then |T∩S|
        lambda_table = np.zeros((self.max_order + 1, self.max_order + 1)) 
        for s_abs in range(self.max_order + 1): # |S| from 0 to max_order
            for t_intersect_s_abs in range(s_abs + 1): # |T∩S| from 0 to |S|
                lambda_table[s_abs, t_intersect_s_abs] = self._bernoulli_weights_coefficient(
                    intersection_size_t_s=t_intersect_s_abs, 
                    interaction_size_s=s_abs
                )
        return lambda_table

    def _get_X_feature_matrix_for_all_interactions(
            self, 
            coalitions_matrix: np.ndarray, # Shape (2^n, n)
            lambda_table: np.ndarray, # Precomputed λ(|S|,|T∩S|) table
            # self.interaction_lookup is used here implicitly
            ) -> Tuple[np.ndarray, List[int], List[List[int]]]:
        # Returns:
        # 1. X_full: The full regression matrix where X_full[T, S_idx] = λ(|S_idx|, |T∩S_idx|)
        #            Columns are ordered by self.interaction_lookup's global indices.
        # 2. interaction_global_indices: List of global indices for all interactions considered.
        # 3. order_to_global_column_indices: Maps order k to list of global column indices in X_full.

        num_total_interactions = len(self.interaction_lookup)
        
        # interaction_masks_arr: columns are 0/1 characteristic vectors of interactions S
        # interaction_sizes_s_arr: |S| for each interaction S
        # These are ordered according to the global_idx from self.interaction_lookup
        interaction_masks_arr = np.zeros((self.n, num_total_interactions), dtype=int)
        interaction_sizes_s_arr = np.zeros(num_total_interactions, dtype=int)
        
        interaction_global_indices = [0] * num_total_interactions # Store global indices for clarity
        order_to_global_column_indices = [[] for _ in range(self.max_order + 1)]

        for interaction_tuple, global_idx in self.interaction_lookup.items():
            interaction_s_size = len(interaction_tuple)
            interaction_sizes_s_arr[global_idx] = interaction_s_size
            if interaction_tuple: # Not empty set
                interaction_masks_arr[list(interaction_tuple), global_idx] = 1
            
            interaction_global_indices[global_idx] = global_idx # Redundant but shows order
            if interaction_s_size <= self.max_order:
                order_to_global_column_indices[interaction_s_size].append(global_idx)
        
        # Sort indices within each order for consistent slicing later
        for order_list in order_to_global_column_indices:
            order_list.sort()

        # intersection_T_S_sizes_matrix: rows are coalitions T, columns are interactions S (by global_idx).
        # Value is |T∩S|. Shape (2^n, num_total_interactions)
        intersection_T_S_sizes_matrix = coalitions_matrix @ interaction_masks_arr 
        
        # X_full: rows are T, cols are S (by global_idx). Value is λ(|S|, |T∩S|).
        # lambda_table is indexed by |S|, then |T∩S|.
        # interaction_sizes_s_arr[j] gives |S_j| for column j.
        # intersection_T_S_sizes_matrix[i, j] gives |T_i ∩ S_j|.
        X_full = lambda_table[
            interaction_sizes_s_arr[np.newaxis, :], # Broadcast |S| for each column
            intersection_T_S_sizes_matrix          # Provides |T∩S| for each (T,S) pair
        ]
        
        return X_full, interaction_global_indices, order_to_global_column_indices


    def _find_interactions_single_group_iterative(
        self, 
        game_values_for_group: np.ndarray, # nu(T) for current group, shape (2^n,)
        coalitions_matrix: np.ndarray,     # Shape (2^n, n)
        X_full_matrix: np.ndarray,         # Precomputed full X matrix, shape (2^n, num_total_interactions)
        kernel_weights_mu_ell_dict: Dict[int, np.ndarray], # Dict: order ℓ -> µ_ℓ(|T|) vector
        interaction_lookup_rev: Dict[int, Tuple[int, ...]], # Maps global_idx back to interaction tuple
        order_to_global_column_indices: List[List[int]], # Maps order ℓ to list of global column indices in X_full
        empty_set_global_idx: int # Global index of the empty set interaction S={}
    ) -> Dict[Tuple[int, ...], float]:

        if isinstance(game_values_for_group, torch.Tensor): # Should be np.ndarray by now
            y_current_residual = game_values_for_group.detach().cpu().numpy().copy()
        else:
            y_current_residual = game_values_for_group.copy() # This is ŷ_1 from algorithm
            
        estimated_phis_all_orders: Dict[Tuple[int, ...], float] = {}

        # Step 1: Estimate phi_0 (interaction of order 0, the empty set S={})
        # phi_0 = v(emptyset) as per Algorithm1 (often implied or done before loop)
        empty_set_tuple = tuple() 
        
        # Find the row in coalitions_matrix for T={} (all zeros)
        empty_set_coalition_row_idx = -1
        for r_idx, row_sum in enumerate(np.sum(coalitions_matrix, axis=1)):
            if row_sum == 0:
                empty_set_coalition_row_idx = r_idx
                break
        if empty_set_coalition_row_idx == -1:
                raise ValueError("Empty set coalition T={} not found in coalitions_matrix.")

        phi_0_val = y_current_residual[empty_set_coalition_row_idx]
        estimated_phis_all_orders[empty_set_tuple] = phi_0_val
        
        # Update residuals: y_current_residual = y_current_residual - X_0 * phi_0
        # X_0 is the column in X_full_matrix for S={}.
        # For S={}, |S|=0. For any T, |T∩S|=0. λ(0,0) = B_0 = 1.
        # So, X_full_matrix[:, empty_set_global_idx] should be a column of 1s.
        # y_current_residual = y_current_residual - X_full_matrix[:, empty_set_global_idx] * phi_0_val
        # This simplifies to (since X_0 column is all 1s):
        y_current_residual = y_current_residual - phi_0_val # Now y_current_residual is ŷ_1' = ŷ_1 - X_0*phi_0

        # Step 2: Iteratively estimate higher-order interactions (ℓ = 1 to max_order)
        # Algorithm line 3: for ℓ = 1,...,k do
        for ell in range(1, self.max_order + 1): # ell is ℓ from Algorithm1
            
            # Get global column indices in X_full_matrix for interactions S of current order ℓ
            global_indices_for_order_ell = order_to_global_column_indices[ell]
            
            if not global_indices_for_order_ell: # No interactions of this order
                continue

            # X_ell: Columns from X_full_matrix specific to interactions S of order ℓ
            # This is (ˆXℓ) from Algorithm line 5
            X_ell = X_full_matrix[:, global_indices_for_order_ell]
            
            # W_star_ell: Kernel weights µ_ℓ(|T|) for WLS when estimating order ℓ effects
            # This is (ˆW∗ℓ) from Algorithm line 6
            kernel_weights_mu_vector_for_ell = kernel_weights_mu_ell_dict[ell] # This is µ_ℓ vector
            coalition_sizes_T_vec = np.sum(coalitions_matrix, axis=1) # |T| for each row
            W_star_ell_weights = kernel_weights_mu_vector_for_ell[coalition_sizes_T_vec]

            if X_ell.shape[1] == 0: # Should be caught by `if not global_indices_for_order_ell`
                continue

            # Solve WLS: ϕ_ℓ ← SOLVEWLS(ˆXℓ, ŷℓ, ˆW∗ℓ) (Algorithm line 9, since ℓ <= 2)
            # y_current_residual here is ŷℓ
            phi_values_for_order_ell = solve_regression(
                X_ell, 
                y_current_residual, 
                W_star_ell_weights
            )

            if phi_values_for_order_ell.size == 0 or np.all(np.isnan(phi_values_for_order_ell)):
                # print(f"Warning: solve_regression for order {ell} returned empty or all NaN. Skipping update.")
                # This might happen if X_ell has issues (e.g. singular after weighting, or empty)
                # or if all kernel weights made the problem ill-posed.
                # If phi is NaN, then y_current_residual update will also be NaN. Best to skip.
                for k_idx_in_order_ell in range(len(global_indices_for_order_ell)):
                    global_col_idx = global_indices_for_order_ell[k_idx_in_order_ell]
                    interaction_tuple = interaction_lookup_rev[global_col_idx]
                    estimated_phis_all_orders[interaction_tuple] = np.nan # Record as NaN
                continue


            # Store the computed ϕ_ℓ values
            for k_idx_in_order_ell, phi_val_s_of_order_ell in enumerate(phi_values_for_order_ell):
                global_col_idx = global_indices_for_order_ell[k_idx_in_order_ell]
                interaction_tuple = interaction_lookup_rev[global_col_idx]
                estimated_phis_all_orders[interaction_tuple] = phi_val_s_of_order_ell
            
            # Update residuals: ŷ_{ℓ+1} ← ŷℓ − ˆXℓ · ˆϕℓ (Algorithm Line 18)
            y_current_residual = y_current_residual - (X_ell @ phi_values_for_order_ell)

        return estimated_phis_all_orders # This is ˆΦk (all SIIs up to k)


    def find_interactions(self, game: Callable[[np.ndarray], np.ndarray]) -> List[Dict[Tuple[int, ...], float]]:
        # --- Preparations ---
        # Algorithm Line 1: Generate coalitions {Ti}
        coalitions_matrix = np.array(list(product([0, 1], repeat=self.n)), dtype=int)
        
        # Algorithm Line 2: Compute game values ŷ1 = [ν(T1),...,ν(Tb)]T
        game_values_matrix_raw = game(coalitions_matrix) # nu(T) for all T
        
        if isinstance(game_values_matrix_raw, torch.Tensor):
            game_values_matrix_np = game_values_matrix_raw.detach().cpu().numpy()
        else:
            game_values_matrix_np = np.asarray(game_values_matrix_raw) # Ensure it's a numpy array

        num_groups = 1
        if game_values_matrix_np.ndim > 1:
            num_groups = game_values_matrix_np.shape[1]
        else: # If game_values_matrix is 1D (single group)
            game_values_matrix_np = game_values_matrix_np[:, np.newaxis] # Reshape to (2^n, 1)

        # Precompute shared items needed for the iterative estimation:
        # 1. λ(|S|,|T∩S|) table (for Algorithm Line 5)
        lambda_table = self._get_bernoulli_weights_table()
        
        # 2. Full X feature matrix (X_full[T,S_idx] = λ(|S_idx|,|T∩S_idx|)), 
        #    and mapping from order to column indices in this X_full
        X_full_matrix, _, order_to_global_column_indices = \
            self._get_X_feature_matrix_for_all_interactions(coalitions_matrix, lambda_table)

        # 3. µ_ℓ(|T|) kernel weights for each order ℓ (for Algorithm Line 6)
        kernel_weights_mu_ell_dict = {
            ell: self._init_kernel_weights(ell) for ell in range(1, self.max_order + 1) 
            # For ell=0, phi_0 is handled directly, so mu_0 not strictly needed here.
        }
        
        # 4. Reverse lookup: global_idx -> interaction_tuple
        interaction_lookup_rev = {v: k for k, v in self.interaction_lookup.items()}
        
        # 5. Global index for the empty set interaction S={}
        empty_set_global_idx = self.interaction_lookup.get(tuple())
        if empty_set_global_idx is None:
            raise ValueError("Empty set interaction not found in self.interaction_lookup.")

        all_group_final_sii_estimates: List[Dict[Tuple[int, ...], float]] = []
        
        # --- Main Loop (per group if multiple outputs from game) ---
        for group_idx in tqdm(range(num_groups), desc="KernelSHAP-IQ (Iterative)"):
            current_group_game_values = game_values_matrix_np[:, group_idx] # This is ŷ1 for the current group
            
            # Perform iterative estimation for this group
            sii_estimates_for_group = self._find_interactions_single_group_iterative(
                current_group_game_values,
                coalitions_matrix,
                X_full_matrix,
                kernel_weights_mu_ell_dict,
                interaction_lookup_rev,
                order_to_global_column_indices,
                empty_set_global_idx
            )
            all_group_final_sii_estimates.append(sii_estimates_for_group)

        # Algorithm Line 21: Return k-SII estimates
        return all_group_final_sii_estimates
    
    
    def find_interactions_per_group(
        self, 
        X_list_for_cme: List[torch.Tensor], # Renamed from X for clarity (data for CME game)
        decision_minus_rho_for_cme: torch.Tensor, 
        model_for_cme, 
        regularization_for_cme: float
    ) -> List[Dict[Tuple[int, ...], float]]:
        
        # --- Preparations (similar to find_interactions, but outside parallel loop) ---
        coalitions_matrix = np.array(list(product([0, 1], repeat=self.n)), dtype=int)
        lambda_table = self._get_bernoulli_weights_table()
        X_full_matrix, _, order_to_global_column_indices = \
            self._get_X_feature_matrix_for_all_interactions(coalitions_matrix, lambda_table)
        kernel_weights_mu_ell_dict = {
            ell: self._init_kernel_weights(ell) for ell in range(1, self.max_order + 1)
        }
        interaction_lookup_rev = {v: k for k, v in self.interaction_lookup.items()}
        empty_set_global_idx = self.interaction_lookup.get(tuple())
        if empty_set_global_idx is None:
            raise ValueError("Empty set interaction not found in self.interaction_lookup.")

        # --- Parallel Execution ---
        # Each worker will compute game values for its assigned group, then run the iterative SII estimation.
        results = Parallel(n_jobs=self.num_cpus)(
            delayed(self._worker_compute_game_and_sii_for_one_group)(
                group_idx, 
                X_list_for_cme, 
                decision_minus_rho_for_cme, 
                model_for_cme, 
                regularization_for_cme, 
                # Pass precomputed items to the worker
                coalitions_matrix,
                X_full_matrix,
                kernel_weights_mu_ell_dict,
                interaction_lookup_rev,
                order_to_global_column_indices,
                empty_set_global_idx
            )
            for group_idx in tqdm(range(len(X_list_for_cme)), desc="KernelSHAP-IQ (Parallel Iterative)")
        )
        return results


    def _worker_compute_game_and_sii_for_one_group(
        self, 
        group_idx: int, 
        X_list_for_cme: List[torch.Tensor], 
        decision_minus_rho_for_cme: torch.Tensor, 
        model_for_cme, 
        regularization_for_cme: float, 
        # Received precomputed items
        coalitions_matrix_param: np.ndarray,
        X_full_matrix_param: np.ndarray,
        kernel_weights_mu_ell_dict_param: Dict[int, np.ndarray],
        interaction_lookup_rev_param: Dict[int, Tuple[int, ...]],
        order_to_global_column_indices_param: List[List[int]],
        empty_set_global_idx_param: int
    ) -> Dict[Tuple[int, ...], float]:
        
        # Step 1: Compute game values ν(T) for the current group_idx (Algorithm Line 2, effectively)
        current_group_game_values_list = []
        for S_characteristic_vector in coalitions_matrix_param: 
            S_bool_tensor = torch.tensor(S_characteristic_vector, dtype=torch.bool)
            
            current_group_data_for_cme = X_list_for_cme[group_idx]
            
            group_filtered_for_S = current_group_data_for_cme[:, S_bool_tensor]
            if S_bool_tensor.sum() == 0: # Handle empty set for CME game function
                 group_filtered_for_S = current_group_data_for_cme[:, S_bool_tensor] # (samples, 0)

            val = self.compute_value_function_for_group( # This is ν(S) for this specific CME game
                group_filtered=group_filtered_for_S,
                S_mask=S_bool_tensor,
                all_X_train=X_list_for_cme, 
                decision_minus_rho=decision_minus_rho_for_cme,
                model=model_for_cme,
                regularization=regularization_for_cme
            )
            current_group_game_values_list.append(val)

        current_group_game_values_np = np.array(current_group_game_values_list) # This is ŷ1 for this group
        
        # Step 2: Perform iterative SII estimation using these game values
        sii_estimates = self._find_interactions_single_group_iterative(
            current_group_game_values_np,
            coalitions_matrix_param,
            X_full_matrix_param,
            kernel_weights_mu_ell_dict_param,
            interaction_lookup_rev_param,
            order_to_global_column_indices_param,
            empty_set_global_idx_param
        )
        return sii_estimates
    
    # compute_value_function_for_group: This function defines your game ν(S).
    # Its internal logic is specific to your application (CME) and remains unchanged as requested.
    def compute_value_function_for_group(
        self,
        group_filtered: torch.Tensor,
        S_mask: torch.BoolTensor, # Renamed from S to S_mask for clarity
        all_X_train: List[torch.Tensor], # Renamed from X
        decision_minus_rho: torch.Tensor,
        model, # Model with kappa_matrix
        regularization: float
    ) -> float:
        # Original implementation from your code
        X_train_S_list = [g[:, S_mask] for g in all_X_train]

        device = next(model.parameters()).device if hasattr(model, 'parameters') and callable(model.parameters) and list(model.parameters()) else torch.device("cpu")
        
        group_filtered_list = [group_filtered]
        X_train_S_list_device = [x for x in X_train_S_list]
        all_X_train_device = [x for x in all_X_train]
        alphas_device = decision_minus_rho

        K_xS_XS_np = model.kappa_matrix(group_filtered_list, X_train_S_list_device) 
        K_XS_XS_np = model.kappa_matrix(X_train_S_list_device, X_train_S_list_device)     
        K_X_X_np   = model.kappa_matrix(all_X_train_device, all_X_train_device)         

        K_xS_XS = torch.tensor(K_xS_XS_np, dtype=torch.float32, device=device).squeeze(0)
        K_XS_XS = torch.tensor(K_XS_XS_np, dtype=torch.float32, device=device)
        K_X_X   = torch.tensor(K_X_X_np, dtype=torch.float32, device=device)
        
        m_train = K_XS_XS.shape[0]
        if m_train == 0:
             if K_xS_XS.numel() == 0 or K_xS_XS.sum() == 0 : # More robust check for empty/zero K_xS_XS
                 return 0.0 
             # if K_XS_XS is empty, K_reg will be regularization * I.
             # If K_xS_XS is non-zero, but there's no basis from X_train_S (m_train=0), result might be ill-defined or 0.
             # This depends on how model.kappa_matrix(X_train_S_list_device, X_train_S_list_device) behaves.
             # Let's assume if m_train is 0, K_XS_XS is effectively zero.
             # Then K_reg = lambda*I. If lambda is also 0, this is an issue.
             if regularization == 0: # and m_train == 0:
                 # This implies K_reg is 0x0 or an empty matrix, solve will fail.
                 # A principled value here depends on the game definition.
                 # print("Warning: m_train is 0 and regularization is 0. Returning 0.0 for game value.")
                 return 0.0 
             # If regularization > 0, K_reg is lambda * I (size 0x0), this path needs careful thought.
             # For now, assuming if m_train=0, the system might not be solvable as expected.
             # The original lstsq path in solve_regression might handle X_w being (N,0) if X_ell is (N,0)
             # but here we are in the game function.
             pass # Allow to proceed, K_reg will be lambda_I of size (0,0)

        lambda_I = regularization * torch.eye(m_train, dtype=torch.float32, device=device)
        K_reg = K_XS_XS + lambda_I

        # Ensure K_X_X @ alphas_device is compatible with K_reg for solve
        # K_X_X is (m_train, m_train), alphas_device is (m_train,)
        # K_X_X @ alphas_device is (m_train,)
        target_for_solve = (K_X_X @ alphas_device) # This is a vector

        if m_train == 0 : # If K_reg is 0x0 (or similar degenerate cases)
            if K_xS_XS.numel() > 0 and K_xS_XS.abs().sum() > 0: # If k_xS_XS has non-zero values
                 # but there's no basis from X_train_S (K_reg is essentially empty or just lambda*I of size 0)
                 # This implies the term (K_XS_XS + lambdaI)^-1 (K_X_X alpha) is problematic.
                 # print(f"Warning: m_train is 0 for coalition S={S_mask.tolist()}. Value might be ill-defined. Returning 0.")
                 pass # Let it attempt solve, might fail if K_reg is truly 0x0.
                 # If K_reg is 0x0, linalg.solve will error.
                 # If K_xS_XS is also 0x0, then dot product is 0.
                 if K_xS_XS.numel() == 0: return 0.0

        solved_part = torch.zeros_like(target_for_solve) # Initialize
        if m_train > 0 : # Only solve if K_reg is not empty
            try:
                # target_for_solve must be (m_train, 1) for torch.linalg.solve(A,B) if B is a matrix
                # If B is a vector, solve(A,B) is fine.
                solved_part = torch.linalg.solve(K_reg, target_for_solve) 
            except torch.linalg.LinAlgError:
                # print(f"Warning: torch.linalg.solve failed. Using lstsq for coalition S={S_mask.tolist()}.")
                try:
                    solved_part = torch.linalg.lstsq(K_reg, target_for_solve).solution
                except Exception as e_lstsq:
                    # print(f"lstsq fallback also failed: {e_lstsq}. Returning 0 for game value.")
                    return 0.0 # Or handle more gracefully
            except RuntimeError as e_rt: # Catch other potential errors
                # print(f"RuntimeError during linalg for S={S_mask.tolist()}: {e_rt}. Using pinv or returning 0.")
                try:
                    pseudo_inv_K_reg = torch.linalg.pinv(K_reg.to(dtype=torch.float64)).to(dtype=torch.float32) # for stability
                    solved_part = pseudo_inv_K_reg @ target_for_solve
                except Exception:
                    return 0.0
        elif m_train == 0 and regularization > 0 : # K_reg is lambda*I (0x0), effectively empty.
            # If target_for_solve is also empty (m_train=0), and K_xS_XS is empty, result is 0.
            if target_for_solve.numel() == 0 and K_xS_XS.numel() == 0:
                return 0.0
            # This case is tricky, depends on conventions for empty matrix operations.
            # print(f"Warning: m_train=0 but regularization > 0. Result might be unexpected. Returning 0 for coalition S={S_mask.tolist()}.")
            return 0.0


        # K_xS_XS is (m_train,), solved_part is (m_train,)
        result = torch.dot(K_xS_XS, solved_part)
        return result.item()