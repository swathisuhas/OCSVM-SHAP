import numpy as np
from typing import List, Tuple, Union, Callable # Added Union, Callable
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, pairwise_kernels
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns # For confusion matrix heatmap


# --- Dataset Generation Function (Copied from User) ---
def generate_gaussian_anomaly_dataset(
    n_normal_groups: int = 30,
    n_rotated_groups: int = 2,
    n_samples_per_group: int = 100,
    seed: int = 32
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[np.ndarray]]:
    """
    Generates a dataset with normal, rotated covariance, and shifted groups
    based on 2D Gaussian distributions.

    Args:
        n_normal_groups (int): Number of normal groups to generate.
        n_rotated_groups (int): Number of anomalous groups with rotated covariance.
        n_samples_per_group (int): Number of samples in each group.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[np.ndarray], List[str], List[np.ndarray], List[np.ndarray]]:
            - List of group data arrays (shape: n_samples_per_group x 2).
            - List of group type labels ('normal', 'rotated', 'shifted').
            - List of the means used to generate each group.
            - List of the covariance matrices used to generate each group.
    """
    np.random.seed(seed)
    dimensions = 2

    # --- Parameters for Distributions ---
    base_cov_normal = np.array([[0.01, 0.008],
                                [0.008, 0.01]])
    rotation_angle_deg = 90
    angle_rad = np.radians(rotation_angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    cov_matrix_rot = rotation_matrix @ base_cov_normal @ rotation_matrix.T
    mean_low = 0
    mean_high = 1
    shift_vector = np.array([ 1, 1])

    # --- Generate Group Means ---
    normal_group_means = np.random.uniform(
        low=mean_low,
        high=mean_high,
        size=(n_normal_groups, dimensions)
    )

    generated_groups = []
    group_labels = []
    group_means = []
    group_covs = []

    # --- Generate Normal Groups ---
    print(f"Generating {n_normal_groups} normal groups...")
    for i in range(n_normal_groups):
        mean = normal_group_means[i]
        data_normal = np.random.multivariate_normal(
            mean=mean,
            cov=base_cov_normal,
            size=n_samples_per_group
        )
        generated_groups.append(data_normal)
        group_labels.append('normal')
        group_means.append(mean)
        group_covs.append(base_cov_normal)

    # --- Generate Rotated Covariance Groups ---
    if n_rotated_groups > n_normal_groups:
        raise ValueError("n_rotated_groups cannot be larger than n_normal_groups")
    print(f"Generating {n_rotated_groups} rotated covariance groups...")
    # Use *different* means for rotated to make them distinct test cases
    rotated_group_means = np.random.uniform(
        low=mean_low,
        high=mean_high,
        size=(n_rotated_groups, dimensions)
    )
    for i in range(n_rotated_groups):
        mean = rotated_group_means[i] # Use a distinct mean
        data_rotated = np.random.multivariate_normal(
            mean=mean,
            cov=cov_matrix_rot,
            size=n_samples_per_group
        )
        generated_groups.append(data_rotated)
        group_labels.append('rotated')
        group_means.append(mean)
        group_covs.append(cov_matrix_rot)

    # --- Generate Shifted Normal Group ---
    # Use a new mean for the shifted group as well
    shifted_mean = np.random.uniform(low=mean_low, high=mean_high, size=dimensions) + shift_vector
    print(f"Generating 1 shifted group...")
    shifted_group_data = np.random.multivariate_normal(
        mean=shifted_mean,
        cov=base_cov_normal, # Use normal covariance
        size=n_samples_per_group
    )
    generated_groups.append(shifted_group_data)
    group_labels.append('shifted')
    group_means.append(shifted_mean)
    group_covs.append(base_cov_normal)

    print("Dataset generation complete.")
    print(f"Total groups generated: {len(generated_groups)}")
    print(f"Label counts: Normal={group_labels.count('normal')}, Rotated={group_labels.count('rotated')}, Shifted={group_labels.count('shifted')}")

    return generated_groups, group_labels, group_means, group_covs


# --- Helper function for plotting confidence ellipse (Copied from User) ---
def plot_confidence_ellipse(ax, mean, cov, n_std=2.0, facecolor='none', **kwargs):
    """ Plots an n_std confidence ellipse centered at mean based on cov. """
    # Check if covariance matrix is positive semi-definite
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
         print(f"Warning: Covariance matrix decomposition failed. Skipping ellipse for mean {mean}.")
         return None
    # Ensure eigenvalues are non-negative after potential numerical issues
    if np.any(vals < -1e-6): # Allow small negative tolerance
         print(f"Warning: Covariance matrix is not positive semi-definite (eigenvalues: {vals}). Skipping ellipse for mean {mean}.")
         return None

    vals = np.maximum(vals, 0) # Clamp small negatives to zero

    # Get angle of rotation from eigenvector corresponding to largest eigenvalue
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1])) # Angle for major axis

    # Ellipse width and height are proportional to sqrt of eigenvalues
    # Handle cases where eigenvalues are zero or very close to zero
    width = 2 * n_std * np.sqrt(vals[0]) if vals[0] > 1e-9 else 1e-5
    height = 2 * n_std * np.sqrt(vals[1]) if len(vals) > 1 and vals[1] > 1e-9 else 1e-5

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)


# --- Helper Functions for OCSMM (Slightly updated print statements) ---

def compute_kappa(group1: np.ndarray, group2: np.ndarray, base_kernel_func: Callable, **kernel_params) -> float:
    """ Computes the empirical estimate of the aggregate kernel κ(P1, P2) = <μ_P1, μ_P2>. """
    n1 = group1.shape[0]
    n2 = group2.shape[0]
    if n1 == 0 or n2 == 0: return 0.0
    # Use the metric string if available, otherwise pass the callable
    metric = kernel_params.pop('metric', base_kernel_func)
    # Handle gamma='scale' specifically for pairwise_kernels if needed
    # Note: gamma='scale' depends on the *combined* data variance which isn't ideal here.
    # It's often better to tune gamma manually or scale data beforehand.
    # We'll pass kernel_params directly.
    kernel_matrix = pairwise_kernels(group1, group2, metric=metric, **kernel_params)
    # Put metric back if it was popped, for next iteration
    if isinstance(metric, str): kernel_params['metric'] = metric
    kappa_value = np.sum(kernel_matrix) / (n1 * n2)
    return kappa_value

def compute_aggregate_kernel_matrix(
    groups: List[np.ndarray],
    base_kernel_func: Union[str, Callable],
    normalize: bool = True,
    epsilon: float = 1e-9,
    **kernel_params
) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the m x m aggregate kernel matrix K, where K[i, j] = κ(Pi, Pj). """
    m = len(groups)
    kernel_matrix_unnorm = np.zeros((m, m))

    # If base_kernel_func is a string, pass it as 'metric'
    metric_arg = base_kernel_func if isinstance(base_kernel_func, str) else base_kernel_func
    params_for_kappa = kernel_params.copy()
    if isinstance(metric_arg, str):
        params_for_kappa['metric'] = metric_arg # Pass name like 'rbf'

    for i in range(m):
        for j in range(i, m):
            # Pass the original function if it's callable, otherwise pass params with 'metric'
            if callable(metric_arg):
                 kappa_ij = compute_kappa(groups[i], groups[j], metric_arg, **kernel_params)
            else:
                 kappa_ij = compute_kappa(groups[i], groups[j], None, **params_for_kappa) # None func, relies on metric='rbf' etc in params

            kernel_matrix_unnorm[i, j] = kappa_ij
            if i != j:
                kernel_matrix_unnorm[j, i] = kappa_ij

    kernel_diag_unnorm = np.diag(kernel_matrix_unnorm).copy()

    if normalize:
        sqrt_diag = np.sqrt(np.maximum(kernel_diag_unnorm, 0) + epsilon) # Ensure non-negative before sqrt
        normalizer = np.outer(sqrt_diag, sqrt_diag)
        kernel_matrix_norm = np.divide(kernel_matrix_unnorm, normalizer,
                                       out=np.zeros_like(kernel_matrix_unnorm),
                                       where=normalizer!=0)
        return kernel_matrix_norm, kernel_diag_unnorm
    else:
        return kernel_matrix_unnorm, kernel_diag_unnorm


# --- OCSMM Class (Slightly updated to handle base_kernel string/callable better) ---
class OCSMM:
    """ One-Class Support Measure Machine (OCSMM) """
    def __init__(self, nu=0.1, base_kernel='rbf', normalize_aggregate=True, **base_kernel_params):
        if not 0 < nu <= 1: raise ValueError("nu must be in the interval (0, 1]")
        self.nu = nu
        self.normalize_aggregate = normalize_aggregate
        self.base_kernel_ = base_kernel # Store the name or callable
        self.base_kernel_params_ = base_kernel_params # Store params like gamma, degree

        self.svm_model_ = None
        self.training_groups_ = None
        self.training_kernel_diag_unnorm_ = None
        self._is_fitted = False

    def fit(self, groups: List[np.ndarray]):
        """ Fit the OCSMM model """
        print("Computing aggregate kernel matrix for training...")
        self.training_groups_ = groups
        n_groups = len(groups)

        # Compute the aggregate kernel matrix
        kernel_matrix, self.training_kernel_diag_unnorm_ = compute_aggregate_kernel_matrix(
            groups,
            self.base_kernel_, # Pass name or callable
            normalize=self.normalize_aggregate,
            **self.base_kernel_params_
        )
        print(f"Aggregate Kernel Matrix shape: {kernel_matrix.shape}")
        # Check for NaNs or Infs in kernel matrix
        if np.any(np.isnan(kernel_matrix)) or np.any(np.isinf(kernel_matrix)):
            print("Warning: NaN or Inf detected in aggregate kernel matrix!")
            # Optional: Add more debugging here, like printing the matrix or params
            num_nan = np.sum(np.isnan(kernel_matrix))
            num_inf = np.sum(np.isinf(kernel_matrix))
            print(f"NaNs: {num_nan}, Infs: {num_inf}")
            # Consider replacing them, though this indicates a deeper issue (e.g., bad gamma)
            # kernel_matrix = np.nan_to_num(kernel_matrix, nan=0.0, posinf=1e6, neginf=-1e6)


        self.svm_model_ = svm.OneClassSVM(kernel='precomputed', nu=self.nu)
        print("Training OneClassSVM on aggregate kernel...")
        try:
            self.svm_model_.fit(kernel_matrix)
        except ValueError as e:
            print(f"\nError during SVM fitting: {e}")
            print("This often happens if the kernel matrix is not valid (e.g., not positive semi-definite after normalization issues, or contains NaN/Inf).")
            print("Check base kernel parameters (especially gamma) and normalization.")
            # You might want to re-raise the error or handle it differently
            raise e
        print("Training complete.")
        self._is_fitted = True
        self.support_indices_ = self.svm_model_.support_
        self.dual_coef_ = self.svm_model_.dual_coef_
        self.intercept_ = self.svm_model_.intercept_
        print(f"Number of support vectors (groups): {len(self.support_indices_)} / {n_groups}")
        return self

    def _compute_prediction_kernel(self, groups_to_predict: List[np.ndarray]) -> np.ndarray:
        """ Helper to compute kernel matrix for prediction """
        if not self._is_fitted:
             raise NotFittedError("This OCSMM instance is not fitted yet. Call 'fit' first.")

        n_new_groups = len(groups_to_predict)
        n_train_groups = len(self.training_groups_)
        kernel_matrix_predict = np.zeros((n_new_groups, n_train_groups))

        # Prepare kernel args like in compute_aggregate_kernel_matrix
        metric_arg = self.base_kernel_ if isinstance(self.base_kernel_, str) else self.base_kernel_
        params_for_kappa = self.base_kernel_params_.copy()
        if isinstance(metric_arg, str):
            params_for_kappa['metric'] = metric_arg

        # print(f"Computing prediction kernel K(New, Train) for {n_new_groups} groups...")
        for i in range(n_new_groups):
            new_group = groups_to_predict[i]
            # Compute kappa(NewGroup_i, TrainGroup_j) for all j
            kappa_new_vs_train = np.zeros(n_train_groups)
            for j in range(n_train_groups):
                 train_group = self.training_groups_[j]
                 if callable(metric_arg):
                      kappa_ij = compute_kappa(new_group, train_group, metric_arg, **self.base_kernel_params_)
                 else:
                      kappa_ij = compute_kappa(new_group, train_group, None, **params_for_kappa)
                 kappa_new_vs_train[j] = kappa_ij


            if self.normalize_aggregate:
                 # Compute self-kernel for the new group
                 if callable(metric_arg):
                    kappa_new_self = compute_kappa(new_group, new_group, metric_arg, **self.base_kernel_params_)
                 else:
                    kappa_new_self = compute_kappa(new_group, new_group, None, **params_for_kappa)

                 # Normalize using stored training diagonals and new group's self-kernel
                 sqrt_diag_train = np.sqrt(np.maximum(self.training_kernel_diag_unnorm_, 0) + 1e-9)
                 sqrt_new_self = np.sqrt(np.maximum(kappa_new_self, 0) + 1e-9)
                 normalizer = sqrt_new_self * sqrt_diag_train
                 kappa_normalized = np.divide(kappa_new_vs_train, normalizer,
                                              out=np.zeros_like(kappa_new_vs_train),
                                              where=normalizer!=0)
                 kernel_matrix_predict[i, :] = kappa_normalized
            else:
                 kernel_matrix_predict[i, :] = kappa_new_vs_train

        # Check for NaNs/Infs in prediction kernel
        if np.any(np.isnan(kernel_matrix_predict)) or np.any(np.isinf(kernel_matrix_predict)):
            print("Warning: NaN or Inf detected in prediction kernel matrix!")
            kernel_matrix_predict = np.nan_to_num(kernel_matrix_predict, nan=0.0, posinf=1e6, neginf=-1e6) # Replace problematic values

        return kernel_matrix_predict


    def predict(self, groups_to_predict: List[np.ndarray]) -> np.ndarray:
        """ Predict whether new groups are normal (1) or anomalous (-1). """
        kernel_matrix_predict = self._compute_prediction_kernel(groups_to_predict)
        # print("Making predictions...")
        predictions = self.svm_model_.predict(kernel_matrix_predict)
        return predictions

    def decision_function(self, groups_to_predict: List[np.ndarray]) -> np.ndarray:
        """ Calculate the decision function score for new groups. """
        kernel_matrix_predict = self._compute_prediction_kernel(groups_to_predict)
        # print("Calculating decision scores...")
        scores = self.svm_model_.decision_function(kernel_matrix_predict)
        return scores


# --- Main Execution Block ---


if __name__ == "__main__":
    # --- 1. Generate Dataset ---
    N_NORMAL_TRAIN = 50
    N_NORMAL_TEST = 47
    N_ROTATED_TEST = 2
    N_SHIFTED_TEST = 1 # Generator only makes 1 shifted group relative to the last normal mean
    N_SAMPLES = 50
    SEED = 42

    # Re-generate the dataset - crucial to ensure consistency
    generated_groups, group_labels_str, group_means, group_covs = \
        generate_gaussian_anomaly_dataset(
            n_normal_groups=N_NORMAL_TRAIN + N_NORMAL_TEST, # Total normal needed
            n_rotated_groups=N_ROTATED_TEST,               # Anomalies to generate
            n_samples_per_group=N_SAMPLES,
            seed=SEED
        )

    # --- 2. Split Data ---
    # (This splitting logic remains the same as before)
    training_groups = []
    test_groups = []
    test_labels_str = []
    test_means = []
    test_covs = []
    test_true_labels_numeric = []

    normal_group_counter = 0
    # Find indices for different types first to ensure correct split
    normal_indices = [i for i, lbl in enumerate(group_labels_str) if lbl == 'normal']
    rotated_indices = [i for i, lbl in enumerate(group_labels_str) if lbl == 'rotated']
    shifted_indices = [i for i, lbl in enumerate(group_labels_str) if lbl == 'shifted']

    # Assign to train/test
    for idx in normal_indices:
        if normal_group_counter < N_NORMAL_TRAIN:
            training_groups.append(generated_groups[idx])
        else:
            test_groups.append(generated_groups[idx])
            test_labels_str.append('normal_test')
            test_means.append(group_means[idx])
            test_covs.append(group_covs[idx])
            test_true_labels_numeric.append(1)
        normal_group_counter += 1

    for idx in rotated_indices:
        test_groups.append(generated_groups[idx])
        test_labels_str.append('rotated')
        test_means.append(group_means[idx])
        test_covs.append(group_covs[idx])
        test_true_labels_numeric.append(-1)

    for idx in shifted_indices:
        test_groups.append(generated_groups[idx])
        test_labels_str.append('shifted')
        test_means.append(group_means[idx])
        test_covs.append(group_covs[idx])
        test_true_labels_numeric.append(-1)

    test_true_labels_numeric = np.array(test_true_labels_numeric)


    print(f"\nData Split:")
    print(f" - Training groups: {len(training_groups)} (all 'normal')")
    print(f" - Test groups: {len(test_groups)}")
    print(f"   - Normal in test: {test_labels_str.count('normal_test')}")
    print(f"   - Rotated in test: {test_labels_str.count('rotated')}")
    print(f"   - Shifted in test: {test_labels_str.count('shifted')}")


    # --- 3. Define and Train OCSMM ---
    # ** Tuning Parameters **
    # INCREASE GAMMA SIGNIFICANTLY TO DETECT ROTATION
    # Let's try gamma = 10.0 or even higher. Keep nu similar or slightly lower.
    GAMMA_VALUE = 15.0  # <-- INCREASED VALUE (Experiment!)
    NU_VALUE = 0.1    # <-- Maybe slightly decreased (Experiment!)

    print(f"\nInitializing OCSMM with: nu={NU_VALUE}, base_kernel='rbf', gamma={GAMMA_VALUE}, normalize={True}")
    ocsmm = OCSMM(nu=NU_VALUE, base_kernel='rbf', gamma=GAMMA_VALUE, normalize_aggregate=True)
    # You can also try normalize_aggregate=False

    ocsmm.fit(training_groups)

    # --- 4. Make Predictions on Test Groups ---
    test_predictions = ocsmm.predict(test_groups)
    test_decision_scores = ocsmm.decision_function(test_groups)

    # --- 5. Evaluate ---
    print("\n--- Evaluation ---")
    print("True Labels (Numeric):", test_true_labels_numeric)
    print("Predicted Labels:     ", test_predictions)
    print("Decision Scores:      ", np.round(test_decision_scores, 3)) # Rounded for clarity

    accuracy = accuracy_score(test_true_labels_numeric, test_predictions)
    print(f"\nAccuracy on test groups: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    target_names = ['Anomaly (-1)', 'Normal (1)']
    try:
        report = classification_report(test_true_labels_numeric, test_predictions, target_names=target_names, zero_division=0) # Added zero_division
        print(report)
    except ValueError as e:
        print(f"Could not generate classification report: {e}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_true_labels_numeric, test_predictions, labels=[1, -1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal (1)', 'Predicted Anomaly (-1)'],
                yticklabels=['True Normal (1)', 'True Anomaly (-1)'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # plt.show() # Wait for combined plot

    # --- 6. Visualize Results ---
    # (Visualization code remains the same as before)

    # 6a. Decision Score Histogram
    plt.figure(figsize=(10, 5))
    # Separate scores for different anomaly types for clarity
    normal_scores = test_decision_scores[np.array(test_labels_str) == 'normal_test']
    rotated_scores = test_decision_scores[np.array(test_labels_str) == 'rotated']
    shifted_scores = test_decision_scores[np.array(test_labels_str) == 'shifted']

    plt.hist(normal_scores, bins=8, alpha=0.6, label=f'Normal Scores (N={len(normal_scores)})', color='blue')
    plt.hist(rotated_scores, bins=5, alpha=0.6, label=f'Rotated Scores (N={len(rotated_scores)})', color='orange')
    plt.hist(shifted_scores, bins=3, alpha=0.6, label=f'Shifted Scores (N={len(shifted_scores)})', color='purple')

    plt.axvline(0, color='black', linestyle='--', label='Decision Boundary (0)')
    plt.title(f'OCSMM Decision Scores (gamma={GAMMA_VALUE}, nu={NU_VALUE})')
    plt.xlabel('Decision Score (Score > 0 -> Normal)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    # plt.show() # Wait for combined plot

    # 6b. Scatter plot of groups with Ellipses and Predictions
    fig, ax = plt.subplots(figsize=(12, 10))
    # (Plotting logic remains the same)
    # Plot training groups (lightly)
    for i, group_data in enumerate(training_groups):
        ax.scatter(group_data[:, 0], group_data[:, 1], alpha=0.05, color='gray', s=5)
    ax.scatter([], [], alpha=0.3, color='gray', s=20, label='Training Groups (Normal)')

    # Plot test groups with ellipses and prediction markers
    colors = {'normal_test': 'blue', 'rotated': 'orange', 'shifted': 'purple'}
    markers = {1: 'o', -1: 'X'}
    prediction_colors = {1: 'green', -1: 'red'}

    for i, group_data in enumerate(test_groups):
        mean = test_means[i]
        cov = test_covs[i]
        true_label_str = test_labels_str[i]
        pred_label = test_predictions[i]

        ax.scatter(group_data[:, 0], group_data[:, 1], alpha=0.15, color=colors[true_label_str], s=10, label=f'_{true_label_str}_points')
        plot_confidence_ellipse(ax, mean, cov, n_std=2.0, edgecolor=colors[true_label_str], linestyle='-', linewidth=1.5, label=f'_{true_label_str}_ellipse')
        ax.scatter(mean[0], mean[1], marker=markers[pred_label], s=100,
                   edgecolor='black', facecolor=prediction_colors[pred_label], linewidth=1.5,
                   label=f'_{true_label_str}_pred')

    # Create custom legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='', linestyle='-', color='gray', label='Training Groups (Normal)', alpha=0.5, linewidth=0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['normal_test'], markersize=8, linestyle='None', label='Test Group: Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['rotated'], markersize=8, linestyle='None', label='Test Group: Rotated Cov'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['shifted'], markersize=8, linestyle='None', label='Test Group: Shifted Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markeredgecolor='k', markersize=10, linestyle='None', label='Prediction: Normal (1)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='k', markersize=10, linestyle='None', label='Prediction: Anomaly (-1)')
    ]

    ax.set_title(f'OCSMM Group Classification (nu={NU_VALUE}, gamma={GAMMA_VALUE})')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    plt.tight_layout()

    plt.show() # Show all plots