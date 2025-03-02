from typing import List
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch import FloatTensor


def compute_median_heuristic_gamma_x(X: FloatTensor) -> int:
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))  # Squared Euclidean distances
    median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  # Ignore zero distances
    return 1 / (2 * median_dist)

def compute_median_heuristic_gamma_d(datasets: List[FloatTensor]) -> int:
    

