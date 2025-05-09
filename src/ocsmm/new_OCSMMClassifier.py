# one_class_smm.py  – ν‑One‑Class Support Measure Machine (final patch)

import numpy as np
import cvxopt
from joblib import Parallel, delayed


class OneClassSMMClassifier:
    def __init__(self, *, nu: float = 0.1, gamma: float | None = None,
                 n_jobs: int = 8, random_state: int | None = None):
        if not 0. < nu <= 1.:
            raise ValueError("nu must be in (0,1].")
        self.nu, self.gamma_init, self.n_jobs = nu, gamma, n_jobs
        self.rng = np.random.default_rng(random_state)

        # filled in fit()
        self.gamma = None
        self.alpha = None
        self.train_kappa = None
        self.C = None
        self.idx_SV = None
        self.idx_bnd = None
        self.rho = None

    # ------------------------------------------ public API
    def fit(self, datasets: list[np.ndarray]):
        self.datasets = datasets
        g = len(datasets)

        # γ
        self.gamma = (self._median_heuristic_gamma()
                      if self.gamma_init is None else float(self.gamma_init))

        # Gram matrix
        self.train_kappa = self.kappa_matrix(datasets, datasets)

        # QP
        ones = np.ones((g, 1), dtype=np.float64)
        self.C = 1.0 / (self.nu * g)
        P = cvxopt.matrix(self.train_kappa.astype("double"))
        q = cvxopt.matrix(np.zeros((g, 1)))
        G = cvxopt.matrix(np.vstack((-np.eye(g), np.eye(g))))
        h = cvxopt.matrix(np.vstack((np.zeros((g, 1)), self.C * ones)))
        A = cvxopt.matrix(ones.T)
        b = cvxopt.matrix(1.0)

        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        self.alpha = np.asarray(sol["x"]).ravel()
        self.idx_SV = np.where(self.alpha > 1e-8)[0]
        self.idx_bnd = np.where((self.alpha > 1e-8) &
                                (self.alpha < self.C - 1e-8))[0]
        self.rho = self._compute_rho()
        return self

    def decision_function(self, test_sets: list[np.ndarray]) -> np.ndarray:
        K_test = self.kappa_matrix(self.datasets, test_sets)
        return self.alpha[self.idx_SV] @ K_test[self.idx_SV] - self.rho

    def predict(self, test_sets):
        dec = self.decision_function(test_sets)
        return np.sign(dec).astype(int), dec

    # ------------------------------------------ internals
    @staticmethod
    def _rbf(X, Y, gamma):
        X = np.asarray(X, dtype=np.float64, order="C")
        Y = np.asarray(Y, dtype=np.float64, order="C")
        sq = (X*2).sum(1)[:, None] + (Y*2).sum(1)[None, :] - 2.0 * X @ Y.T
        return np.exp(-gamma * sq)

    def kernel(self, X, Y):
        return self._rbf(X, Y, self.gamma).mean()

    def _median_heuristic_gamma(self):
        def group_gamma(G):
            G = np.asarray(G, dtype=np.float64, order="C")
            sq = (G**2).sum(1)[:, None]
            d2 = sq + sq.T - 2.0 * G @ G.T
            med = np.median(d2[d2 > 0])
            return 15 / max(med, 1e-6)

        gammas = Parallel(n_jobs=self.n_jobs)(
            delayed(group_gamma)(g) for g in self.datasets)
        return float(np.median(gammas))

    def kappa_matrix(self, X_sets, Y_sets):
        g1, g2 = len(X_sets), len(Y_sets)
        K = np.empty((g1, g2), dtype=np.float64)

        def entry(i, j):
            return i, j, self.kernel(X_sets[i], Y_sets[j])

        for i, j, val in Parallel(n_jobs=self.n_jobs)(
            delayed(entry)(i, j) for i in range(g1) for j in range(g2)
        ):
            K[i, j] = val

        norm1 = np.sqrt(np.diag(K)).reshape(-1, 1)
        if X_sets is Y_sets:
            norm2 = norm1.T
        else:
            norm2 = np.sqrt(np.diag(self.kappa_matrix(Y_sets, Y_sets)))[None, :]
        return K #/ (norm1 @ norm2)

    def _compute_rho(self):
        K_sv = self.train_kappa[self.idx_SV][:, self.idx_SV]
        dec_sv = K_sv @ self.alpha[self.idx_SV]

        mask_bnd = (self.alpha[self.idx_SV] > 1e-8) & \
                   (self.alpha[self.idx_SV] < self.C - 1e-8)

        if mask_bnd.any():
            return float(np.min(dec_sv[mask_bnd]))
        else:                        # all SVs at C
            return float(dec_sv.max())