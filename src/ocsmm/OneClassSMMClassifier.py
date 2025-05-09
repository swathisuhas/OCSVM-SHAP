import numpy as np
import cvxopt 
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed


class OneClassSMMClassifier:
    def __init__(self, nu):
        self.nu = nu
        self.datasets = None
        self.gamma = None
        
    def fit(self, datasets):
        self.datasets = datasets
        self.gamma = self.find_best_gamma()
        print("Best gamma found.\n")
        num_groups = len(self.datasets)
        self.train_kappa = self.kappa_matrix(self.datasets, self.datasets)
        print("Kappa matrix calculated.\n")
        ones = np.ones(shape=(num_groups,1))
        zeros = np.zeros(shape=(num_groups,1))
        P = cvxopt.matrix(self.train_kappa)
        q = cvxopt.matrix(zeros)
        G = cvxopt.matrix(np.vstack((-np.identity(num_groups), np.identity(num_groups))))
        C = 1.0/(self.nu*num_groups)
        h = cvxopt.matrix(np.vstack((zeros, C*ones)))
        A = cvxopt.matrix(ones.T)
        b = cvxopt.matrix(1.0)
        self.C=C
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        print("Found the alphas.\n")
        self.alpha = np.ravel(solution['x'])
        
    def predict(self, test_dataset):
        print("Calculating kappa.\n")
        self.test_kappa = self.kappa_matrix(self.datasets, test_dataset)  # Can be uncommented if test_dataset is different
        self.idx_support = np.squeeze(np.where(self.alpha > 1e-5))
        G = np.matmul(self.test_kappa[self.idx_support,:].T, np.expand_dims(self.alpha[self.idx_support],axis=1))
        print("calculating rho.\n")
        rho = self.compute_rho() 
        print(rho)
        decision = G-rho
        return decision.ravel(), np.sign(decision).ravel()
    
    def compute_rho(self):  
        K_support = self.train_kappa[self.idx_support][:, self.idx_support]
        # Calculate decision score for support vectors
        decision_values_sv = np.dot(K_support, self.alpha[self.idx_support])
        # Often averaged over SVs on the boundary, but averaging over all is common
        self.rho = np.mean(decision_values_sv)
        return self.rho

    def compute_group_gamma(self, group):
        pairwise_sq_dists = squareform(pdist(group, 'sqeuclidean'))
        median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])
        return 2 / (median_dist) if median_dist > 0 else 0  

    def find_best_gamma(self):
        gamma_values = Parallel(n_jobs=20)(
            delayed(self.compute_group_gamma)(group) for group in self.datasets
        )
        return np.median([g for g in gamma_values if g > 0])

    def kernel(self, X, Y):
        dist = np.sum(np.square(X)[:,np.newaxis,:],axis=2)-2*X.dot(Y.T)+np.sum(np.square(Y)[:,np.newaxis,:],axis=2).T
        return np.exp(-self.gamma*dist)

    def norm_squared(self, X):
        n = len(X)
        K = np.zeros(shape=(n,1))
        for i in range(n):
            K[i,0] = np.average(self.kernel(X[i],X[i]))
        return K
    
    def compute_entry(self, i, j, X, Y):
        k = self.kernel(X[i], Y[j])
        return (i, j, np.average(k))

    def kappa_matrix(self, X, Y):
        n1, n2 = len(X), len(Y)
        Kcross = np.zeros(shape=(n1,n2))
        results = Parallel(n_jobs=100)(
            delayed(self.compute_entry)(i, j, X, Y)
            for i in range(n1)
            for j in range(n2)
        )
        for i, j, val in results:
            Kcross[i, j] = val

        norm_sq1 = self.norm_squared(X)
        norm_sq2 = self.norm_squared(Y)
        normalizer = np.reciprocal(np.sqrt(norm_sq1*norm_sq2.T))
        Kcross = np.multiply(Kcross, normalizer)
        return Kcross