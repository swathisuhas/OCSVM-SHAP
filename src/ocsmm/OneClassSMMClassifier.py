import numpy as np
import cvxopt 
from scipy.spatial.distance import pdist, squareform

class OneClassSMMClassifier:
    def __init__(self, nu):
        self.nu = nu
        self.datasets = None
        self.gamma_x = None
        self.gamma_d = None
        
    def fit(self, datasets):
        self.datasets = datasets
        self.gamma_x = self.find_best_gamma_x()
        self.gamma_d = self.find_best_gamma_d()
        num_groups = len(self.datasets)
        kappa = self.kappa_matrix(self.datasets, self.datasets, self.gamma_d)
        ones = np.ones(shape=(num_groups,1))
        zeros = np.zeros(shape=(num_groups,1))
        P = cvxopt.matrix(kappa)
        q = cvxopt.matrix(zeros)
        G = cvxopt.matrix(np.vstack((-np.identity(num_groups), np.identity(num_groups))))
        C = 1./(self.nu*num_groups)
        h = cvxopt.matrix(np.vstack((zeros, C*ones)))
        A = cvxopt.matrix(ones.T)
        b = cvxopt.matrix(1.0)
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')
        self.alpha =np.ravel(solution['x'])
        
    def predict(self, test_dataset):
        self.kappa = self.kappa_matrix(self.datasets, test_dataset, self.gamma_d)
        self.support_index = np.squeeze(np.where(self.alpha > 1e-5))
        G = np.matmul(self.kappa[self.support_index,:].T, np.expand_dims(self.alpha[self.support_index],axis=1))
        rho = self.compute_rho() 
        decision = G-rho
        return decision.ravel(), np.sign(decision).ravel()
    
    def compute_rho(self):
        valid_support_index = np.where((self.alpha > 1e-5) & (self.alpha < (1 / (self.nu * len(self.datasets)))))[0]
        support_lists = [self.datasets[i] for i in valid_support_index]
        kappa_support = self.kappa_matrix(self.datasets, support_lists, self.gamma_d)
        rho = np.mean(np.sum(self.alpha[:, None] * kappa_support, axis=0))
        return rho

    def find_best_gamma_x(self):
        gamma_x_values = []
        for group in self.datasets: 
            pairwise_sq_dists = squareform(pdist(group, 'sqeuclidean'))  # Squared Euclidean distances
            median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  # Ignore zero distances
            gamma_x_values.append( 1 / (median_dist))
        return np.mean(gamma_x_values) 
    
    def find_best_gamma_d(self):
        group_distances = []
        for i in range(len(self.datasets)):
            for j in range(i + 1, len(self.datasets)):
                dist = self.compute_mmd_squared(self.datasets[i], self.datasets[j], self.gamma_x)
                group_distances.append(dist)
        return  1 / (np.median(group_distances))
    
    def compute_mmd_squared(self,X, Y, gamma):
        n, m = len(X), len(Y)
        K_XX = self.kernel(X, X, gamma) 
        K_YY = self.kernel(Y, Y, gamma) 
        K_XY = self.kernel(X, Y, gamma)  
        mmd2 = (K_XX.sum() / (n * n)) + (K_YY.sum() / (m * m)) - (2 * K_XY.sum() / (n * m))
        return mmd2

    def kernel(self, X, Z, gamma):
        dists_2 = np.sum(np.square(X)[:,np.newaxis,:],axis=2)-2*X.dot(Z.T)+np.sum(np.square(Z)[:,np.newaxis,:],axis=2).T
        k_XZ = np.exp(-gamma*dists_2)
        return k_XZ

    def measureNormSquare(self, S, gamma):
        n = len(S)
        K = np.zeros(shape=(n,1))
        for i in range(n):
            K[i,0] = np.average(self.kernel(S[i],S[i], gamma))
        return K

    def kappa_matrix(self, S1, S2, gamma):
        n1, n2 = len(S1), len(S2)
        Kcross = np.zeros(shape=(n1,n2))
        for i in range(n1):
            for j in range(n2):
                k=self.kernel(S1[i],S2[j], self.gamma_x)
                Kcross[i,j] = np.average(k)
        normK1 = self.measureNormSquare(S1, gamma)
        normK2 = self.measureNormSquare(S2, gamma)
        normalizer = np.sqrt(normK1*normK2.T)
        Kcross = np.multiply(Kcross, np.reciprocal(normalizer))
        return Kcross