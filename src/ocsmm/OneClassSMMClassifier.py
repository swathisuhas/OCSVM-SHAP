import numpy as np
import cvxopt 
from scipy.spatial.distance import pdist, squareform

class OneClassSMMClassifier:
    def __init__(self, nu):
        self.nu = nu
        self.datasets = None
        self.gamma = None
        
    def fit(self, datasets):
        self.datasets = datasets
        self.gamma = self.find_best_gamma()

        num_groups = len(self.datasets)
        kappa = self.kappa_matrix(self.datasets, self.datasets)
        ones = np.ones(shape=(num_groups,1))
        zeros = np.zeros(shape=(num_groups,1))
        P = cvxopt.matrix(kappa)
        q = cvxopt.matrix(zeros)
        G = cvxopt.matrix(np.vstack((-np.identity(num_groups), np.identity(num_groups))))
        C = 1.0/(self.nu*num_groups)
        h = cvxopt.matrix(np.vstack((zeros, C*ones)))
        A = cvxopt.matrix(ones.T)
        b = cvxopt.matrix(1.0)
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')
        self.alpha = np.ravel(solution['x'])
        
    def predict(self, test_dataset):
        self.kappa = self.kappa_matrix(self.datasets, test_dataset)
        self.idx_support = np.squeeze(np.where(self.alpha > 1e-5))
        G = np.matmul(self.kappa[self.idx_support,:].T, np.expand_dims(self.alpha[self.idx_support]*self.nu*len(self.kappa),axis=1))
        rho = self.compute_rho() 
        decision = G-rho
        return decision.ravel(), np.sign(decision).ravel()
    
    def compute_rho(self):  
        K_support = self.kappa[self.idx_support][:, self.idx_support]
        # rho = np.matmul(self.kappa[self.idx_support][:, self.idx_support].T, np.expand_dims(self.alpha,axis=1))        
        rho = np.dot(self.alpha[self.idx_support]*self.nu*len(self.kappa), K_support)
        print(rho)
        return np.mean(rho)
    
    def find_best_gamma(self):
        gamma_values = []
        for group in self.datasets: 
            pairwise_sq_dists = squareform(pdist(group, 'sqeuclidean')) 
            median_dist = np.median(pairwise_sq_dists[pairwise_sq_dists > 0])  
            gamma_values.append( 1 / (median_dist))
        return np.median(gamma_values) 

    def kernel(self, X, Y):
        dist = np.sum(np.square(X)[:,np.newaxis,:],axis=2)-2*X.dot(Y.T)+np.sum(np.square(Y)[:,np.newaxis,:],axis=2).T
        return np.exp(-self.gamma*dist)

    def norm_squared(self, X):
        n = len(X)
        K = np.zeros(shape=(n,1))
        for i in range(n):
            K[i,0] = np.average(self.kernel(X[i],X[i]))
        return K

    def kappa_matrix(self, X, Y):
        n1, n2 = len(X), len(Y)
        Kcross = np.zeros(shape=(n1,n2))
        for i in range(n1):
            for j in range(n2):
                k=self.kernel(X[i],Y[j])
                Kcross[i,j] = np.average(k)
        norm_sq1 = self.norm_squared(X)
        norm_sq2 = self.norm_squared(Y)
        normalizer = np.reciprocal(np.sqrt(norm_sq1*norm_sq2.T))
        Kcross = np.multiply(Kcross, normalizer)
        return Kcross