import numpy as np
import scipy.spatial as sp


class PCA(object):
    """
    PCA algorithm for dimensionality reduction.
    """

    def __init__(self, normalize=False):
        self._eigenvals = None
        self._eigenvecs = None
        self._variance_explained = None
        self._normalize = normalize
    
    
    def fit(self, X):
        X_cnt = X - X.mean(axis=0)
        C_cnt = np.cov(X_cnt.T) # covariance matrix
        eigenvals, eigenvecs = np.linalg.eig(C_cnt)
        
        if self._normalize:
            X_dim = X.shape[1]
            eigenvecs = eigenvecs / np.sqrt(np.repeat(eigenvals.reshape(1,-1), repeats=X_dim, axis=0))
        idxs_sorted = np.argsort(eigenvals)[::-1]
        self._eigenvals = eigenvals[idxs_sorted]
        self._eigenvecs = eigenvecs[idxs_sorted,:]
        self._variance_explained = self._eigenvals / self._eigenvals.sum()
    
    
    def transform(self, X, dims):
        if dims > X.shape[1]:
            raise ValueError("Invalid dimensionality")
        W = self._eigenvecs[:,:dims]
        return X.dot(W)




class KernelPCA(object):
    """
    Kernel PCA algorithm for dimensionality reduction.
    """
    
    def __init__(self, kernel="rbf", gamma=None):
        self._eigenvals = None
        self._eigenvecs = None
        self._variance_explained = None
        self._kernel = kernel
        self._gamma = gamma
        
    
    def fit(self, X):        
        if self._kernel != "rbf":
            raise NotImplementedError("Only 'rbf' kernel is available")
        
        N, X_dim = X.shape
        self._gamma = 1./X_dim if self._gamma is None else self._gamma
        
        X_dist_flat = sp.distance.pdist(X, metric="sqeuclidean")
        X_dist_matr = sp.distance.squareform(X_dist_flat)
        K = np.exp(-self._gamma * X_dist_matr) # kernel matrix 
        one_n = np.ones((N,N)) / N
        K_cnt = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) # centering
        eigenvals, eigenvecs = np.linalg.eig(K_cnt)
        idxs_sorted = np.argsort(eigenvals)[::-1]
        self._eigenvals = eigenvals[idxs_sorted]
        self._eigenvecs = -eigenvecs[idxs_sorted,:]
        self._variance_explained = self._eigenvals / self._eigenvals.sum()
    
    
    def transform(self, dims):
        return self._eigenvecs[:,:dims]
