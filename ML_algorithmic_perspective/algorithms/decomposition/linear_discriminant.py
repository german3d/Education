import numpy as np
import scipy.spatial as sp


class LDA(object):
    """
    Linear Discriminant algorithm for dimensionality reduction.
    Note: LDA is able to retrieve only (K-1) features, where K - number of classes.
    """
    
    def __init__(self):
        self._eigenvals = None
        self._eigenvecs = None
        self._variance_explained = None
    
    
    def fit(self, X, y):
        classes_labels = np.unique(y)
        N = X.shape[0]
        S = N*np.cov(X.T)
        S_w = np.zeros(shape=(X.shape[1], X.shape[1]))
        S_b = np.zeros(shape=(X.shape[1], X.shape[1]))
        for c in classes_labels:
            X_c = X[y==c,:]
            N_c = X_c.shape[0]
            S_c = np.cov(X_c.T)
            S_w += (N_c-1)*S_c
        S_b = S - S_w
        eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
        
        idxs_sorted = np.argsort(eigenvals)[::-1]
        self._eigenvals = eigenvals[idxs_sorted]
        self._eigenvecs = -eigenvecs[idxs_sorted,:]
        self._variance_explained = self._eigenvals / self._eigenvals.sum()
    
    
    def transform(self, X, dims):
        if dims > X.shape[1]:
            raise ValueError("Invalid dimensionality")
        W = self._eigenvecs[:,:dims]
        return X.dot(W)
