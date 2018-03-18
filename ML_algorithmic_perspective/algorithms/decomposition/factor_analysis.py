import numpy as np
import scipy.spatial as sp


class FactorAnalysis(object):
    """
    Finds hidden (latent) factors, which can describe distribution of X.
    Factor Analysis basically uses EM-algorithm.
    """
    
    def __init__(self, random_state=13):
        np.random.seed(random_state)


    def fit_transform(self, X, dim, n_iter=100, tol=1e-4):
        Ndata, N = X.shape
        X_cnt = X-X.mean(axis=0)
        C = np.cov(X_cnt.T)
        Cd = C.diagonal()
        Psi = Cd
        scaling = np.linalg.det(C)**(1./N)

        W = np.random.normal(loc=0, scale=np.sqrt(scaling/dim), size=(N,dim))
        oldL = -np.inf

        for i in range(n_iter):
            # E-step
            A = np.dot(W,W.T) + np.diag(Psi)
            logA = np.log(np.abs(np.linalg.det(A)))
            A = np.linalg.inv(A)

            WA = np.dot(W.T,A)
            WAC = np.dot(WA,C)
            Exx = np.eye(dim) - np.dot(WA,W) + np.dot(WAC,WA.T) 

            # M-step
            W = np.dot(WAC.T, np.linalg.inv(Exx))
            Psi = Cd - (np.dot(W,WAC)).diagonal()

            tAC = (A*C.T).sum()

            L = -N/2*np.log(2.*np.pi) -0.5*logA - 0.5*tAC
            self._logL = L
            if L - oldL < tol:
                break
            oldL = L
        A = np.linalg.inv(np.dot(W,W.T) + np.diag(Psi))
        Ex = np.dot(A.T,W)
        self._n_iter=n_iter
        self._tol=tol
        self._dim = dim        
        return np.dot(X_cnt, Ex)
