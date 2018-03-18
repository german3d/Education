import numpy as np
from sklearn.cluster import KMeans


class RBF(object):
    
    """This algorithm is suitable for regression problems only"""
    
    def __init__(self, num_neurons, use_kmeans=True, normalize=True):
        self.num_neurons = num_neurons
        self.use_kmeans = use_kmeans
        self.normalize = normalize
        self._is_fitted = False
    

    def activate(self, X):
        self.hidden = np.ones((X.shape[0], self.num_neurons+1)) # with extra bias term
        for i in range(self.num_neurons):
            self.hidden[:,i] = np.exp(-np.sum((X-self.centroids[i,:])**2,axis=1) / (2*self.sigma**2))
        if self.normalize:
            self.hidden[:,:-1] /= self.hidden[:,:-1].sum(axis=1, keepdims=True)
    

    def fit(self, X, y):
        y = RBF.resize_data(y)
        X = RBF.resize_data(X)
        # calculating centroids from input data
        if self.use_kmeans:
            kmeans = KMeans(n_clusters=self.num_neurons);
            kmeans.fit(X);
            centroids = kmeans.cluster_centers_            
        else:
            centroids = X[np.random.choice(X.shape[0], self.num_neurons, replace=False), :]
        self.centroids = centroids
        
        # activations of hidden RBF neurons        
        self.sigma = (X.max(axis=0)-X.min(axis=0)).max() / np.sqrt(2*self.num_neurons)
        self.activate(X)        
        
        # get the weights for RBFs making use of pseudo-inverse matrix operations
        self.W = np.dot(np.linalg.pinv(self.hidden), y)
        self._is_fitted = True
    

    def predict(self, X):
        X = RBF.resize_data(X)
        if self._is_fitted is False:
            raise Exception("model must be fitted before predicting")
        self.activate(X)
        preds = np.dot(self.hidden, self.W)
        return preds
    

    @staticmethod
    def resize_data(data):
        if len(data.shape)==1:
            return data.reshape((data.shape[0],-1))
        else:
            return data
