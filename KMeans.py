import numpy as np
from sklearn.metrics import  pairwise_distances # , pairwise_distances_argmin

class KMeans:
    def __init__(self, k, method='random', max_iter=300):
        self.k = k 
        self.method = method
        self.max_iter = max_iter
        
    def init_centers(self, X):
        if self.method == 'random':
            return X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        if self.method == 'k-means++':
            centroids = [X[np.random.choice(len(X))]]
            while len(centroids)<self.k:
                distances = pairwise_distances(X, centroids).min(axis=1)
                new_i = np.random.choice(len(X), p=(distances**2).reshape(-1)/sum(distances**2))
                centroids.append(X[new_i])
            
            return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self.init_centers(X)
        for _ in range(self.max_iter):
            clusters = self.expectation(X, self.centroids)
            new_centroids = self.maximization(X, clusters)
            if np.array_equal(new_centroids, self.centroids):
                break
            self.centroids = new_centroids     
    
    def expectation(self, X, centroids):
        clusters = [[] for _ in range(self.k)]
        distance = [min([[i , index_j, sum(map(lambda x: x**2, np.array(j)-np.array(i)))**0.5] for index_j, j in enumerate(centroids)], key= lambda x: x[2]) for i in X]      
        
        for i in range(len(distance)):
            clusters[distance[i][1]].append(list(distance[i][0]))
        
        '''
        return pairwise_distances_argmin(X, centroids)
        '''
        return clusters 

    
    def maximization(self, X, clusters):
        new_centroids = [[np.mean([clusters[j][i][z] for i in range(len(clusters[j]))]) for j in range(len(clusters))] for z in range(2)]
        new_centroids = [[i, j] for i, j in zip(new_centroids[0], new_centroids[1])]
        
        '''
        return np.array([X[clusters==i].mean(axis=0) for i in range(self.k)])
        '''
        return new_centroids    
    
    def predict(self, X):
        predictions_clusters = self.expectation(X, self.centroids)
        y_kmeans = []
        for i in X:
            for j in range(len(predictions_clusters)):
                if list(i) in predictions_clusters[j]:
                    y_kmeans.append(j)
        '''
        return self.expectation(X, self.centroids)
        '''
        return y_kmeans