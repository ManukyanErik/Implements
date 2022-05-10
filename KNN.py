import numpy as np

from scipy.spatial import distance



class KNeighborsRegressor:
    def __init__(self, n_neighbors=5,  metric='minkowski'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def __calculate_dist(self, i, j):
        if self.metric == 'minkowski':
            return distance.minkowski(i, j, 2)
        if self.metric == 'euclidean':
            pass

    def fit_predict(self, X_train, y_train, X_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        
        dist_matrix = []
        for i in X_test:
            dist = []
            for j in X_train:
                dist.append(self.__calculate_dist(i, j))
            dist_matrix.append(dist)
            dist = []
        
        index = []
        for l in dist_matrix:
            idx = []
            sort_l = sorted(l)
            for i in sort_l[:self.n_neighbors]:
                idx.append(l.index(i))
            index.append(idx)

        pred = [[y_train[j] for j in i] for i in index]

        return [np.mean(i) for i in pred]