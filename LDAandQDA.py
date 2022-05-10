import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.priors = self.calculate_priors()
        self.means = self.calculate_means()
        self.cov = self.calculate_covariance_matrix()

    def calculate_priors(self):
        priors = np.unique(self.y_train, return_counts=True)[1]/len(self.y_train)
        return priors


    def calculate_means(self):
        means = []
        self.y_train = np.reshape(self.y_train, ( -1, 1))
        
        arr = np.hstack((self.X_train, self.y_train))
        for i in np.unique(self.y_train):
            means.append(arr[arr[:, -1] == i].mean(axis=0)[:-1])
        return np.array(means)


    def calculate_covariance_matrix(self):
        cov = []
        # cov2 = []
        self.y_train = np.reshape(self.y_train, ( -1, 1))
        arr = np.hstack((self.X_train, self.y_train))

        for k in np.unique(self.y_train):
            x = arr[arr[:, -1] == k][:, :-1]
            nr_data = x.shape[0]

            cov.append((1 / (nr_data-2))* (x - x.mean(axis=0)).T @ (x-x.mean(axis=0)) )
            # cov2.append(np.cov(x, rowvar=False, ddof=2))
        return np.linalg.inv(sum(cov))
        

    def predict(self, X_test):
        SIGMA = []
        for k in np.unique(self.y_train):
            sigma = X_test @ self.cov @ self.means[k] - 0.5*(self.means[k].T @self.cov@self.means[k]) + np.log(self.priors[k])
            SIGMA.append(sigma)
        return np.argmax(np.array(SIGMA), axis=0)
            