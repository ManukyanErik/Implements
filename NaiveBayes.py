import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self, distribution='gaussian'):
        self.distribution = distribution
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.priors = self.calculate_priors() # 1. կետի հաշվարկներ
        self.mean, self.var = self.calculate_mean_var() # 2. կետի հաշվարկներ

    def calculate_priors(self):
            priors = np.unique(self.y_train, return_counts=True)[1]/len(self.y_train)
            
            return priors

    def calculate_mean_var(self):
            df = pd.DataFrame(self.X_train)
            df['y'] = self.y_train
            
            mean = df.groupby('y').mean()
            var = df.groupby('y').var()

            return mean, var

    def pdf(self, mean, var, point, distribution='gaussian'):
            if distribution == 'gaussian':
                lik = (1/var * np.sqrt(2*np.pi))*np.exp(-0.5*((point-mean)/var)**2)
            
            return lik

    def predict(self, X_test):
        priors = self.calculate_priors()
        
        result = []
        for point in X_test:
            class_probability = []
            for i in range(self.mean.shape[0]):
                point_probability = []
                for j in range(self.mean.shape[1]):
                    lk = self.pdf(mean = self.mean.iloc[i, j], var = self.var.iloc[i, j], point=point[j])
                    if lk == 0.0:
                        lk = 1/point.shape[0]
                    point_probability.append(lk)

                class_probability.append(np.prod(point_probability) *  priors[i])
            result.append(np.argmax(class_probability))

        return result      