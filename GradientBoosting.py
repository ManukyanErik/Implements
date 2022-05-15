import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, ensemble_size=100, max_depth=1):
        self.T = ensemble_size 
        self.max_depth = max_depth

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.models = []

        for _ in range(self.T):
            clf = DecisionTreeRegressor(max_depth=self.max_depth)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_train -= y_pred
            self.models.append(clf)

    def predict(self, X_test):
        y_pred = sum([clf.predict(X_test) for clf in self.models])
        return y_pred
