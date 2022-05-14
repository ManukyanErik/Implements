import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, ensemble_size=10, learning_algorithm="DecisionTree"):
        self.T = ensemble_size 
        self.A = learning_algorithm

    def confidence_model(self, w, y_pred, y_train):
        eps = sum(w[y_pred != y_train]) # weighted error
        return 0.5*np.log((1-eps)/eps )

    def update_weights(self, w, alfa, y_train, y_pred):
        w = w * np.exp(-alfa*y_train*y_pred) 
        w = w / sum(w)
        return w

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        w = np.full(X_train.shape[0], 1/X_train.shape[0])
        
        self.confidences = []
        self.models = []

        for _ in range(self.T):
            if self.A == "DecisionTree":
                clf = DecisionTreeClassifier(random_state=0, max_depth=1)
            clf.fit(X_train, y_train, sample_weight=w)
            y_pred = clf.predict(X_train)

            alfa = self.confidence_model(w, y_pred, y_train) # confidence for this model
            w = self.update_weights(w, alfa, y_train, y_pred)
            
            self.confidences.append(alfa)
            self.models.append(clf)
            

    def predict(self, X_test):
        y_pred =  sum([alfa * clf.predict(X_test) for alfa, clf in zip(self.confidences, self.models)])
        return [1 if i > 0 else -1 for i in y_pred]
