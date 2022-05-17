import numpy as np

class LinearRegression:
    def __init__(self, lr=0.1,  max_iter=100, metric_change=0.001):
        self.lr = lr
        self.max_iter = max_iter
        self.metric_change = metric_change
    
    def gradient(self, x, w, y):
        derivative = []
        for i in range(w.shape[0]):
            derivative_i = 2 * np.mean(((x @ w.T) - y) * x[:, i])
            derivative.append(derivative_i)
        return np.array(derivative)

    def update_weights(self, x,  y):
        derivatives = self.gradient(x, self.w, y)
        self.w = self.w - self.lr * derivatives
        return self.w

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        ones = np.ones(self.X_train.shape[0]).reshape(-1, 1)
        x = np.hstack((ones, self.X_train)) 
        self.w = np.random.random(x.shape[1])
        
        loss = np.mean(((x @ self.w.T) - self.y_train)**2)

        for _ in range(self.max_iter):
            self.update_weights(x, self.y_train)
            new_loss = np.mean(((x @ self.w.T) - self.y_train)**2)
            temp_change = loss - new_loss
            if temp_change < 0:
                raise ValueError('Bad learning rate or lambda, loss function increases')
            loss = new_loss
            if temp_change < self.metric_change:
                break
        
    def predict(self, X_test):
        ones = np.ones(X_test.shape[0]).reshape(-1, 1)
        X_test = np.hstack((ones, X_test))
        y_pred = X_test @ self.w.T
        
        return y_pred