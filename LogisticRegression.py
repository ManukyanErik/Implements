import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, max_iter=500, metric_change=0.001):
        self.lr = lr
        self.max_iter = max_iter
        self.metric_change = metric_change

    def sigmoid(self, x):
        sigmoid = 1 / (1 + np.exp(-x @ self.w.T))
        return sigmoid

    def loss_function(self, x, y, epsilon=1e-5):
        loss = np.sum( (1 - y)*np.log(1 - self.sigmoid(x) + epsilon) - y*np.log(self.sigmoid(x) + epsilon) )
        return loss

    def gradient(self, x, y):
        grad = []
        for i in range(x.shape[1]):
            grad.append(np.sum(x[:, i]*(self.sigmoid(x) - y)))
        return np.array(grad)

    def update_weights(self, x, y):
        grad = self.gradient(x, y)
        self.w = self.w - self.lr * grad
    

    def fit(self, X_train, y_train):
        ones = np.ones(X_train.shape[0]).reshape(-1, 1)
        x = np.hstack((ones, X_train)) 
        self.w = np.random.random(x.shape[1])
        self.L = self.loss_function(x, y_train)
        
        for _ in range(self.max_iter):
            self.update_weights(x, y_train)
            temp_change = np.abs(self.L - self.loss_function(x, y_train))
            self.L = self.loss_function(x, y_train)

            if temp_change < self.metric_change:
                break

        return self.w


    def predict(self, X_test): 
        ones = np.ones(X_test.shape[0]).reshape(-1, 1)
        x = np.hstack((ones, X_test))
         
        result = 1.0 / (1.0 + np.exp(-(x @ self.w.T)))
        return np.array(list(map(lambda i: round(i), result)))