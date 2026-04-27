import numpy as np


class GradientDescent:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)
        self.loss_history = []

        for i in range(self.n_iters):
            y_pred = X.dot(self.theta)

            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            gradient = (-2 / n_samples) * X.T.dot(y - y_pred)
            self.theta -= self.lr * gradient

        return self.theta

    def predict(self, X):
        return X.dot(self.theta)