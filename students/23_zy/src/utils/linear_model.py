from utils.optimizer import GradientDescent


class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.optimizer = GradientDescent(lr, n_iters)
        self.theta = None

    def fit(self, X, y):
        self.theta = self.optimizer.fit(X, y)

    def predict(self, X):
        return self.optimizer.predict(X)