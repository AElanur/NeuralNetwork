import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        print("Initial Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(x, y):
                y_pred = self.predict(xi)
                update = self.eta * (target - y_pred)
                self.w_[0] += update
                self.w_[1:] += update * xi
                error += int(update != 0.0)
            self.errors_.append(error)
        return self
