

import numpy as np


class PolyReg():
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model = np.polyfit(X, y, self.degree)

    def predict(self, X):
        return np.polyval(self.model, X)

    def score(self, X, y):
        return np.sum((self.predict(X) - y) ** 2
