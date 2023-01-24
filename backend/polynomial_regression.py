

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PolyReg():
    def __init__(self, degree = 3):
        self.degree = degree
        self.model = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model = np.polyfit(X, y, self.degree)

    def predict(self, X):
        return np.polyval(self.model, X)

    def eval(self, X, y):
        print("Mean Squared Error: ", np.mean(sum((self.predict(X) - y) ** 2)))
        self.plot(X, y)

    def plot(self, X, y):
        plt.plot(X, y, 'o')
        plt.plot(X, self.predict(X), 'r')
        plt.show()

