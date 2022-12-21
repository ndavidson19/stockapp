import numpy as np


class Fourier():
    '''
    performs a fourier transform on the input data
    '''
    def __init__(self, n=1):
        self.n = n
        self.model = None

    def fit(self, X_train, y_train):
        self.model = np.fft.fft(X_train, n=self.n)

    def predict(self, X):
        return np.fft.ifft(X, n=self.n)


