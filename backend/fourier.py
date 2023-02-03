import numpy as np
import pylab as pl
from numpy import fft


class Fourier():
    '''
    performs a fourier transform on the input data
    '''
    def __init__(self, n=10):
        self.n = n
        self.model = None

    def fourierExtrapolation(self, x, n_predict):
        n = x.size
        n_harm = self.n                     # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)         # find linear trend in x
        x_notrend = x - p[0] * t        # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)              # frequencies
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key = lambda i: np.absolute(f[i]))
    
        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = self.fourierExtrapolation(X_train, self.n)

    def predict(self, X):
        return self.fourierExtrapolation(X, self.n)


