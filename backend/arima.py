import numpy as np
from statsmodels.tsa.arima_model import ARIMA



class ARIMA():
    '''
    performs an ARIMA model on the input data
    '''
    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def fit(self, X_train, y_train):
        self.model = ARIMA(y_train, order=(self.p, self.d, self.q))
        self.model = self.model.fit(disp=0)

    def predict(self, X):
        return self.model.forecast(steps=len(X))[0]

