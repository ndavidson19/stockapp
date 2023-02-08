'''
This file will use Facebook Prophet to make predictions on time series data
'''

import numpy as np

from fbprophet import Prophet


class ProphetModel():
    '''
    Creates a Prophet class for time series forecasting
    '''
    def __init__(self):
        self.model = Prophet()

    def train(self, train_data, epochs=100, lr=0.01):
        self.model.fit(train_data)

    def predict(self, test_data):
        future = self.model.make_future_dataframe(periods=test_data.shape[0])
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-test_data.shape[0]:]