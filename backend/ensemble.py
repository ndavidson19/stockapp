from arima import ARIMA
from lstm import LSTMModel
from xgb import XGBOOST
from svm import SVM
from fourier import Fourier
from polynomial_regression import PolyReg
from sklearn.ensemble import StackingRegressor
from data_pipeline import DataPipeline
import torch 

class EnsembleModel:
  def __init__(self, config):

    self.ARIMA = ARIMA(config)
    self.LSTM = LSTMModel()
    self.XGB = XGBOOST(config)
    #self.model4 = SVM()
    self.FOURIER = Fourier()
    self.model6 = PolyReg()
    self.config = config
    
  def fit(self, training_data, validation_data, training_data_1d, validation_data_1d):
    # Train each individual model on the training data


    #self.ARIMA.fit(training_data_1d.x) # no y_train for ARIMA
    #self.LSTM.fit(model = self.LSTM, X_train = training_data.x, y_train = training_data.y, X_val=validation_data.x, y_val = validation_data.y)
    self.XGB.fit(X_train  = training_data_1d.x, y_train = training_data_1d.y, X_val = validation_data_1d.x, y_val = validation_data_1d.y)
    #self.model4.fit(training_data.x, training_data.y)
    self.FOURIER.fit(training_data_1d.x, training_data_1d.y)
    self.model6.fit(training_data_1d.x, training_data_1d.y)

    
  def predict(self, validation_data, validation_data_1d, validation_dataloader, x_unseen):
    # Use each individual model to make predictions on the input data
    y_pred1 = self.ARIMA.forecast(validation_data_1d, n_steps=20)
    y_pred2 = self.LSTM.predict(test_dataloader=validation_dataloader, x_unseen=x_unseen)
    y_pred3 = self.XGB.predict()
    #y_pred4 = self.model4.predict(X)
    y_pred5 = self.FOURIER.predict(validation_data.y)
    y_pred6 = self.model6.eval(validation_data_1d.x, validation_data_1d.y)
    
    # Combine the predictions of the individual models in a way that suits your needs
    #y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6) / 6

    #y_new = StackingRegressor(estimators=[('arima', self.model1), ('lstm', self.model2), ('xgboost', self.model3), ('svm', self.model4), ('fourier', self.model5), ('polyreg', self.model6)], final_estimator=LinearRegression())

   

    return y_pred1, y_pred2, y_pred5, y_pred6

