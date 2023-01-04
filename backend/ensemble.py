from arima import ARIMA
from lstm import LSTM
from xgb import XGBOOST
from svm import SVM
from fourier import Fourier
from polynomial_regression import PolyReg
from sklearn.ensemble import StackingRegressor
from data_pipeline import DataPipeline


class EnsembleModel:
  def __init__(self, config):

    #self.model1 = ARIMA(config)
    self.model2 = LSTM(config)
    #self.model3 = XGBOOST()
    #self.model4 = SVM()
    #self.model5 = Fourier()
    #self.model6 = PolyReg()
    
  def fit(self, training_data):
    # Train each individual model on the training data


    #self.model1.fit(training_data) # no y_train for ARIMA
    self.model2.fit(training_data)
    #self.model3.fit(training_data.x, training_data.y)
    #self.model4.fit(training_data.x, training_data.y)
    #self.model5.fit(training_data.x, training_data.y)
    #self.model6.fit(training_data.x, training_data.y)

    
  def predict(self, X):
    # Use each individual model to make predictions on the input data
    #y_pred1 = self.model1.predict(X)
    y_pred2 = self.model2.predict(X)
    #y_pred3 = self.model3.predict(X)
    #y_pred4 = self.model4.predict(X)
    #y_pred5 = self.model5.predict(X)
    #y_pred6 = self.model6.predict(X)
    
    # Combine the predictions of the individual models in a way that suits your needs
    #y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6) / 6

    #y_new = StackingRegressor(estimators=[('arima', self.model1), ('lstm', self.model2), ('xgboost', self.model3), ('svm', self.model4), ('fourier', self.model5), ('polyreg', self.model6)], final_estimator=LinearRegression())

    y_pred = y_pred2

    return y_pred

