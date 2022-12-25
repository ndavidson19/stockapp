from arima import ARIMA
from lstm import LSTM
from xgb import XGBOOST
from svm import SVM
from fourier import Fourier
from polynomial_regression import PolyReg
from sklearn.ensemble import StackingRegressor


class EnsembleModel:
  def __init__(self, config):

    self.model1 = ARIMA()
    self.model2 = LSTM(config)
    self.model3 = XGBOOST()
    self.model4 = SVM()
    self.model5 = Fourier()
    self.model6 = PolyReg()
    
  def fit(self, X_train, y_train):
    # Train each individual model on the training data
    self.model1.fit(X_train, y_train)
    self.model2.fit(X_train, y_train)
    self.model3.fit(X_train, y_train)
    self.model4.fit(X_train, y_train)
    self.model5.fit(X_train, y_train)
    self.model6.fit(X_train, y_train)

    
  def predict(self, X):
    # Use each individual model to make predictions on the input data
    y_pred1 = self.model1.predict(X)
    y_pred2 = self.model2.predict(X)
    y_pred3 = self.model3.predict(X)
    y_pred4 = self.model4.predict(X)
    y_pred5 = self.model5.predict(X)
    y_pred6 = self.model6.predict(X)
    
    # Combine the predictions of the individual models in a way that suits your needs
    y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6) / 6

    y_new = StackingRegressor(estimators=[('arima', self.model1), ('lstm', self.model2), ('xgboost', self.model3), ('svm', self.model4), ('fourier', self.model5), ('polyreg', self.model6)], final_estimator=LinearRegression())

    return y_pred, y_new

