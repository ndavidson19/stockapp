from xgboost import XGBRegressor
import numpy as np



class XGBOOST():
    def __init__(self):
        self.model = XGBRegressor()
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)