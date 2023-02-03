import xgboost as xgb
import numpy as np



class XGBOOST():
    def __init__(self, config):
        self.model = xgb.XGBRegressor(
            n_estimators=config['xgb']["n_estimators"],
            max_depth=config['xgb']["max_depth"],
            subsample=config['xgb']["subsample"],
            min_child_weight=config['xgb']["min_child_weight"],
            objective="reg:squarederror",
            tree_method="hist"
            )
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)