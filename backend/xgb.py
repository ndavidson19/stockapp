import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance, plot_tree
import numpy as np
import matplotlib.pyplot as plt



class XGBOOST():
    def __init__(self, config):
        self.model = xgb.XGBRegressor(
            n_estimators=config["xgb"]["n_estimators"],
            max_depth=config["xgb"]["max_depth"],
            subsample=config["xgb"]["subsample"],
            min_child_weight=config["xgb"]["min_child_weight"],
            objective="reg:squarederror",
            tree_method="hist"
            )
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(np.vstack(X_train), np.vstack(y_train), eval_set = [np.vstack(X_val), np.vstack(y_val)], eval_metric = "mae", early_stopping_rounds = 50, verbose = False)
        print("XGBoost Model Trained")
        self.model.predict(X_train)
        self.model.predict(X_val)
        print("XGBoost Model Predicted")
        print("XGBoost Model Training Error: ", mean_absolute_error(y_train, self.model.predict(X_train)))
        print("XGBoost Model Validation Error: ", mean_absolute_error(y_val, self.model.predict(X_val)))
        print("XGBoost Model Training Error: ", mean_squared_error(y_train, self.model.predict(X_train)))
        print("XGBoost Model Validation Error: ", mean_squared_error(y_val, self.model.predict(X_val)))
        plot_importance(self.model)
        plot_tree(self.model)
        plt.show()

    def predict(self):
        return print('to be implemented')