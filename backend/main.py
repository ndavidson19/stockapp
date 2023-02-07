import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline import DataPipeline
from ensemble import EnsembleModel
from nnet import NeuralNetwork

config = {
    "alpha_vantage": {
        "key": "COR04RQ1Z1P74ETF", 
        "symbol": "SPY",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    }, 
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 5,
        "lstm_size": 128,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
        "scheduler_gamma": 0.1,
        "model_path": "model.pth",
    },
    "arima_para": {
        "p": range(2),
        "d": range(2),
        "q": range(2),
        "seasonal_para": 2,
    },
    "xgb": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.5,
        "colsample_bytree": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "min_child_weight": 1,
    },
}

# load in the data
    
data_pipeline = DataPipeline(config)

training_data = data_pipeline.train_dataset
training_data_1d = data_pipeline.train_dataset_1d
validation_data = data_pipeline.val_dataset
validation_data_1d = data_pipeline.val_dataset_1d
training_dataloader = data_pipeline.train_dataloader
validation_dataloader = data_pipeline.val_dataloader
x_unseen = data_pipeline.data_x_unseen
print("--------------------------------")
print("Train data shape", training_data.x.shape, training_data.y.shape)
print("Train 1D data shape", training_data_1d.x.shape, training_data_1d.y.shape)
print("Validation data shape", validation_data.x.shape, validation_data.y.shape)
print("Validation 1D data shape", validation_data_1d.x.shape, validation_data_1d.y.shape)

print("--------------------------------")
print('Training data X:' , training_data.x)
print('Training data Y:' , training_data.y)
print('Training data 1D X:' , training_data_1d.x)
print('Training data 1D Y:' , training_data_1d.y)
print('Validation data X:' , validation_data.x)
print('Validation data Y:' , validation_data.y)
print('Validation data 1D X:' , validation_data_1d.x)
print('Validation data 1D Y:' , validation_data_1d.y)

# Define the ensemble model architecture and hyperparameters
ensemble_model = EnsembleModel(config)

# Train the ensemble model using supervised learning
ensemble_model.fit(training_data, validation_data, training_data_1d=training_data_1d, validation_data_1d=validation_data_1d)
ensemble_model.predict(validation_data, validation_data_1d = validation_data_1d, validation_dataloader = validation_dataloader, x_unseen = x_unseen)

# Define the neural network architecture and hyperparameters
neural_network = NeuralNetwork(...)

# Set up the RL algorithm
rl_algorithm = RLAgent(...)

# Train the RL agent using the neural network to guide the reward
for t in range(T):
  # Get the current state of the system
    state = get_state()

  # Select an action
    action = select_action()

# Use the ensemble model to predict the next state

    next_state = ensemble_model.predict(state, action)
  # Use the neural network to approximate the reward
    reward = neural_network.predict(next_state, action)

  # Update the RL algorithm using the reward from the neural network
    rl_algorithm.update(state, action, reward, next_state)

# Use the trained RL agent to make decisions in the real environment
while True:
  state = get_state()
  action = rl_algorithm.predict(state)
  take_action(action)
