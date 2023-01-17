import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline import DataPipeline
from ensemble import EnsembleModel


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
    }
}

# load in the data
    
data_pipeline = DataPipeline(config)

training_data = data_pipeline.train_dataset
validation_data = data_pipeline.val_dataset
training_dataloader = data_pipeline.train_dataloader
validation_dataloader = data_pipeline.val_dataloader
print("--------------------------------")
print("Train data shape", training_data.x.shape, training_data.y.shape)
print("Validation data shape", validation_data.x.shape, validation_data.y.shape)
print("--------------------------------")
print('Training data X:' , training_data.x)
print('Training data Y:' , training_data.y)

# Define the ensemble model architecture and hyperparameters
ensemble_model = EnsembleModel(config)

# Train the ensemble model using supervised learning
ensemble_model.fit(training_data, validation_data)
#ensemble_model.predict(validation_data)

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
