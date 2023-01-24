import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym


class NeuralNetwork(nn.Module):
    '''
    Neural Network with the purpose of predicting the reward for the RL Agent
    The Reward in this case is the return of the portfolio
    This is the neural network that will be trained using supervised learning from the simulated ensemble model
    '''
    def __init__(self, s_size, a_size, h_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, state, action):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.from_numpy(action).float().unsqueeze(0).to(device)
        reward = self.forward(state, action)
        return reward
    
    def fit(self, X, y):
        '''
        Train the neural network using supervised learning
        '''
        # Define the loss function and the optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        
        # Train the neural network
        for t in range(1000):
            # Forward pass
            y_pred = self.forward(X)
            loss = criterion(y_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if t % 100 == 99:
                print(t, loss.item())

