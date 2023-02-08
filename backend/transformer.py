'''
This file will contain a transformer based model that can be used for probabilistic time series forecasting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class Transformer(nn.Module):
    '''
    Transformer Base class for time series forecasting
    '''
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Transformer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.attention = nn.MultiheadAttention(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, train_data, epochs=100, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for i in range(epochs):
            for seq, labels in train_data:
                optimizer.zero_grad()
                y_pred = self.forward(seq)
                single_loss = loss_fn(y_pred, labels)
                single_loss.backward()
                optimizer.step()
            if i%10 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    
    def predict(self, test_data):
        predictions = []
        for seq, labels in test_data:
            y_pred = self.forward(seq)
            predictions.append(y_pred)
        return predictions
