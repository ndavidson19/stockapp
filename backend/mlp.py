import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLP(nn.Module):
    '''
    The idea is a simple MLP with 3 layers to predict the next value in the sequence
    '''

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
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
    
    def evaluate(self, test_data):
        predictions = self.predict(test_data)
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(len(test_data)):
                seq, labels = test_data[i]
                y_pred = predictions[i]
                if torch.argmax(y_pred) == torch.argmax(labels):
                    correct += 1
                total += 1
        return round(correct/total, 3)

if __name__ == '__main__':
    # create a toy dataset
    X = np.linspace(0, 50, 501)
    y = np.sin(X)
    df = pd.DataFrame(data=y, index=X, columns=['Sine'])
    # create train and test sets
    test_percent = 0.1
    test_point = np.round(len(df)*test_percent)
    train = df.iloc[:-test_point]
    test = df.iloc[-test_point:]
    # scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    # create sequences
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    # create sequences
    from utils import TimeseriesGenerator
    length = 2
    batch_size = 1
    train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
    test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=batch_size)
    # create model
    n_features = 1
    model = MLP(input_size=n_features, hidden_size=100, output_size=n_features)
    # train model
    model.train(train_generator, epochs=100, lr=0.001)
    # evaluate model
    model.evaluate(test_generator)
    # predict
    predictions = model.predict(test_generator)
    first_eval_batch = scaled_train[-length:]
    current_batch = first_eval_batch.reshape((1, length, n_features))
    for i in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
    true_predictions = scaler.inverse_transform(current_batch)[0]
    test['Predictions'] = true_predictions
    test.plot(figsize=(12,8))




