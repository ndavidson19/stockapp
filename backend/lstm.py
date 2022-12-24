import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        


        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

class LSTM(LSTMModel):
    def __init__(self):
        super().__init__()
        self.model = LSTMModel()
        self.model.to(config["training"]["device"])
        self.model.load_state_dict(torch.load(config["training"]["model_path"]))
        self.model.eval()
    
    def predict(self, X):
        X = torch.from_numpy(X).float()
        X = X.to(config["training"]["device"])
        X = X.unsqueeze(0)
        return self.model(X).detach().cpu().numpy()

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

        for epoch in range(config["training"]["epochs"]):
            train_loss, lr = run_epoch(train_dataloader, is_training=True)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, LR: {lr:.6f}")

        torch.save(self.model.state_dict(), config["training"]["model_path"])
    
    