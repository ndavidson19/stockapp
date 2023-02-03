import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        "model_name": "lstm",
    },
    "arima_para": {
        "p": range(2),
        "d": range(2),
        "q": range(2),
        "seasonal_para": 2,
    }
}


class LSTMModel(nn.Module):
    def __init__(self, config=config, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()
        self.config = config
        self.device = config["training"]["device"]


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

    def run_epoch(self, model, dataloader, is_training = False):
        
        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)
        criterion = nn.MSELoss()
        epoch_loss = 0

        if is_training:
            model.train()
        else:
            model.eval()

        for idx, (x, y) in enumerate(dataloader):
            if is_training:
                optimizer.zero_grad()

            batchsize = x.shape[0]

            x = x.to(self.config["training"]["device"])
            y = y.to(self.config["training"]["device"])

            out = model(x)
            loss = criterion(out.contiguous(), y.contiguous())

            if is_training:
                loss.backward()
                optimizer.step()
                scheduler.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = scheduler.get_last_lr()[0]
        

        return epoch_loss, lr

    def predict(self, test_dataloader, x_unseen):
        model = torch.load(self.config["training"]["model_name"] + ".pt")
        model.eval()
        predictions = []
        for idx, (x, y) in enumerate(test_dataloader):
            x = x.to(self.config["training"]["device"])
            y = y.to(self.config["training"]["device"])
            out = model(x)
            predictions.append(out.detach().cpu().numpy())

        x = torch.tensor(x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
        prediction = model(x)
        prediction = prediction.cpu().detach().numpy()
        return print("Validation Predictions:", np.concatenate(predictions), "Next day price is:", prediction)

        #X = torch.from_numpy(X).float()
        #X = X.to(self.device)
        #X = X.unsqueeze(0)
        #return self.model(X).detach().cpu().numpy()

    def fit(self, model, X_train, y_train, X_val, y_val):
        model = model.to(self.config["training"]["device"])

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["training"]["batch_size"], shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config["training"]["batch_size"], shuffle=False) 

        for epoch in range(self.config["training"]["epochs"]):
            train_loss, lr = self.run_epoch(dataloader=train_dataloader, model=model, is_training=True)
            loss_val, lr_val = self.run_epoch(dataloader=val_dataloader, model=model, is_training=False)
            
            #print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | LR: {lr:.6f}")
            print('Epoch[{}/{}] | loss train:{:.6f} | loss validation:{:.6f} | lr:{:.6f}'
              .format(epoch+1, config["training"]["epochs"], train_loss, loss_val, lr))

        torch.save(model.state_dict(), self.config["training"]["model_path"])
        torch.save(model, self.config["training"]["model_name"] + ".pt")
        
        X_test = torch.from_numpy(X_val).float()
        y_test = torch.from_numpy(y_val).float()
        
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config["training"]["batch_size"], shuffle=False)

        test_loss, lr = self.run_epoch(model, dataloader=test_dataloader, is_training=False)
        print(f"Test Loss: {test_loss:.4f}")

    

    

    
    