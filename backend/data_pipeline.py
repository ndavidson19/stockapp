import numpy as np


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from alpha_vantage.timeseries import TimeSeries 


class DataPipeline(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_date, self.data_close_price, self.num_data_points, self.display_date_range = self.download_data()
        self.normalized_data_close_price = self.normalize()
        self.data_x, self.data_x_unseen = self.prepare_data_x()
        self.data_y = self.prepare_data_y()
        self.data_x_train, self.data_x_val, self.data_y_train, self.data_y_val = self.split_data()
        self.to_plot_data_y_train, self.to_plot_data_y_val = self.prepare_data_for_plotting()

    def download_data(self):
        config = self.config
        ts = TimeSeries(key='COR04RQ1Z1P74ETF') #you can use the demo API key for this project, but please make sure to eventually get your own API key at https://www.alphavantage.co/support/#api-key. 
        data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

        data_date = [date for date in data.keys()]
        data_date.reverse()

        data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
        data_close_price.reverse()
        data_close_price = np.array(data_close_price)

        num_data_points = len(data_date)
        display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
        print("Number data points", num_data_points, display_date_range)

        return data_date, data_close_price, num_data_points, display_date_range

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)


class DataCleaning():
    def __init__(self):
        self.data_x = None
        self.data_x_unseen = None
        self.data_y = None
        self.data_x_train = None
        self.data_x_val = None
        self.data_y_train = None
        self.data_y_val = None
        self.to_plot_data_y_train = None
        self.to_plot_data_y_val = None

    def prepare_data_x(self, x, window_size):
        # perform windowing
        n_row = x.shape[0] - window_size + 1
        output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
        return output[:-1], output[-1]

    def prepare_data_y(self, x, window_size):
        # # perform simple moving average
        # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

        # use the next day as label
        output = x[window_size:]
        return output

    def split_data(self, data_x, data_y, train_split_size):
        split_index = int(data_y.shape[0]*train_split_size)
        data_x_train = data_x[:split_index]
        data_x_val = data_x[split_index:]
        data_y_train = data_y[:split_index]
        data_y_val = data_y[split_index:]
        return data_x_train, data_x_val, data_y_train, data_y_val

    def prepare_data_for_plotting(self, num_data_points, data_y_train, data_y_val):
        to_plot_data_y_train = np.zeros(num_data_points)
        to_plot_data_y_val = np.zeros(num_data_points)

        to_plot_data_y_train[config["data"]["window_size"]:] = data_y_train
        to_plot_data_y_val[config["data"]["window_size"]:split_index] = data_y_train[-1]
        to_plot_data_y_val[split_index:] = data_y_val
        return to_plot_data_y_train, to_plot_data_y_val

    def get_data(self):
        return self.data_x, self.data_x_unseen, self.data_y, self.data_x_train, self.data_x_val, self.data_y_train, self.data_y_val, self.to_plot_data_y_train, self.to_plot_data_y_val

    def run(self, config, normalized_data_close_price):
        self.data_x, self.data_x_unseen = self.prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
        self.data_y = self.prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])
        self.data_x_train, self.data_x_val, self.data_y_train, self.data_y_val = self.split_data(self.data_x, self.data_y, train_split_size=config["data"]["train_split_size"])
        self.to_plot_data_y_train, self.to_plot_data_y_val = self.prepare_data_for_plotting(num_data_points, self.data_y_train, self.data_y_val)


data_cleaning = DataCleaning()

data_cleaning.run(config, normalized_data_close_price)

data_x, data_x_unseen, data_y, data_x_train, data_x_val, data_y_train, data_y_val, to_plot_data_y_train, to_plot_data_y_val = data_cleaning.get_data()

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)



