import math
import numpy as np


class TimeseriesGenerator():
    def __init__(self, data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128):
        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil((self.end_index - self.start_index) / self.batch_size)

    def __getitem__(self, idx):
        if self.shuffle:
            rows = np.random.randint(self.start_index, self.end_index, size=self.batch_size)
        else:
            i = self.start_index + idx * self.batch_size
            rows = np.arange(i, min(i + self.batch_size, self.end_index), self.stride)
        samples = np.zeros((len(rows), self.length // self.sampling_rate, self.data.shape[-1]))
        targets = np.zeros((len(rows), self.data.shape[-1]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - self.length, rows[j], self.sampling_rate)
            samples[j] = self.data[indices]
            targets[j] = self.targets[rows[j]]
        if self.reverse:
            return samples[:, ::-1, :], targets
        return samples, targets
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    