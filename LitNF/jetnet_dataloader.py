import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import QuantileTransformer
from helpers import mass


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native
        functions. The module does not expect the tensors to be of any specific shape;
         as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values):
        return (values * self.std) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def to(self, dev):
        self.std = self.std.to(dev)
        self.mean = self.mean.to(dev)
        return self


import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class JetNetDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
    Currently only jets with 30 particles are used but this maybe changes soon"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_dim = config["n_dim"]
        self.n_part = config["n_part"]
        self.batch_size = config["batch_size"]

    def setup(self, stage):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        data_dir = os.environ["HOME"] + "/JetNet_NF/train_{}_jets.csv".format(self.config["parton"])
        data = pd.read_csv(data_dir, sep=" ", header=None)
        df = pd.DataFrame()
        limit = int(self.config["limit"] * 1.1)

        # masks=np.sum(data.values[:,np.arange(3,120,4)],axis=1)
        masks = data.values[:, np.arange(3, 120, 4)][:limit] # zero padded particles are masked and given as feature
        df = data.drop(np.arange(3, 120, 4), axis=1)[:limit]
        self.n = masks.sum(axis=1) # number particles per jet

        masks=~masks.astype(bool)
        z = torch.tensor(df.values).reshape(len(df), 30, 3)
        m = (torch.tensor(masks).reshape(len(df), 30)).bool()

        self.data = z

        self.data[m, :] = torch.normal(mean=torch.zeros_like(self.data[m, :]), std=1).abs() * 1e-7
        # standard scaling
        self.scalers=[]
        self.ptscalers=[]
        self.data=self.data.float()
        for i in range(30):
            self.scalers.append(StandardScaler())
            # self.scaler = 
            if self.config["quantile"]:
                self.ptscalers.append( QuantileTransformer(output_distribution="normal"))
                self.data[~m[:,i],i,:2] = self.scalers[i].fit_transform(self.data[~m[:,i],i, :2])
                self.data[~m[:,i],i, 2] = torch.tensor(self.ptscalers[i].fit_transform(self.data[~m[:,i],i, 2].reshape(-1,1).numpy())).reshape(-1)
                self.min_pt = self.data[~m[:,i],i, :2].min(axis=0)[0]
                
            else:
                self.data[~m[:,i],i, :] = self.scalers[i].fit_transform(self.data[~m[:,i],i, :])
        self.data = self.data.reshape(len(self.data), 90)
        self.data = torch.tensor(np.hstack((self.data.reshape(len(self.data), self.n_part * self.n_dim), m)))
        self.data, self.test_set = train_test_split(self.data.cpu().numpy(), test_size=0.3)

        self.test_set = torch.tensor(self.test_set).float()
        self.data = torch.tensor(self.data).float()
        self.num_batches = len(self.data) // self.config["batch_size"]
        #         assert self.data.shape[1]==92
        assert (torch.isnan(self.data)).sum() == 0

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=len(self.test_set), drop_last=True)
