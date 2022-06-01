import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl   
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from helpers import *
from nflows.distributions.base import Distribution
import matplotlib.pyplot as plt

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
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
    def inverse_transform(self,values):
        return (values *self.std)+self.mean
    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    def to(self,dev):
        self.std=self.std.to(dev)
        self.mean=self.mean.to(dev)
        return self
  
class JetNetDataloader(pl.LightningDataModule):
    '''This is more or less standard boilerplate coded that builds the data loader of the training
       one thing to note is the custom standard scaler that works on tensors
       Currently only jets with 30 particles are used but this maybe changes soon'''
    def __init__(self,config): 
        super().__init__()
        self.config=config
        self.n_dim=config["n_dim"]
        self.batch_size=config["batch_size"]
    def setup(self,stage):
    # transforms for images
        data_dir=os.environ["HOME"]+"/JetNet_NF/train_{}_jets.csv".format("q")
        data=pd.read_csv(data_dir,sep=" ",header=None)
        jets=[]
        limit=int(self.config["limit"]*1.1)
        
        for njets in range(1,31):
            masks=np.sum(data.values[:,np.arange(3,120,4)],axis=1)
            df=data.loc[masks==njets,:]
            df=df.drop(np.arange(3,120,4),axis=1)
            df["n"]=njets
            if len(df)>0:
                jets.append(df[:self.config["limit"]])
        for i in range(len(jets)):
            if i==0:
                self.data=torch.tensor(jets[i].values[:,:self.n_dim]).float()
                self.n=torch.tensor(jets[i]["n"].values)
            else:
                x=torch.tensor(jets[i].values[:,:self.n_dim]).float()
                n=torch.tensor(jets[i]["n"].values).float()
                self.data=torch.vstack((self.data,x))
                self.n=torch.vstack((self.n.reshape(-1,1),n.reshape(-1,1)))
        
        self.scaler=StandardScaler()
        if self.config["canonical"]:
            self.data=preprocess(self.data)
        
        self.m=mass(self.data[:,:self.n_dim],self.config["canonical"]).reshape(-1,1)
      
        
        self.data=torch.hstack((self.data,self.m))
        self.scaler.fit(self.data)
        self.data=self.scaler.transform(self.data)
        self.data=torch.hstack((self.data,self.n))
        # for i in range(30):
        #     self.data[self.data[-1]==i,3*i:]=torch.normal(torch.zeros_like((self.data[-1]==i,90-3*i)))*1e-7
        self.data,self.test_set=train_test_split(self.data.cpu().numpy(),test_size=0.1)
        self.test_set=torch.tensor(self.test_set).float()
        self.data=torch.tensor(self.data).float()
        assert (torch.isnan(self.data)).sum()==0
        assert self.data.shape[1]==self.n_dim+self.config["context_features"]
        # plt.hist(self.scaler.inverse_transform(torch.vstack((self.data,self.test_set)))[:,self.n_dim].numpy(),bins=30,alpha=0.3,label='reversescaled')
        # plt.hist(self.m.numpy(),bins=30,alpha=0.3,label='true')
        # plt.legend()
        # plt.show()
        
    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=len(self.test_set))
        