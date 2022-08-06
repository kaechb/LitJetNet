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
from torch import optim
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

            def __init__(self, optimizer, warmup, max_iters):
                self.warmup = warmup
                self.max_num_iters = max_iters
                super().__init__(optimizer)

            def get_lr(self):
                lr_factor = self.get_lr_factor(epoch=self.last_epoch)
                return [base_lr * lr_factor for base_lr in self.base_lrs]

            def get_lr_factor(self, epoch):
                lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
                if epoch <= self.warmup:
                    lr_factor *= epoch * 1.0 / self.warmup
                return lr_factor
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
  
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
class JetNetDataloader(pl.LightningDataModule):
    '''This is more or less standard boilerplate coded that builds the data loader of the training
       one thing to note is the custom standard scaler that works on tensors
       Currently only jets with 30 particles are used but this maybe changes soon'''
    def __init__(self,config): 
        super().__init__()
        self.config=config
        self.n_dim=config["n_dim"]
        self.n_part=config["n_part"]
        self.batch_size=config["batch_size"]
    def setup(self,stage):
    # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
    # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        data_dir=os.environ["HOME"]+"/JetNet_NF/train_{}_jets.csv".format(self.config["parton"])
        data=pd.read_csv(data_dir,sep=" ",header=None)
        jets=[]
        limit=int(self.config["limit"]*1.1)
        for njets in range(1,31):
            masks=np.sum(data.values[:,np.arange(3,120,4)],axis=1)
            df=data.loc[masks==njets,:]
            df=df.drop(np.arange(3,120,4),axis=1)
            df["n"]=njets
            if len(df)>100:
                jets.append(df[:self.config["limit"]])
        #stacking together differnet samples with different number particles per jet
        self.n=torch.empty((0,1))
        self.data=torch.empty((0,self.n_dim*self.n_part))
        for i in range(len(jets)):
            x=torch.tensor(jets[i].values[:,:self.n_dim*self.n_part]).float()
            n=torch.tensor(jets[i]["n"].values).float()
            self.data=torch.vstack((self.data,x))
            self.n=torch.vstack((self.n.reshape(-1,1),n.reshape(-1,1)))        
        
      
        # calculating mass per jet
#         self.m=mass(self.data[:,:self.n_dim]).reshape(-1,1)  
      # Adding noise to zero padded jets.
        for i in torch.unique(self.n):
            i=int(i)
            self.data[self.data[:,-1]==i,3*i:90]=torch.normal(mean=torch.zeros_like(self.data[self.data[:,-1]==i,3*i:90]),std=1).abs()*1e-7
        #standard scaling 
        self.scaler=StandardScaler()
#         self.data=torch.hstack((self.data,self.m))        
        self.scaler.fit(self.data)
        self.data=self.scaler.transform(self.data)
#         self.min_m=self.scaler.transform(torch.zeros((1,self.n_dim+1)))[0,-1]
# #         self.data=torch.hstack((self.data,self.n))
        
#         #calculating mass dist in different bins, this is needed for the testcase where we need to generate the conditoon
#         if self.config["variable"]:
#             self.mdists={}
#             for i in torch.unique(self.n):
#                 self.mdists[int(i)]=F(self.data[self.n[:,0]==i,-2])    
        self.data,self.test_set=train_test_split(self.data.cpu().numpy(),test_size=0.3)
        
#         self.n_train=self.data[:,-1]
#         self.n_test=self.test_set[:,-1]
        
            
        self.test_set=torch.tensor(self.test_set).float()
        self.data=torch.tensor(self.data).float()
#         assert self.data.shape[1]==92
        assert (torch.isnan(self.data)).sum()==0
    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=len(self.test_set),drop_last=True)
