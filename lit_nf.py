import time
import sys
import json
import traceback
import copy
import os
import traceback
import nflows as nf

from nflows.flows.base import Flow
from nflows.utils.torchutils import create_random_binary_mask
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.transforms import *
from nflows.nn import nets
from scipy.stats import norm
from nflows.flows.base import Flow
from nflows.transforms.coupling import *
from nflows.transforms.autoregressive import *
from nflows.transforms.permutations import ReversePermutation
from neural_spline_flows.nde.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from neural_spline_flows.nde.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from neural_spline_flows.nn import ResidualNet
from neural_spline_flows.utils import create_alternating_binary_mask

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR,StepLR,CyclicLR,OneCycleLR
from torch.nn import functional as FF

import numpy as np
import matplotlib.pyplot as plt
import jetnet 
from jetnet.evaluation import w1p,w1efp,w1m,cov_mmd

import mplhep as hep
import hist
from hist import Hist
from markdown import markdown
from pytorch_lightning.loggers import TensorBoardLogger
from collections import OrderedDict
from ray import tune
from helpers import *
from plotting import *


class LitNF(pl.LightningModule):
    
    def create_resnet(self,in_features, out_features):
                    return nets.ResidualNet(
                        in_features,
                        out_features,
                        hidden_features=self.config["network_nodes"],
                        context_features=1 if config["conditional"] else None,
                        num_blocks=self.config["network_layers"],
                        activation=FF.leaky_relu,
                        dropout_probability=config["dropout"],
                        use_batch_norm=self.config["batchnorm"],
                        
                            )
    def __init__(self,config):
       
        super().__init__()
        self.n_dim=config["n_dim"]  
        self.config=config
        self.batch_size=self.config["batch_size"]
        self.lr=self.config["lr"]
        self.losses=[]
        self.mlosses=[None for i in range(min(self.config["n_mse"],self.config["max_steps"]))]
        self.metrics={"w1p":[],"w1m":[],"w1efp":[]}
        self.logprobs=[]
            
        self.hparams.update(config)
        self.save_hyperparameters()
        self.flows = []
        K=self.config["coupling_layers"]
        for i in range(K):
#             mask=create_random_binary_mask(self.n_dim//3)
            
#             mask=mask.repeat_interleave(3)
            if self.config["autoreg"]:
                self.flows += [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=self.n_dim,
                    num_blocks=self.config["network_layers"], 
                    hidden_features=self.config["network_nodes"],
                    context_features=1 if config["conditional"] else None,
                    tails='linear',
                    tail_bound=3,
                    num_bins=self.config["bins"],
                    use_residual_blocks=True,
                    use_batch_norm=self.config["batchnorm"],
                    activation=FF.relu
                            )]
            elif self.config["UMNN"]:
                     
                mask=create_random_binary_mask(self.n_dim)
                self.flows += [UMNNCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=create_resnet,
                     integrand_net_layers=[300,200],
                    cond_size=20,
                    nb_steps=20,
                    solver="CCParallel",
                    apply_unconditional_transform=False
                )]
            else:
                mask=create_random_binary_mask(self.n_dim)
                self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet, 
                    tails='linear',
                    tail_bound=self.config["tail_bound"],
                    num_bins=self.config["bins"],
                            )]
#             mask=create_random_binary_mask(self.n_dim)
#             self.flows+= [AffineCouplingTransform(mask=mask,transform_net_create_fn=self.create_resnet)]
# #             if True:
#                     self.flows += [RandomPermutation(self.n_dim)]
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim])
        self.flows=CompositeTransform(self.flows)
        # Construct flow model
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)
    def load_datamodule(self,data_module):
        self.data_module=data_module
    def configure_optimizers(self):
        lr = self.lr
        self.trainer.reset_train_dataloader()
        
        opt_flow = torch.optim.AdamW(self.parameters(), lr=lr)
#         if self.config["mse"]:
#             opt_mse = torch.optim.AdamW(self.parameters(), lr=lr)
#             return({"optimizer": [opt_flow,opt_mse]})
        if self.config["lr_schedule"]:
            lr_scheduler=StepLR(opt_flow, self.config["n_sched"], gamma=self.config["gamma"])
            return({"optimizer": opt_flow,"lr_scheduler":lr_scheduler})
        else:
             return({"optimizer": opt_flow})
    def build_discriminator(self):
        net=[]
        net.append(nn.Linear(self.n_dim,100))
        for i in range(2):
            net.append(nn.LeakyReLU())
            net.append(nn.Linear(100,100))
        net.append(nn.Linear(100,1))
        net.append(nn.Sigmoid())
        self.disc=nn.Sequential(*net)
                
    def inverse(self,numsamples=1000,c=None):
        #Sample from gauss and transform from latent space to physical space
        if self.config["conditional"]:
            x=self.flow.sample(1,c)
        else:
            x = self.flow.sample(numsamples)
        return x
    def adversarial_loss(self, y_hat, y):
        return FF.binary_cross_entropy(y_hat, y)
    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.flow.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr)
        return [opt_g, opt_d], []
    def training_step(self, batch, batch_idx,optimizer_idx):
        if  self.config["conditional"]:
            x, c = batch[:,:self.n_dim],batch[:,self.n_dim].reshape(-1,1)
        
        else: 
            x=batch[:,:self.n_dim]
            c=None
        self.data_module.scaler.to(self.device)
        self.flow.to(self.device)
        
        # or  self.config["mse"] and self.current_epoch>self.config["n_mse"]:
         
        gen=self.flow.sample(1,c.reshape(-1,1)).reshape(-1,self.n_dim).to(self.device)
        gen=self.data_module.scaler.inverse_transform(torch.hstack((gen[:,:self.n_dim]
            .reshape(-1,self.n_dim),torch.ones(len(gen)).to(self.device).unsqueeze(1))))
        m_g=mass(gen[:,:self.n_dim].to(self.device) ,
                 self.config["canonical"]).to(self.device) 
        true=self.data_module.scaler.inverse_transform(batch.to(self.device)).to(self.device)
        m=true[:,self.n_dim]
        if self.config["gan"]:
            if optimizer_idx == 0:
                g_loss = -self.flow.log_prob(x,c).mean()/self.n_dim
                self.log("logprob", g_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
                self.logprobs.append(g_loss.detach().cpu().numpy())

               
                mloss=self.config["lambda"]*FF.mse_loss(m_g.to(self.device).reshape(-1),m.to(self.device).reshape(-1))
                self.log("mass_loss", mloss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                if  self.global_step>self.config["n_mse"] and self.global_step<self.config["n_turnoff"]:
                    g_loss+=mloss
                # ground truth result (ie: all fake)
                g_loss += self.adversarial_loss(self.disc(gen[:,:self.n_dim]), torch.ones(len(gen),1).to(self.device))
                self.mlosses.append(mloss.detach().cpu().numpy())
                self.losses.append(g_loss.detach().cpu().numpy())
                self.log("loss", g_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                return OrderedDict({"loss":g_loss})
            # train discriminator
            if optimizer_idx == 1:
                # Measure discriminator's ability to classify real from generated samples

                # how well can it label as real?
                valid = torch.ones(len(batch),1).to(self.device)
                real_loss = self.adversarial_loss(self.disc(batch[:,:self.n_dim]), valid)

                # how well can it label as fake?
                fake = torch.zeros(len(gen),1).to(self.device)
                fake_loss = self.adversarial_loss(self.disc(gen[:,:self.n_dim].detach()), fake)

                # discriminator loss is the average of these
                d_loss = (real_loss + fake_loss) / 2
                self.log("d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                return OrderedDict({"loss":d_loss})
      
#         output = OrderedDict({
#           "loss": loss,
#         })
#         return output
    def validation_step(self, batch, batch_idx):
        self.data_module.scaler.to("cpu")        
        with torch.no_grad():
            if self.config["conditional"]:
                gen=self.flow.to("cpu").sample(1,batch[:,self.n_dim].reshape(-1,1).to("cpu")).to("cpu")
                gen=self.data_module.scaler.inverse_transform(torch.hstack((gen[:,:self.n_dim].cpu().detach()
                    .reshape(-1,self.n_dim),torch.ones(len(gen)).unsqueeze(1))))
            else:
                gen=self.flow.sample(len(batch)).to("cpu")
                gen=self.data_module.scaler.inverse_transform(torch.hstack((gen[:,:self.n_dim]
                    .cpu().detach().reshape(-1,self.n_dim),torch.ones(len(gen)).unsqueeze(1))))
        true=self.data_module.scaler.inverse_transform(batch.cpu())
        m_t=true[:,self.n_dim].to(self.device)
        m_gen=mass(gen[:,:self.n_dim],self.config["canonical"])
        self.metrics["w1p"].append(w1p(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1m"].append(w1m(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1efp"].append(w1efp(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.log("val_w1m",self.metrics["w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.plot=plotting(model=self,gen=gen,true=true,config=config,step=self.global_step,logger=self.logger.experiment)
        try:
            self.plot.plot_mass(save=True,quantile=True)
#             self.plot.plot_marginals(save=True)
            self.plot.plot_2d(save=True)
            self.plot.losses(save=True)
        except Exception as e:
            traceback.print_exc()
        self.flow.to("cuda")
