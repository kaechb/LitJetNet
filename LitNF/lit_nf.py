import time
import sys
import json
import traceback
import copy
import os
import traceback
import nflows as nf

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
        '''This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case 
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third'''
        return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=self.config["network_nodes"],
                context_features=1 if self.config["conditional"] else None,
                num_blocks=self.config["network_layers"],
                activation=self.config["activation"]  if "activation" in self.config.keys() else FF.relu,
                dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
                use_batch_norm=self.config["batchnorm"] if "batchnorm" in self.config.keys() else 0,

                    )
    def __init__(self,config):
        '''This initializes the model and its hyperparameters'''
        super().__init__()
        
        self.config=config
        
        #Metrics to track during the training
        self.metrics={"w1p":[],"w1m":[],"w1efp":[]}
        #Loss function of the Normalizing flows
        self.logprobs=[]
        self.hparams.update(config)
        self.save_hyperparameters()
        #This is the Normalizing flow model to be used later, it uses as many
        #coupling_layers as given in the config 
        self.flows = []
        self.n_dim=self.config["n_dim"]
        K=self.config["coupling_layers"]
        for i in range(K):
            '''This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not'''
            if "particle_masks" in self.config.keys() and self.config["particle_masks"] and i==0:
                mask=create_random_binary_mask(self.n_dim//3)            
                mask=mask.repeat_interleave(3)
            else:
                mask=create_random_binary_mask(self.n_dim)
            #Flows can be used as an autoregressive transform, in theory this would be better
            #but this is much slower
            if self.config["autoreg"]:
                self.flows += [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=self.n_dim,
                    num_blocks=self.config["network_layers"], 
                    hidden_features=self.config["network_nodes"],
                    context_features=1 if self.config["conditional"] else None,
                    tails='linear',
                    tail_bound=3,
                    num_bins=self.config["bins"],
                    use_residual_blocks=True,
                    use_batch_norm=self.config["batchnorm"],
                    activation=FF.relu
                            )]
            #Something fun to test, this uses a monotonic network which is invertible
            elif self.config["UMNN"]:
                self.flows += [UMNNCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=create_resnet,
                     integrand_net_layers=[300,200],
                    cond_size=20,
                    nb_steps=20,
                    solver="CCParallel",
                    apply_unconditional_transform=False
                )]
            #Just the standard in this study, use coupling layers
            else:
                self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet, 
                    tails='linear',
                    tail_bound=self.config["tail_bound"],
                    num_bins=self.config["bins"],
                            )]
        #This sets the distribution in the latent space on which we want to morph onto        
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim])
        #Creates working flow model from the list of layer modules
        self.flows=CompositeTransform(self.flows)
        # Construct flow model
        self.flow = Flow(distribution=self.q0, transform=self.flows)


    def load_datamodule(self,data_module):
        '''needed for lightning training to work'''
        self.data_module=data_module
        
    def build_discriminator(self,wgan=False):
        '''this builds a discriminator that can be used to distinguish generated from real
        data, optimally we would want it to have a 50% accuracy, meaning it can distinguish
        This is just a Feed Forward Neural Network
        The wgan keyword makes it a Wasserstein type of discriminator/critic'''
        net=[]
        net.append(nn.Linear(self.n_dim,256))
        for i in range(2):
            net.append(nn.LeakyReLU())
            net.append(nn.Linear(256,256))
        net.append(nn.Linear(256,1))
        if not wgan:
            net.append(nn.Sigmoid())
        self.disc=nn.Sequential(*net)
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """This is only need for WGAN trainign Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.disc(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    
    def sampleandscale(self,batch,c=None):
        '''This is a helper function that samples from the flow (i.e. generates a new sample) 
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one''' 
            
        if  self.config["conditional"]:
            x= batch[:,:self.n_dim].to(self.device),
            c= batch[:,self.n_dim].reshape(-1,1).to(self.device) if c==None else c
        else: 
            x=batch[:,:self.n_dim].to(self.device)
            c=None
        #This make sure that everything is on the right device
        self.data_module.scaler.to(self.device)
        self.flow.to(self.device)
        #Not here that this sample is conditioned on the mass of the current batch allowing the MSE 
        #to be calculated later on
        gen=self.flow.sample(1,c.reshape(-1,1)).reshape(-1,self.n_dim).to(self.device)
        gen=self.data_module.scaler.inverse_transform(torch.hstack((gen[:,:self.n_dim]
            .reshape(-1,self.n_dim),torch.ones(len(gen)).to(self.device).unsqueeze(1))))
        true=self.data_module.scaler.inverse_transform(batch.to(self.device)).to(self.device)
        m=true[:,self.n_dim]
        m_g=mass(gen[:,:self.n_dim].to(self.device) ,
                 self.config["canonical"]).to(self.device) 
        return gen,true,m,m_g
    
    def configure_optimizers(self):
        self.batch_size=self.config["batch_size"]
        #learning rate
        self.lr=self.config["lr"]
        #Total Loss (can be a sum of multiple terms)
        self.losses=[]
        #mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        self.mlosses=[None for i in range(min(self.config["n_mse_delay"],self.config["max_steps"]))]
        self.n_dim=self.config["n_dim"]  
        lr = self.lr
        opt_g = torch.optim.Adam(self.flow.parameters(), lr=lr)
        
        if self.config["disc"]:
            n_critic = 1 #this sets how many critic iterations per one generator iteration
            opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr)
            return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': n_critic}
        )
        else:
            return (
            {'optimizer': opt_g, 'frequency': 1},
        )
    def w_loss(self,gen,true):
        #Wasserstein loss that can be used for gan like training 8which doesnt make a lot of sense)
        lambda_gp=10
        real_validity = self.disc(true[:,:self.n_dim])
                # Fake images
        fake_validity = self.disc(gen[:,:self.n_dim])
        gradient_penalty = self.compute_gradient_penalty(true[:,:self.n_dim], gen[:,:self.n_dim])
        return  -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    
    def bce(self,true,gen): 
        #calculates binary cross entropy, used to train discriminator
        real=self.disc(true[:,:self.n_dim]).reshape(-1,1)
        fake=self.disc(gen[:,:self.n_dim]).reshape(-1,1)
        bc=torch.nn.BCELoss()
        loss=bc(real,torch.ones_like(real))
        loss+=bc(fake,torch.zeros_like(real))
        return  loss
    
    def training_step(self, batch, batch_idx,optimizer_idx=0,wgan=False):
        if  self.config["conditional"]:
            x, c = batch[:,:self.n_dim],batch[:,self.n_dim].reshape(-1,1)
        else: 
            x=batch[:,:self.n_dim]
            c=None
        if self.config["calc_massloss"]:   
            gen,true,m,m_g=self.sampleandscale(batch)   
            #the mass loss is an interesting metric, if we dont add it to the generator loss it will not influence the training
            mloss=self.config["lambda"]*FF.mse_loss(m_g.to(self.device).reshape(-1),m.to(self.device).reshape(-1))

        if optimizer_idx == 0:
            g_loss = -self.flow.log_prob(x,c).mean()/self.n_dim #This is the classic Normalizing Flow loss
            self.log("logprob", g_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
#                 self.logprobs.append(g_loss.detach().cpu().numpy())                
            if  self.global_step>self.config["n_mse_delay"] and self.global_step<self.config["n_mse_turnoff"]:
                g_loss+=mloss
            self.log("mass_loss", mloss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.losses.append(g_loss.detach().cpu().numpy())
            self.mlosses.append(mloss.detach().cpu().numpy())
            return OrderedDict({"loss":g_loss})
        # train discriminator, if config["disc"]= False, this wont be used
        if optimizer_idx == 1:
            if wgan:
                d_loss=self.w_loss()
            else:
                d_loss=self.bce(true,gen)
                yhat=torch.round(model.disc(batch[:,:self.n_dim]))
                TP=yhat.sum()/len(yhat)
                TN=torch.round(torch.ones_like(yhat)-model.disc(gen[:,:self.n_dim])).sum()/len(yhat)
            if self.config["disc"]:
                self.log("d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log("TP",TP,on_step=False, on_epoch=True, prog_bar=True, logger=True  ) 
                self.log("TN",TN,on_step=False, on_epoch=True, prog_bar=True, logger=True  ) 
                self.log("AC",(TP+TN)/2,on_step=False, on_epoch=True, prog_bar=True, logger=True  ) 
            return OrderedDict({"loss":d_loss})

    def validation_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set'''
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
        self.metrics["w1efp"].append(0)    
        self.log("val_w1m",self.metrics["w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p",self.metrics["w1p"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.plot=plotting(model=self,gen=gen,true=true,config=self.config,step=self.global_step,logger=self.logger.experiment)
        if self.config["disc"]:
            with torch.no_grad():
                    self.plot.plot_scores(self.disc(true[:,:self.n_dim].to(self.device)).cpu().numpy(),self.disc(gen[:,:self.n_dim].to(self.device)).cpu().numpy(),save=True)
        try:
            self.plot.plot_mass(m_gen.cpu().numpy(),m_t.cpu().numpy(),save=True,quantile=True)
#             self.plot.plot_marginals(save=True)
            self.plot.plot_2d(save=True)
            self.plot.losses(save=True)
        except Exception as e:
            traceback.print_exc()
# #         self.flow.to("cuda")
