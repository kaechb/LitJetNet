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
print("imports ok")
plt.style.use(hep.style.ROOT)
print(torch.cuda.is_available())



class LitNF(pl.LightningModule):
    
    def create_resnet(self,in_features, out_features):
                    return nets.ResidualNet(
                        in_features,
                        out_features,
                        hidden_features=self.config["network_nodes"],
                        context_features=1 if self.config["conditional"] else None,
                        num_blocks=self.config["network_layers"],
                        activation=FF.leaky_relu,
                        dropout_probability=self.config["dropout"],
                        use_batch_norm=self.config["batchnorm"],
                        
                            )
    def __init__(self,config):
       
        super().__init__()
         
        self.config=config
        self.n_dim=self.config["n_dim"] 
        self.batch_size=self.config["batch_size"]
        self.lr=self.config["lr"]
        self.losses=[]
        self.mlosses=[None for i in range(min(self.config["n_mse"],self.config["max_steps"]))]
        self.metrics={"w1p":[],"w1m":[],"w1efp":[]}
        self.logprobs=[]
            
        self.hparams.update(config)
        self.save_hyperparameters()
        self.flows = []
        print(self.config["coupling_layers"])
        K=self.config["coupling_layers"]
        for i in range(K):
#             mask=create_random_binary_mask(self.n_dim//3)
            
#             mask=mask.repeat_interleave(3)
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
        self.flow = Flow(distribution=self.q0, transform=self.flows)
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
#         if self.config["mse"]:
#             opt_mse = torch.optim.Adam(self.parameters(), lr=lr)
#             return opt_flow, opt_mse
#         else:
        
    def get_metrics(self):
        
        # don't show the version number
        items = super().get_metrics()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items
#     def train_dataloader(self):        
#         return DataLoader(self.data, batch_size=self.batch_size
#                           ,shuffle=True,drop_last=True)
           
#     def training_epoch_end(self, training_step_outputs):
#         print('training steps', training_step_outputs)
        
#         avg_loss = torch.mean(torch.stack([x["loss"]/self.n_dim for x in training_step_outputs]))
# #         avg_logprob = torch.mean(torch.stack([x.logprob for x in training_step_outputs]))
        
#     def forward(self, x,c=None):
#         # in lightning, forward defines the prediction/inference actions
#         z,logprob = self.flow.transform_to_noise(x,c)
#         return z,logprob
    
    
    def inverse(self,numsamples=1000,c=None):
        #Sample from gauss and transform from latent space to physical space
        if self.config["conditional"]:
            x=self.flow.sample(1,c)
        else:
            x = self.flow.sample(numsamples)
        return x
    
    
    def training_step(self, batch, batch_idx,hyperopt=True):
#         self.automatic_optimization=False
        # access your optimizers with use_pl_optimizer=False. Default is True,
#         if self.config["mse"]:
#             opt_flow, opt_mse = self.optimizers(use_pl_optimizer=True)
#         else:
#             opt_flow = self.optimizers(use_pl_optimizer=True)

        if  self.config["conditional"]:
            x, c = batch[:,:self.n_dim],batch[:,self.n_dim].reshape(-1,1)
        
        else: 
            x=batch[:,:self.n_dim]
            c=None
        self.data_module.scaler.to(self.device)
        self.flow.to(self.device)
        loss = -self.flow.log_prob(x,c).mean()/self.n_dim
        self.logprobs.append(loss.detach().cpu().numpy())
            
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  
        if  self.global_step>self.config["n_mse"] and self.global_step<self.config["n_turnoff"]:# or  self.config["mse"] and self.current_epoch>self.config["n_mse"]:
        
            gen=self.flow.sample(1,c.reshape(-1,1)).reshape(-1,self.n_dim).to(self.device)

            gen=self.data_module.scaler.inverse_transform(torch.hstack((gen[:,:self.n_dim]
                .reshape(-1,self.n_dim),torch.ones(len(gen)).to(self.device).unsqueeze(1))))
            m_g=mass(gen[:,:self.n_dim].to(self.device) ,
                     self.config["canonical"]).to(self.device) 
            true=self.data_module.scaler.inverse_transform(batch.to(self.device)).to(self.device)
            m=true[:,self.n_dim]
#             if self.global_step>1000:
#                 _,b,_=plt.hist(m_g.detach().cpu().numpy(),alpha=0.3,bins=30)
#                 plt.hist(m.detach().cpu().numpy(),alpha=0.3,bins=b)
            
#                 plt.savefig("test{}.png".format(self.global_step))
#                 plt.close()
#             if self.global_step>110:
#                 raise
            #loss=0 
            mloss=self.config["lambda"]*FF.mse_loss(m_g.to(self.device).reshape(-1),m.to(self.device).reshape(-1))
            if not hyperopt:
                self.log("mass_loss", mloss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            else:
                
                self.log("mass_loss", mloss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            loss+=mloss
            self.mlosses.append(mloss.detach().cpu().numpy())
            
            
#             loss=mloss
#         return loss

        self.losses.append(loss.detach().cpu().numpy())
        # if  self.global_step>self.config["n_mse"] and self.global_step<self.config["n_turnoff"]:
        #     tune.report(step=self.global_step,mloss=self.mlosses[-1],logprob=self.logprobs[-1],loss=self.losses[-1])    
        # else:
        #     tune.report(step=self.global_step,mloss=0,logprob=self.logprobs[-1],loss=self.losses[-1])    
        output = OrderedDict({
#             'loss': loss.detach(),
            "loss": loss,
#             "mloss": mloss.detach(),
#             'progress_bar': tqdm_dict,
#             'log': tqdm_dict
        })
        return output
    def validation_step(self, batch, batch_idx,hyperopt=True):
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
#         gen_corr=torch.clone(gen[:,:self.n_dim]).reshape(-1,self.n_dim//3,3)
#         gen_corr[gen_corr[:,:,2]<0]=0
#         cov,mmd=cov_mmd(gen_corr,true[:,:self.n_dim].reshape(-1,self.n_dim//3,3),use_tqdm=False)
#         self.log("cov",cov, prog_bar=True, logger=True)
#         self.log("mmd",mmd, prog_bar=True, logger=True)
        gen[:,self.n_dim]=m_gen
        self.metrics["w1p"].append(w1p(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1m"].append(w1m(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1efp"].append(w1efp(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        
        if not hyperopt:
            self.log("val_w1m",self.metrics["w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            tune.report(w1m=self.metrics["w1m"][-1][0],w1p=self.metrics["w1p"][-1][0],w1efp=self.metrics["w1efp"][-1][0],logprob=self.logprobs[-1] if len(self.logprobs)>0 else None
            ,mloss=self.mlosses[-1] if len(self.mlosses)>0 else None) 
            self.log("val_w1m",self.metrics["w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.plot=plotting(model=self,gen=gen,true=true,config=self.config,step=self.global_step,logger=self.logger.experiment)
        try:
            self.plot.plot_mass(save=True,quantile=True)
            # self.plot.plot_marginals(save=True)
            self.plot.plot_2d(save=True)
            self.plot.losses(save=True)

        except Exception as e:
            traceback.print_exc()
        self.flow.to("cuda")
