
import traceback
import os
import nflows as nf
from nflows.utils.torchutils import create_random_binary_mask
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.nn import nets
from nflows.flows.base import Flow
from nflows.flows import base
from nflows.transforms.coupling import *
from nflows.transforms.autoregressive import *
from particle_net import ParticleNet
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as FF
import numpy as np
from jetnet.evaluation import w1p, w1efp, w1m, cov_mmd,fpnd
import mplhep as hep
import hist
from hist import Hist
from pytorch_lightning.loggers import TensorBoardLogger
from collections import OrderedDict
from ray import tune
from helpers import *
from plotting import *


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F




class LitNF(pl.LightningModule):
    
   
    def create_resnet(self,in_features, out_features):
        '''This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case 
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third'''
        c=self.config["context_features"]
        return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=self.config["network_nodes"],
                context_features=c,
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
        self.metrics={"w1p":[],"w1m":[],"w1efp":[],"cov":[],"mmd":[],"fpnd":[]}
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
            mask=create_random_binary_mask(self.n_dim)  
            if "particle_masks" in self.config.keys() and self.config["particle_masks"] :
                mask=create_random_binary_mask(self.n_dim//3)            
                mask=mask.repeat_interleave(3)
            #Here are the coupling layers of the flow. There seem to be 3 choices but actually its more or less only 2
            #The autoregressive one is incredibly slow while sampling which does not work together with the constraint
            if self.config["spline"]:
                self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet, 
                    tails='linear',
                    tail_bound=self.config["tail_bound"],
                    num_bins=self.config["bins"] )]
            elif self.config["autoreg"]:
                self.flows += [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=self.n_dim,
                    num_blocks=self.config["network_layers"], 
                    hidden_features=self.config["network_nodes"],
                    context_features=self.config["context_features"] ,
                    tails='linear',
                    tail_bound=self.config["tail_bound"],
                    num_bins=self.config["bins"],
                    use_residual_blocks=True,
                    use_batch_norm=self.config["batchnorm"],
                    activation=self.config["activation"])]
            else:
                self.flows+=[ AffineCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet)]
        #This sets the distribution in the latent space on which we want to morph onto        
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim])
        self.q_test =nf.distributions.normal.StandardNormal([self.n_dim])
        #Creates working flow model from the list of layer modules
        self.flows=CompositeTransform(self.flows)
        # Construct flow model
        self.flow_test= base.Flow(distribution=self.q_test, transform=self.flows)
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)
        
    def build_disc(self,config=None):
        '''this builds a discriminator that can be used to distinguish generated from real
        data, optimally we would want it to have a 50% accuracy, meaning it can distinguish
        This is just a Feed Forward Neural Network
        The wgan keyword makes it a Wasserstein type of discriminator/critic
        I played around with this but it did not work particularly well'''
        if config:
                settings = {
                "conv_params": [
                    (8, (64, 64, 64)),
                    (8, (128, 128, 128)),
                    (8, (128, 128, 128)),
                ],
                "fc_params": [
                    (0.0, 128)
                ],
                "input_features": 3,
                "output_classes": self.config["context_features"]}
                self.particle_net=ParticleNet(settings)
                
                return 0


    def load_datamodule(self,data_module):
        '''needed for lightning training to work, it just sets the dataloader for training and validation'''
        self.data_module=data_module
        
    def on_after_backward(self) -> None:
        '''This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues'''
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()
    def sampleandscale(self,batch,c=None,n=None):
        '''This is a helper function that samples from the flow (i.e. generates a new sample) 
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation''' 
        x= batch[:,:self.n_dim].to(self.device),
        n_c=self.config["context_features"]
        if n_c>0:            
            c= batch[:,self.n_dim:].reshape(-1,n_c).to(self.device) if c==None else c
            gen=self.flow.sample(1,c.reshape(-1,n_c)).reshape(-1,self.n_dim).to(self.device)
        else:
            gen=self.flow.sample(len(batch)).reshape(-1,self.n_dim).to(self.device)
        #This make sure that everything is on the right device
        self.data_module.scaler.to(self.device)
        self.flow.to(self.device)
        #Not here that this sample is conditioned on the mass of the current batch allowing the MSE 
        #to be calculated later on
        gen=self.data_module.scaler.inverse_transform(torch.hstack((gen[:,:self.n_dim]
            .reshape(-1,self.n_dim),torch.ones(len(gen)).to(self.device).unsqueeze(1))))
        true=self.data_module.scaler.inverse_transform(batch[:,:self.n_dim+1].to(self.device)).to(self.device)
        m=true[:,self.n_dim]
        m_g=mass(gen[:,:self.n_dim].to(self.device) ,
                 self.config["canonical"]).to(self.device)
        gen=torch.column_stack((gen,m_g))
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
        opt_g = torch.optim.AdamW(self.flow.parameters(), lr=self.lr)
        self.opt_g=opt_g
        return (
        {'optimizer': opt_g, 'frequency': 1},
    )
    def test_cond(self,num):
        """this sampels mass and number particle conditions in an autoregressive manner needed for data generation, 
            First the number particles are sampeled randomly from the pmf, and then the mass distribution for every case with 
            n particles is calculated - this distribution is then interpolated and a 1d flows is constructed """
        ns=self.data_module.n[torch.randint(low=0,high=len(self.data_module.n),size=(num,))]
        c=torch.empty((0,2))
        counts=torch.unique(ns,return_counts=True)
        for n,count in zip(counts[0],counts[1]):
            m_temp=torch.tensor(self.data_module.mdists[int(n)][1](torch.rand(size=(count,)).numpy())).reshape(-1,1).float()
            n_temp=torch.tensor(int(n)).repeat(count).reshape(-1,1)
            c_temp=torch.hstack((m_temp,n_temp))
            c=torch.vstack((c,c_temp))
        return c
    def training_step(self, batch, batch_idx,optimizer_idx=0,wgan=False):
        """training loop of the model, here all the data is passed forward to a gaussian
            This is the important part what is happening here. This is all the training we do """
        x,c= batch[:,:self.n_dim],batch[:,self.n_dim:]
        self.opt_g.zero_grad()
        
        # This is the mass constraint, which constrains the flow to generate events with a mass which is the same as the mass it has been conditioned on, we can choose to not calculate this when we work without mass constraint to make training faster
        if self.config["calc_massloss"]  :
                self.gen,self.true,self.m,self.m_g=self.sampleandscale(batch,c=c)

                mloss=FF.mse_loss(self.m_g.to(self.device).reshape(-1),self.m.to(self.device).reshape(-1))
                assert not torch.any(self.m_g.isnan()) or not torch.any(self.m.isnan())
                self.mlosses.append(mloss.detach().cpu().numpy())
                self.log("mass_loss", mloss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        ##Normalizing Flow loss Normalizing Flow loss
        g_loss = -self.flow.to(self.device).log_prob(x,c if self.config["context_features"] else None).mean()/self.n_dim
        self.log("logprob", g_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) 
        self.logprobs.append(g_loss.detach().cpu().numpy())
        #some conditions on when we want to actually add the mass loss to our training loss, if we dont add it, it is as it wouldnt exist
        if self.global_step>self.config["n_mse_delay"] and self.config["context_features"]>0 and self.config["calc_massloss"]:
            g_loss+=self.config["lambda"]*mloss
            self.log("combined_loss", g_loss, on_epoch=True, prog_bar=True, logger=True)
        self.losses.append(g_loss.detach().cpu().numpy())
        return OrderedDict({"loss":g_loss})
        
    def validation_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set (checking for overtraining)'''
        self.data_module.scaler.to("cpu")  
        batch=batch.to("cpu")
        c=batch[:,-self.config["context_features"]:] #this is the condition
        c_test=self.test_cond(len(batch))#this is the condition in the case of testing
        with torch.no_grad():
            if self.config["context_features"]:
                gen=self.flow_test.to("cpu").sample(1,c).to("cpu")
                test=self.flow_test.to("cpu").sample(1,c_test.float()).to("cpu")
                gen=torch.hstack((gen[:,:self.n_dim].cpu().detach().reshape(-1,self.n_dim),torch.ones(len(gen)).unsqueeze(1)))                
                test=torch.hstack((test[:,:self.n_dim].cpu().detach().reshape(-1,self.n_dim),torch.ones(len(test)).unsqueeze(1)))
            else:
                gen=self.flow.to("cpu").sample(len(batch)).to("cpu")
                test=self.flow.to("cpu").sample(len(batch)).to("cpu")
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        test=self.data_module.scaler.inverse_transform(test)
        gen=self.data_module.scaler.inverse_transform(gen)
        true=self.data_module.scaler.inverse_transform(batch[:,:self.n_dim+1])
        # We overwrite in cases where n is smaller 30 the particles after n with 0
        if self.config["context_features"]>1:
            for i in torch.unique(batch[:,-1]):
                i=int(i)
                gen[c[:,-1]==i,3*i:]=0
                test[c_test[:,-1]==i,3*i:-1]=0
        true=true[:,:self.n_dim]
        #This is just a nice check to see whether we overtrain 
        logprob = -self.flow.to("cpu").log_prob(batch[:,:self.n_dim],batch[:,self.n_dim:]).detach().mean().numpy()/self.n_dim
        
        #calculate mass distrbutions & concat them to training sample
        m_t=mass(true[:,:self.n_dim].to(self.device),self.config["canonical"]).cpu()
        m_gen=mass(gen[:,:self.n_dim],self.config["canonical"]).cpu()
        m_test=mass(test[:,:self.n_dim],self.config["canonical"]).cpu()
        gen=torch.column_stack((gen[:,:90],m_gen))
        test=torch.column_stack((test[:,:90],m_test))       
        # Again checking for overtraining
        mse=FF.mse_loss(m_t,m_test).detach()

        if self.config["canonical"]:
            gen[:,:90]=preprocess(gen[:,:90],rev=True)
            test[:,:90]=preprocess(test[:,:90],rev=True)
        # For one metric the pt needs to always be bigger or equal 0, so we overwrite the cases where it isnt (its not physical possible to ahve pt smaller 0)
        for i in range(30):
            i=2+3*i
            gen[gen[:,i]<0,i]=0
            test[test[:,i]<0,i]=0
            true[true[:,i]<0,i]=0
        #Some metrics we track
        cov,mmd=cov_mmd(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3),use_tqdm=False)
        fpndv=fpnd(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),use_tqdm=False,jet_type=self.config["parton"])

        self.metrics["fpnd"].append(fpndv)
        self.metrics["mmd"].append(mmd)
        self.metrics["cov"].append(cov)
        self.metrics["w1p"].append(w1p(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1m"].append(w1m(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1efp"].append(w1efp(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        
        self.log("val_w1m",self.metrics["w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p",self.metrics["w1p"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1efp",self.metrics["w1efp"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_logprob",logprob,prog_bar=True,logger=True)
        self.log("val_cov",cov,prog_bar=True,logger=True,on_step=False, on_epoch=True)
        self.log("val_fpnd",fpndv,prog_bar=True,logger=True,on_step=False, on_epoch=True)
        self.log("val_mmd",mmd,prog_bar=True,logger=True,on_step=False, on_epoch=True)
        self.log("val_mse",mse,prog_bar=True,logger=True,on_step=False, on_epoch=True)
        # This part here adds the plots to tensorboard
        self.plot=plotting(model=self,gen=test[:,:self.n_dim],true=true[:,:self.n_dim],config=self.config,step=self.global_step,logger=self.logger.experiment)
        self.flow=self.flow.to(self.device)
        try:
            self.plot.plot_mass(m_test.cpu().numpy(),m_t.cpu().numpy(),save=True,bins=30,quantile=True,plot_vline=False)
#             self.plot.plot_marginals(save=True)
            self.plot.plot_2d(save=True)
            self.plot.losses(save=True)
        
        except Exception as e:
            traceback.print_exc()
            
# #         self.flow.to("cuda")
        self.plot.plot_correlations()

