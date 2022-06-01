
import traceback
import os
import nflows as nf
from nflows.distributions.base import Distribution
from nflows.utils.torchutils import create_random_binary_mask
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.nn import nets
from nflows.flows.base import Flow
from nflows.flows import base
from nflows.transforms.coupling import *
from nflows.transforms.autoregressive import *
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as FF
import numpy as np
from jetnet.evaluation import w1p, w1efp, w1m, cov_mmd
import mplhep as hep
import hist
from hist import Hist
from pytorch_lightning.loggers import TensorBoardLogger
from collections import OrderedDict
from ray import tune
from helpers import *
from plotting import *


class StandardNormalTemp(Distribution):
        """A multivariate Normal with zero mean and a covariance that 
            can be chosen to be any value.
            From images generation it resulted that a lower variance gives 
            better sample result - this did not show the same effect here"""
        def __init__(self, shape,temp=1):
            super().__init__()
            self._shape = torch.Size(shape)
            self.temp=temp
            self.register_buffer("_log_z",
                                 torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                              dtype=torch.float64),
                                 persistent=False)

        def _log_prob(self, inputs, context):
            # Note: the context is ignored.
            if inputs.shape[1:] != self._shape:
                raise ValueError(
                    "Expected input of shape {}, got {}".format(
                        self._shape, inputs.shape[1:]
                    )
                )
            neg_energy = -0.5 * \
                torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
            return neg_energy - self._log_z

        def _sample(self, num_samples, context):
            if context is None:
                return torch.normal(std=torch.ones(self._shape, device=self._log_z.device)*self.temp)
            else:
                # The value of the context is ignored, only its size and device are taken into account.
                context_size = context.shape[0]
                samples = torch.randn(context_size * num_samples, *self._shape,
                                      device=context.device)
                return torchutils.split_leading_dim(samples, [context_size, num_samples])

        def _mean(self, context):
            if context is None:
                return self._log_z.new_zeros(self._shape)
            else:
                # The value of the context is ignored, only its size is taken into account.
                return context.new_zeros(context.shape[0], *self._shape)

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
            mask=create_random_binary_mask(self.n_dim)  
            if "particle_masks" in self.config.keys() and self.config["particle_masks"] :
                mask=create_random_binary_mask(self.n_dim//3)            
                mask=mask.repeat_interleave(3)
            self.flows += [PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=self.create_resnet, 
            tails='linear',
            tail_bound=self.config["tail_bound"],
            num_bins=self.config["bins"],
                        )]
        #This sets the distribution in the latent space on which we want to morph onto        
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim])
        self.q_test =StandardNormalTemp(shape=[self.n_dim],temp=0.8)
        #Creates working flow model from the list of layer modules
        self.flows=CompositeTransform(self.flows)
        # Construct flow model
        self.flow_test= base.Flow(distribution=self.q_test, transform=self.flows)
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)
        
    def build_disc(self,config=None):
        '''this builds a discriminator that can be used to distinguish generated from real
        data, optimally we would want it to have a 50% accuracy, meaning it can distinguish
        This is just a Feed Forward Neural Network
        The wgan keyword makes it a Wasserstein type of discriminator/critic'''
        if config:
            if config["particle_net"]:
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
        '''needed for lightning training to work'''
        self.data_module=data_module
        
    
    def sampleandscale(self,batch,c=None,n=None):
        '''This is a helper function that samples from the flow (i.e. generates a new sample) 
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one''' 
            
        
        x= batch[:,:self.n_dim].to(self.device),
        if self.config["context_features"]>0:
            n_c=self.config["context_features"]
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
        opt_g = torch.optim.Adam(self.flow.parameters(), lr=self.lr)
        return (
        {'optimizer': opt_g, 'frequency': 1},
    )

    def training_step(self, batch, batch_idx,optimizer_idx=0,wgan=False):
                
        x,c= batch[:,:self.n_dim],batch[:,self.n_dim:]
       
        
        if self.config["calc_massloss"]  :# and self.global_step//10==0   
            self.gen,self.true,self.m,self.m_g=self.sampleandscale(batch,c=c)
            mloss=FF.mse_loss(self.m_g.to(self.device).reshape(-1),self.m.to(self.device).reshape(-1))
            self.mlosses.append(mloss.detach().cpu().numpy())
            self.log("mass_loss", mloss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

#           
        
#             self.mlosses.append(mloss.detach().cpu().numpy())
        g_loss = -self.flow.to("cuda").log_prob(x,c if self.config["context_features"] else None).mean()/self.n_dim #This is the classic Normalizing Flow loss
        self.log("logprob", g_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) 
        self.logprobs.append(g_loss.detach().cpu().numpy())
        
        if self.global_step>self.config["n_mse_delay"] and self.config["context_features"]>0:
        #the mass loss is an interesting metric, if we dont add it to the generator loss it will not influence the training

            g_loss+=self.config["lambda"]*mloss
            self.log("combined_loss", g_loss, on_epoch=True, prog_bar=True, logger=True)
        self.losses.append(g_loss.detach().cpu().numpy())
        return OrderedDict({"loss":g_loss})
        

    def validation_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set'''
        self.data_module.scaler.to("cpu")  
#         batch_cloud=Batch().from_data_list([Data(x=batch[i,:90].reshape(30,3),pos=batch[i,:90].reshape(30,3)) for i in range(len(batch))])#
        
        with torch.no_grad():
            if self.config["context_features"]:
                gen=self.flow_test.to("cpu").sample(1,batch[:,self.n_dim:].reshape(-1,self.config["context_features"]).to("cpu"))  
                gen=torch.hstack((gen[:,:self.n_dim].cpu().detach().reshape(-1,self.n_dim),torch.ones(len(gen)).unsqueeze(1)))
            else:
                gen=self.flow.to("cpu").sample(len(batch)).to("cpu")
        gen=self.data_module.scaler.inverse_transform(gen)
        if self.config["context_features"]>1:
            for i in torch.unique(batch[:,-1]):
                i=int(i)
                gen[batch[:,-1]==i,3*i:]=0
        batch=batch.to("cpu")
        logprob = -self.flow.to("cpu").log_prob(batch[:,:self.n_dim],batch[:,self.n_dim:]).detach().mean().numpy()/self.n_dim
        
        true=self.data_module.scaler.inverse_transform(batch[:,:self.n_dim+1].cpu())
        m_t=mass(true[:,:self.n_dim].to(self.device))
        m_gen=mass(gen[:,:self.n_dim],self.config["canonical"])
        gen=torch.column_stack((gen[:,:90],m_gen))
        
        self.metrics["w1p"].append(w1p(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1m"].append(w1m(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.metrics["w1efp"].append(w1efp(gen[:,:self.n_dim].reshape(-1,self.n_dim//3,3).numpy(),true[:,:self.n_dim].reshape(-1,self.n_dim//3,3)))
        self.log("val_w1m",self.metrics["w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p",self.metrics["w1p"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1efp",self.metrics["w1efp"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_logprob",logprob,prog_bar=True,logger=True)
        
        self.plot=plotting(model=self,gen=gen[:,:self.n_dim],true=true[:,:self.n_dim],config=self.config,step=self.global_step,logger=self.logger.experiment)
        self.flow=self.flow.to("cuda")
        try:
            self.plot.plot_mass(m_gen.cpu().numpy(),m_t.cpu().numpy(),save=True,bins=30,quantile=True,plot_vline=False)
#             self.plot.plot_marginals(save=True)
            self.plot.plot_2d(save=True)
            self.plot.losses(save=True)
        
        except Exception as e:
            traceback.print_exc()
            
# #         self.flow.to("cuda")
        self.plot.plot_correlations()

