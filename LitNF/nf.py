import os
import time
import traceback

import matplotlib.pyplot as plt
import nflows as nf
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.autograd as autograd
from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p
from nflows.flows import base
from nflows.nn import nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import \
    PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_random_binary_mask
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as FF
from torch.nn.functional import leaky_relu, sigmoid

from helpers import CosineWarmupScheduler, Scheduler
from plotting import *


class NF(pl.LightningModule):
    def create_resnet(self, in_features, out_features):
        """This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third"""
        c = self.config["context_features"]
        return nets.ResidualNet(
            in_features,
            out_features,
            hidden_features=self.config["network_nodes_nf"],
            context_features=c,
            num_blocks=self.config["network_layers_nf"],
            activation=self.config["activation"] if "activation" in self.config.keys() else FF.relu,
            dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
            use_batch_norm=self.config["batchnorm"] if "batchnorm" in self.config.keys() else 0,
        )

    def __init__(self, config, num_batches):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.hyperopt = True
        self.start = time.time()
        self.config = config
        self.automatic_optimization = False
        self.freq_d = config["freq"]
        self.wgan = config["wgan"]
        # Loss function of the Normalizing flows
        self.logprobs = []
        self.n_part = config["n_part"]
        self.save_hyperparameters()
        self.flows = []
        self.fpnds = []
        self.w1ms = []
        self.stop_train=False
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.alpha = 1
        self.num_batches = int(num_batches)
        self.build_flow()
        
        self.sig = nn.Sigmoid()
        self.df = pd.DataFrame()
       
        self.train_nf = config["max_epochs"] // config["frac_pretrain"]
        self.refinement=False

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def _summary(self, temp):
        self.summary_path = "/beegfs/desy/user/{}/{}/summary.csv".format(os.environ["USER"], self.config["name"])
        if os.path.isfile(self.summary_path):
            summary = pd.read_csv(self.summary_path).set_index(["path_index"])
        else:
            print("summary not found")
            summary = pd.DataFrame()
        summary.loc[self.logger.log_dir, self.config.keys()] = self.config.values()
        summary.loc[self.logger.log_dir, temp.keys()] = temp.values()
        summary.loc[self.logger.log_dir, "time"] = time.time() - self.start
        summary.to_csv(self.summary_path, index_label=["path_index"])
        return summary

    def _results(self, temp):
        self.df = pd.concat([self.df,pd.DataFrame([temp],index=[self.current_epoch])])
        self.df.to_csv(self.logger.log_dir + "result.csv", index_label=["index"])

    def on_after_backward(self) -> None:
        """This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues"""
        valid_gradients = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("not valid grads", self.counter)
            self.zero_grad()
            self.counter += 1
            if self.counter > 5:
                raise ValueError("5 nangrads in a row")
            self.stop_train=True
        else:
            self.counter = 0
    
    def sample_n(self, mask):
        #Samples a mask where the zero padded particles are True, rest False
        mask_test = torch.ones_like(mask)
        n, counts = np.unique(self.data_module.n, return_counts=True)
        counts_prob = torch.tensor(counts / len(self.data_module.n) )
        n_test=n[torch.multinomial(counts_prob,replacement=True,num_samples=(len(mask)))] 
        indices = torch.arange(30, device=mask.device)
        mask_test = (indices.view(1, -1) < torch.tensor(n_test).view(-1, 1))      
        mask_test=~mask_test.bool()
        return (mask_test)
    
    def build_flow(self):
        K = self.config["coupling_layers"]
        for i in range(K):
            """This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not"""
            mask = create_random_binary_mask(self.n_dim * self.n_part)
            self.flows += [PiecewiseRationalQuadraticCouplingTransform(mask=mask,transform_net_create_fn=self.create_resnet,
                            tails="linear",tail_bound=self.config["tail_bound"],num_bins=self.config["bins"],)]
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim * self.n_part])
        # Creates working flow model from the list of layer modules
        self.flows = CompositeTransform(self.flows)
        # Construct flow model
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)

    

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        opt_nf = torch.optim.AdamW(self.flow.parameters(), lr=self.config["lr_nf"])
        
        return opt_nf

   
    def training_step(self, batch, batch_idx):
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        mask = batch[:, self.n_part*self.n_dim:].bool()
        batch = batch[:, : self.n_part*self.n_dim]
        if  self.config["context_features"]==1:
                c=mask.clone()
                c = (~c).sum(axis=1).reshape(-1,1).float()
        else: 
            c=None
        opt_nf = self.optimizers()
        nf_loss = -self.flow.to(self.device).log_prob(batch,context=c).mean()
        nf_loss /= self.n_dim * self.n_part
        opt_nf.zero_grad()
        self.manual_backward(nf_loss)
        opt_nf.step()
        self.log("logprob", nf_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        mask = batch[:, 90:].cpu().bool()
        batch = batch[:, :90].cpu()
        mask_test=self.sample_n(mask).bool()

        if  self.config["context_features"]==1:
            c=mask_test.clone()
            c = (~c).sum(axis=1).reshape(-1,1).float()
        else: 
            c=None
        self.flow.eval()
        batch = batch.to("cpu")
        self.flow = self.flow.to("cpu")
        
        with torch.no_grad():
            logprob = -self.flow.log_prob(batch,context=c).mean() / 90
            z = self.flow.sample(len(batch) if self.config["context_features"]==0 else 1,context=c).reshape(len(batch),self.n_part, self.n_dim) 
            self.data_module.scaler = self.data_module.scaler.to(batch.device)

            z_scaled=self.data_module.scaler.inverse_transform(z)
            true_scaled=self.data_module.scaler.inverse_transform(batch.reshape(len(batch),self.n_part, self.n_dim) )
            z_scaled[mask_test]=0
        m_t = mass(true_scaled,self.config["canonical"])
        m_gen = mass(z_scaled, self.config["canonical"])
        m_c = mass(z_scaled, self.config["canonical"])
        
        for i in range(30):
            
            z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        # Some metrics we track
        cov, mmd = cov_mmd( true_scaled,z_scaled, use_tqdm=False)
        
        try:
            
            fpndv = fpnd(z_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.config["parton"])
        except:
            fpndv = 1000

        w1m_ = w1m(z_scaled, true_scaled)[0]
        w1p_ = w1p(z_scaled, true_scaled)[0]
        w1efp_ = w1efp(z_scaled, true_scaled)[0]

        
        self.w1ms.append(w1m_)
        self.fpnds.append(fpndv)
        temp = {"val_logprob": float(logprob.numpy()),"val_fpnd": fpndv,"val_mmd": mmd,"val_cov": cov,"val_w1m": w1m_,
                "val_w1efp": w1efp_,"val_w1p": w1p_,"step": self.global_step,}
        print("epoch {}: ".format(self.current_epoch), temp)
       
        
        if self.hyperopt and self.global_step > 3:
            try:
                self._results(temp)
            except:
                print("error in results")
            if (fpndv<self.fpnds).all():             
                summary = self._summary(temp)
        
        self.log("hp_metric", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1m", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p", w1p_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1efp", w1efp_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_logprob", logprob, prog_bar=True, logger=True)
        self.log("val_cov", cov, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.log("val_fpnd", fpndv, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_mmd", mmd, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        self.plot = plotting(model=self,gen=z_scaled.reshape(-1,self.n_part*self.n_dim),true=true_scaled.reshape(-1,self.n_part*self.n_dim),config=self.config,step=self.global_step,nf=z_scaled.reshape(-1,self.n_part*self.n_dim),
            logger=self.logger.experiment,p=self.config["parton"])
        self.plot.plot_mom(self.global_step)
        try:
            self.plot.plot_mass(m=m_c.cpu().numpy(), m_t=m_t.cpu().numpy(), save=None, bins=50, quantile=True, plot_vline=False)
            # self.plot.plot_2d(save=True)
        #     self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,
        #                          m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc()
        self.flow = self.flow.to("cuda")
        