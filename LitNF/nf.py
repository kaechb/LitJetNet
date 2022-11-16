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
from performer_pytorch import SelfAttention
from helpers import CosineWarmupScheduler, Scheduler
from plotting import *

class TransformerEncoderLayer2(torch.nn.TransformerEncoderLayer):
    def __init__(self,d_model,nhead,
                batch_first,
                norm_first,
                dim_feedforward,
                dropout,
                activation):
        super(TransformerEncoderLayer2,self).__init__(d_model=d_model,nhead=nhead,
                batch_first=False,
                norm_first=norm_first,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation)
        self._sa_block=SelfAttention(dim=d_model,heads=nhead,nb_features=dim_feedforward,causal=False)
    def forward(self, src: Tensor, src_mask= None,src_key_padding_mask = None):
        if (src.dim() == 3 and not self.norm_first and not self.training and
            self.self_attn.batch_first and
            self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
            self.norm1.eps == self.norm2.eps and
            src_mask is None and
                not (src.is_nested and src_key_padding_mask is not None)):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,  # TODO: split into two args
                )
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1( self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2( self._ff_block(x))
        return x

class Disc(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,n_part=2,dropout=0.5,mass=False,clf=False,affine_add=True,momentum=False):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        
        self.affine_add=affine_add
        self.embbed = nn.Linear(n_dim, l_dim)
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer2(d_model=self.l_dim,nhead=num_heads,dim_feedforward=hidden,dropout=dropout,batch_first=False,
                                        norm_first=False,activation=lambda x: leaky_relu(x, 0.2)),
            num_layers=num_layers,
        )
        
        # if not self.affine_add:
            
        self.hidden = nn.Linear(l_dim , 2 * hidden)
        # else:
        #     self.cond = nn.Linear(max(int(mass)+int(momentum),1),l_dim)
        #     self.hidden = nn.Linear(l_dim, 2 * hidden)
        self.hidden2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.mass=mass
        self.momentum=momentum

    def forward(self, x, m=None,p=None, mask=None,noise=0):
        x = self.embbed(x)
        with torch.no_grad():
            mask = torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device).bool()
            x = torch.concat((torch.zeros_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
        # assert mask.shape[1]==x.shape[1]  
        x = x.transpose(0,1)
        x = self.encoder(x, src_key_padding_mask=mask.bool())#
        x = x.transpose(0,1)
        x = x[:, 0, :]
        x = leaky_relu(self.hidden(x), 0.2)
        x = leaky_relu(self.hidden2(x), 0.2)
        x = self.out(x)
        return x

class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_nodes = config["hidden"]
        self.n_dim =config["n_dim"]
        self.l_dim =config["l_dim"]*config["heads"]
        self.n_part =config["n_part"]
        self.embbed =nn.Linear(self.n_dim, self.l_dim)
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer2(d_model=self.l_dim,nhead=config["heads"],dim_feedforward=config["hidden"],dropout=config["dropout"],batch_first=False,
                                        norm_first=False,activation=lambda x: leaky_relu(x, 0.2)),
            num_layers=config["num_layers"],
        )
        self.hidden = nn.Linear(self.l_dim ,config["hidden"])
        self.hidden2 = nn.Linear( self.hidden_nodes, self.hidden_nodes)
        self.out = nn.Linear(self.hidden_nodes, config["context_dim"])

    def forward(self, x, mask=None):
        x = self.embbed(x)
        mask = torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device).bool()
        x = torch.concat((torch.zeros_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
        x = x.transpose(0,1)
        x = self.encoder(x, src_key_padding_mask=mask.bool())
        x = x.transpose(0,1)
        x = x[:, 0, :]
        x = leaky_relu(self.hidden(x), 0.2)
        x = leaky_relu(self.hidden2(x), 0.2)
        x = self.out(x)
        return x

class NF(pl.LightningModule):
    def create_resnet(self, in_features, out_features):
        """This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third"""
        c = self.config["context_dim"]
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
        if "momentum" not in config.keys():
            config["momentum"]=False
        if "affine_add" not in config.keys():
            config["affine_add"]=False
        self.config = config
        # Loss function of the Normalizing flows
        self.save_hyperparameters()
        self.n_dim=config["n_dim"]
        self.n_part=config["n_part"]
        self.alpha = 1
        self.num_batches = int(num_batches)
        self.dis_net = Disc(n_dim=self.n_dim,hidden=config["hidden"],l_dim=config["l_dim"]*config["heads"],num_layers=config["num_layers"],
                            mass=self.config["mass"],num_heads=config["heads"],n_part=config["n_part"],momentum=self.config["momentum"],
                            dropout=config["class_dropout"],affine_add=config["affine_add"]).cuda()
        self.sig = nn.Sigmoid()
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        self.refinement=False
        self.hyperopt = True
        self.start = time.time()
        self.config = config
        self.automatic_optimization = False
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
        self.encode= Encoder(config)
        self.build_flow()
        self.reset_counter=0
        self.automatic_optimization = False
        self.df = pd.DataFrame()
        self.error=0
        self.n_current=2
        self.refinement=False
        self.train_g=False
        self.streak=0
        self.losses_g=np.ones(5)
        self.g_counter=0
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
            mask = create_random_binary_mask(self.n_dim)
            self.flows += [PiecewiseRationalQuadraticCouplingTransform(mask=mask,transform_net_create_fn=self.create_resnet, tails="linear",tail_bound=self.config["tail_bound"],num_bins=self.config["bins"],)]
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim])
        # Creates working flow model from the list of layer modules
        self.flows = CompositeTransform(self.flows)
        # Construct flow model
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)

    

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        opt_nf = torch.optim.AdamW(self.flow.parameters(), lr=self.config["lr_nf"])
    
        opt_g = torch.optim.Adam(self.encode.parameters(), lr=self.config["lr_g"], betas=(0, 0.9))
        opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.config["lr_d"], betas=(0, 0.9))
        return  [opt_nf,opt_d, opt_g]

    def nf_loss(self,batch,c):
        nf_loss = -self.flow.to(self.device).log_prob(batch,context=c).mean()
        nf_loss /= self.n_dim
        return nf_loss

    def condition(self,batch,indices,mask):
        c=torch.repeat_interleave(self.encode(batch,mask=mask),self.n_current,0).to(batch.device)
        return c

    def sample(self,batch,c,indices,mask):
        empty=torch.zeros((len(c),self.n_dim),device=c.device)
        rest=torch.ones((len(batch),self.n_part),device=mask.device).bool()
        rest[:,:self.n_current]=torch.zeros((len(batch),self.n_current),device=mask.device).bool()
        mask=mask & rest
        self.flow.eval()
        for param in self.flow.parameters():
            param.requires_grad = False
        worked=False
        counter=0
        while not worked:
            try:
                fake = self.flow.sample(1,c[indices])
                worked=True
            except:
                counter+=1
                if counter>10:
                    print("10 errors while sampling, abort")
                    raise
                
        empty[indices,:]=fake.reshape(-1,self.n_dim)
        fake=torch.zeros((len(mask),self.n_part,self.n_dim),device=mask.device)
        fake[:,:self.n_current,:]=empty.reshape(-1,self.n_current,self.n_dim)
        return fake,mask

    def critic(self,batch,fake,mask,gen=False):
        if gen:
            pred = self.dis_net(fake.reshape(len(batch),self.n_part,self.n_dim), mask=mask)
            target = torch.ones_like(pred)
        else:
            pred_fake = self.dis_net(fake.detach(), mask=mask)
            pred_real = self.dis_net(batch, mask=mask)
            target_fake = torch.zeros_like(pred_fake)
            target_real = torch.ones_like(pred_real)
            pred = torch.vstack((pred_real, pred_fake))
            target = torch.vstack((target_real, target_fake))
        d_loss = nn.MSELoss()(pred, target).mean()
        if gen:
            return d_loss 
        else:
            return d_loss,pred_real.detach().cpu().numpy(),pred_fake.detach().cpu().numpy()

    def training_step(self, batch, batch_idx):
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        opt_nf,opt_d, opt_g = self.optimizers()
        mask = batch[:, self.n_part*self.n_dim:].bool()
        if self.error>10:
            print("10 errors in a row, stopping training")
            raise
        batch =batch[:, : self.n_part*self.n_dim].reshape(len(batch),self.n_part,self.n_dim)
        batch[mask] = 0
        flat_batch=batch.reshape(len(batch),self.n_part,self.n_dim)[:,:self.n_current,:].reshape(len(batch)*self.n_current,self.n_dim)
        
        indices=(flat_batch!=0).all(axis=1)
        flat_batch=flat_batch[indices,:]
        c=self.condition(batch.reshape(len(batch),self.n_part,self.n_dim),indices,mask)

        nf_loss=self.nf_loss(flat_batch,c[indices])      
        opt_nf.zero_grad()
        self.manual_backward(nf_loss)
        opt_nf.step()

        c=self.condition(batch.reshape(len(batch),self.n_part,self.n_dim),indices,mask)
        fake,mask = self.sample(batch,c,indices,mask)
        d_loss,pred_t,pred_f=self.critic(batch,fake.detach(),mask,gen=False)
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        if d_loss<0.15 and self.reset_counter>1000 and not self.train_g:
            print("start training with {} particles".format(self.n_current))
            self.train_g=True
        else:
            self.reset_counter+=1
        if self.train_g:
            fake,mask = self.sample(batch,c,indices,mask)
            g_loss=self.critic(batch,fake,mask,gen=True)
            opt_g.zero_grad()        
            self.manual_backward(g_loss)
            opt_g.step()
            self.log("g_loss", g_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.losses_g[self.g_counter%5]=g_loss.cpu().detach().numpy()
            self.g_counter+=1

            if np.mean(self.losses_g)<0.3 and self.n_current<self.n_part :
               
                self.plot = plotting(model=self,gen=fake[:,:self.n_current,:self.n_dim].detach().cpu(),true=batch[:,:self.n_current,:self.n_dim].detach().cpu(),config=self.config,step=self.global_step,n=self.n_current,
                    logger=self.logger.experiment,p=self.config["parton"])
                try:
                    
                    self.plot.plot_mass(save=None, bins=10)
                    self.plot.plot_scores(pred_t,pred_f,True,self.global_step)
                except:
                    print("error while plotting pre-increase")
                    traceback.print_exc()
                self.n_current+=1
                print("increased number particles to ",self.n_current)
                self.reset_counter=0
                self.train_g=False
               
              
         

        self.log("logprob", nf_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("d_loss", d_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for param in self.flow.parameters():
            param.requires_grad = True
        
              

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        mask = batch[:, 90:].cpu().bool()
        rest=torch.ones((len(batch),self.n_part),device=mask.device).bool()
        rest[:,:self.n_current]=torch.zeros((len(batch),self.n_current),device=mask.device).bool()
        mask=mask & rest
        batch = batch[:, :90].cpu()
        empty = torch.zeros_like(batch)
        self.encode=self.encode.to("cpu")
        self.flow.eval()
        batch = batch.to("cpu")
        self.flow = self.flow.to("cpu")
        
        
        flat_batch=batch.reshape(len(batch),self.n_part,self.n_dim)[:,:self.n_current,:].reshape(len(batch)*self.n_current,self.n_dim)
        indices=(flat_batch!=0).all(axis=1)
        c=self.condition(batch.reshape(len(batch),self.n_part,self.n_dim),indices,mask)
        fake,mask = self.sample(batch,c,indices,mask)

        
        with torch.no_grad():
            logprob = -self.flow.log_prob(flat_batch[indices],context=c[indices]).mean() / self.n_dim
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            z_scaled=self.data_module.scaler.inverse_transform(fake)
            true_scaled=self.data_module.scaler.inverse_transform(batch.reshape(len(batch),self.n_part, self.n_dim) )
            z_scaled[mask]=0
        m_t = mass(true_scaled[:,:self.n_current,:self.n_dim])
        m_c = mass(z_scaled[:,:self.n_current,:self.n_dim])
        
        for i in range(self.n_part):
            z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        # Some metrics we track
        cov, mmd = cov_mmd( true_scaled[:,:self.n_current],z_scaled[:,:self.n_current], use_tqdm=False)
        
        try:
            
            fpndv = fpnd(z_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.config["parton"])
        except:
            fpndv = 1000

        w1m_ = w1m(z_scaled[:,:self.n_current], true_scaled[:,:self.n_current])[0]
        w1p_ = w1p(z_scaled[:,:self.n_current], true_scaled[:,:self.n_current])[0]
        w1efp_ = w1efp(z_scaled[:,:self.n_current], true_scaled[:,:self.n_current])[0]

        
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
        self.plot = plotting(model=self,gen=z_scaled[:,:self.n_current,:self.n_dim],true=true_scaled[:,:self.n_current,:self.n_dim],config=self.config,step=self.global_step,n=self.n_current,
            logger=self.logger.experiment,p=self.config["parton"])
        self.plot.plot_mom(self.global_step)
        try:
            
            self.plot.plot_mass( save=None, bins=50)
            # self.plot.plot_2d(save=True)
        #     self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,
        #                          m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc()
        self.flow = self.flow.to("cuda")
        self.encode=self.encode.to("cuda")