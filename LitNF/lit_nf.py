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

from helpers import CosineWarmupScheduler
from plotting import *


class TransformerEncoderLayer2(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super(TransformerEncoderLayer2,self).__init__()
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

class Gen(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,n_part=5,dropout=0.5,gen_mask=True,no_hidden=True):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.no_hidden_gen = no_hidden
        self.gen_mask = gen_mask

        
        self.embbed = nn.Linear(n_dim, l_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=l_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=False,
                dim_feedforward=hidden,
                dropout=dropout,
                activation= "relu"
            ),
            num_layers=num_layers,
        )
        if not self.no_hidden_gen==True:
            self.hidden = nn.Linear(l_dim, hidden)
            self.hidden2 = nn.Linear(hidden, hidden)
        if self.no_hidden_gen=="more":
            self.hidden3 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout / 2)
        self.out = nn.Linear(hidden, n_dim)
        self.out2 = nn.Linear(l_dim, n_dim)
        
        

    def forward(self, x, mask=None):
        if not self.gen_mask:
            mask = None
        x = self.embbed(x)
        x = self.encoder(x, src_key_padding_mask=mask,)#attention_mask.bool()
        if self.no_hidden_gen==True:
            x = leaky_relu(x)
            x = self.out2(x)
        else:
            x = leaky_relu(self.hidden(x))
            x = self.dropout(x)
            x = leaky_relu(self.hidden2(x))
            x = self.dropout(x)
            if self.no_hidden_gen=="more":
                x = leaky_relu(self.hidden3(x))
                x = self.dropout(x)   
            
            x = self.out(x)
        return x


class Disc(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,n_part=2,last_clf = True,dropout=0.5,mass=False,clf=False,bullshitbingo=True,momentum=False):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.last_clf = last_clf
        self.clf = clf
        self.bullshitbingo=bullshitbingo
        self.embbed = nn.Linear(n_dim, l_dim)
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model=self.l_dim,nhead=num_heads,dim_feedforward=hidden,dropout=dropout,
                                        norm_first=False,activation=lambda x: leaky_relu(x, 0.2),batch_first=True),
            num_layers=num_layers,
        )
        self.encoder_class=TransformerEncoderLayer(d_model=self.l_dim,nhead=num_heads,dim_feedforward=hidden,dropout=dropout, norm_first=False,activation=lambda x: leaky_relu(x, 0.2),batch_first=True)
        self.hidden = nn.Linear(l_dim + int(mass)+int(momentum), 2 * hidden)
        self.hidden2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, m=None,p=None, mask=None,noise=0):
        x = self.embbed(x)
        if self.clf:
            if not self.last_clf:
                if self.bullshitbingo and m is not None:
                    x = torch.concat((torch.ones_like(x[:, 0, :]).reshape(len(x), 1, -1)*m.reshape(len(x),1,1), x), axis=1)
                else:
                    x = torch.concat((torch.zeros_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
                if not mask==None:
                    mask = torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device).bool()
                x = self.encoder(x, src_key_padding_mask=mask)
            else:
                x = self.encoder(x, src_key_padding_mask=mask)
                # x = torch.concat((nn.Parameter(torch.zeros_like(x[0,0,:]).reshape(1,1,-1)).expand(len(x),-1,-1),x),dim=1)
                x = torch.concat((torch.zeros_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
                if not mask==None:
                    mask = torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).bool()
                x=self.encoder_class(x,src_key_padding_mask=mask)
            x = x[:, 0, :]
        else:
            x = self.encoder(x, src_key_padding_mask=mask)
            x = torch.sum(x, axis=1)
        if m is not None:
            x = torch.concat((m.reshape(len(x), 1), x), axis=1)
        if p is not None:
            p=p+noise
            x = torch.concat((p.reshape(len(x), 1), x), axis=1)
        x = leaky_relu(self.hidden(x), 0.2)
        x = leaky_relu(self.hidden2(x), 0.2)
        x = self.out(x)
        return x


class TransGan(pl.LightningModule):
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
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.alpha = 1
        self.num_batches = int(num_batches)
        self.build_flow()
        self.gen_net = Gen(n_dim=self.n_dim,hidden=config["hidden"],num_layers=config["num_layers"],dropout=config["dropout"],
                            no_hidden=config["no_hidden_gen"],n_part=config["n_part"],l_dim=config["l_dim"],
                            num_heads=config["heads"],gen_mask=config["gen_mask"]).cuda()
        self.dis_net = Disc(n_dim=self.n_dim,hidden=config["hidden"],l_dim=config["l_dim"],num_layers=config["num_layers"],
                            mass=self.config["mass"],num_heads=config["heads"],last_clf=config["last_clf"],n_part=config["n_part"],momentum=self.config["momentum"],
                            dropout=config["dropout"],clf=config["clf"],bullshitbingo=config["bullshitbingo"]).cuda()
        self.sig = nn.Sigmoid()
        self.df = pd.DataFrame()
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)

        for p in self.gen_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
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

    def sampleandscale(self, batch, mask=None, scale=False):
        """This is a helper function that samples from the flow (i.e. generates a new sample)
        and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
        on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
        because calculating the mass is a non linear transformation and does not commute with the mass calculation"""
        assert mask.dtype==torch.bool
        with torch.no_grad():
            if  self.config["context_features"]==1:
                c=mask.clone()
                c = (~c).sum(axis=1).reshape(-1,1).float()
            elif  self.config["context_features"]==2:
                c=mask.clone()
                c = torch.cat(( mass(batch).reshape(-1,1),(~c).sum(axis=1).reshape(-1,1)),dim=1).float()                
            else: 
                c=None
            z = self.flow.sample(len(batch) if self.config["context_features"]==0 else 1,context=c).reshape(len(batch), self.n_part, self.n_dim)
        fake = z + self.gen_net(z, mask=mask)
        # fake = fake*((~mask).reshape(len(batch),30,1).float()) #somehow this gives nangrad
        fake[mask]=0
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            z_scaled = z.clone()
            if self.config["scalingbullshit"]:
                self.data_module.scaler = self.data_module.scaler.to(batch.device)
                fake_scaled=self.data_module.scaler.inverse_transform(fake_scaled)
                z_scaled=self.data_module.scaler.inverse_transform(z_scaled)
                true=self.data_module.scaler.inverse_transform(true)
            else:
                for i in range(self.n_part):
                    if self.config["quantile"]:
                        self.data_module.scaler = self.data_module.scalers[i].to(batch.device)
                        fake_scaled[:, i, :2] = self.data_module.scalers[i].inverse_transform(fake[:, i, :2])
                        z_scaled[:, :, :2] = self.data_module.scalers[i].inverse_transform(z[:, i, :2])
                        true[:, :, :2] = self.data_module.scalers[i].inverse_transform(true[:, i, :2])
                        fake_scaled[:, :, 2] = torch.tensor(self.data_module.ptscalers[i].inverse_transform(fake[:, i, 2].reshape(len(batch), self.n_part).numpy())).float()
                        z_scaled[:, :, 2] = torch.tensor(self.data_module.ptscalers[i].inverse_transform(z[:, i, 2].reshape(len(batch), self.n_part).numpy())).float()
                        true[:, :, 2] = torch.tensor(self.data_module.ptscalers[i].inverse_transform(batch.reshape(len(batch), self.n_part, 3)[:, i, 2].numpy())).float()
                        return fake, fake_scaled, true, z_scaled
                    else:
                        
                        self.data_module.scalers[i] = self.data_module.scalers[i].to(batch.device)
                        fake_scaled[:,i,:] = self.data_module.scalers[i].inverse_transform(fake[:,i,:])
                        z_scaled[:,i,:] = self.data_module.scalers[i].inverse_transform(z[:,i,:])
                        true[:,i,:] = self.data_module.scalers[i].inverse_transform(batch[:,i,:])
            fake_scaled[mask]=0
            z_scaled[mask]=0
            return fake, fake_scaled, true, z_scaled
        else:
            return fake

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        opt_nf = torch.optim.AdamW(self.flow.parameters(), lr=self.config["lr_nf"])
        if self.config["opt"] == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9))
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.config["lr_d"], betas=(0, 0.9))
        elif self.config["opt"] == "AdamW":
            opt_g = torch.optim.AdamW(self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9))
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.config["lr_d"], betas=(0, 0.9))
        else:
            opt_g = torch.optim.RMSprop(self.gen_net.parameters(), lr=self.config["lr_g"])
            opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.config["lr_d"])
        if self.config["sched"] == "cosine":
            lr_scheduler_nf = CosineWarmupScheduler(opt_nf, warmup=1, max_iters=10000000 * self.config["freq"])
            max_iter_d = (self.config["max_epochs"] - self.train_nf // 2) * self.num_batches
            if self.config["bullshitbingo2"]:
                self.freq_d+=1
            max_iter_g = (self.config["max_epochs"] - self.train_nf) * self.num_batches // (self.freq_d-1)
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter_d)
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter_g)
            if self.config["bullshitbingo2"]:
                self.freq_d-=1
        elif self.config["sched"] == "cosine2":
            lr_scheduler_nf = CosineWarmupScheduler(opt_nf, warmup=1, max_iters=10000000 * self.config["freq"])
            max_iter_d = (self.config["max_epochs"] - self.train_nf // 2) * self.num_batches
            if self.config["bullshitbingo2"]:
                self.freq_d+=1
            max_iter_g = (self.config["max_epochs"] - self.train_nf) * self.num_batches//(self.freq_d-1)
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter_d//3 )#15,150 // 3
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=self.config["warmup"] * self.num_batches //(self.freq_d-1), max_iters=max_iter_g//3)#  // 3
            if self.config["bullshitbingo2"]:
                self.freq_d-=1
        else:
            lr_scheduler_nf = None
            lr_scheduler_d = None
            lr_scheduler_g = None
        if self.config["sched"] != None:
            return [opt_nf, opt_d, opt_g], [lr_scheduler_nf, lr_scheduler_d, lr_scheduler_g]
        else:
            return [opt_nf, opt_d, opt_g]

   
    def training_step(self, batch, batch_idx):
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        mask = batch[:, 90:].bool()
        noise=torch.normal(torch.zeros(len(batch),device="cuda"),2*torch.exp(torch.tensor(-20*self.current_epoch/self.config["max_epochs"],device="cuda")))
        batch = batch[:, :90]
        if  self.config["context_features"]==1:
                c=mask.clone()
                c = (~c).sum(axis=1).reshape(-1,1).float()
        elif  self.config["context_features"]==2:
                c=mask.clone()
                c = torch.cat(( mass(batch).reshape(-1,1),(~c).sum(axis=1).reshape(-1,1)),dim=1).float()                
        else: 
            c=None
        opt_nf, opt_d, opt_g = self.optimizers()
        if self.config["sched"]:
            sched_nf, sched_d, sched_g = self.lr_schedulers()
        # ### NF PART
        if self.config["sched"] != None:
            self.log("lr_g", sched_g.get_last_lr()[-1], logger=True, on_epoch=True,on_step=False)
            self.log("lr_nf", sched_nf.get_last_lr()[-1], logger=True, on_epoch=True,on_step=False)
            self.log("lr_d", sched_d.get_last_lr()[-1], logger=True, on_epoch=True,on_step=False)

        if self.current_epoch < 4*self.train_nf:
            if self.config["sched"] != None:
                sched_nf.step()
            nf_loss = -self.flow.to(self.device).log_prob(batch,context=c).mean()
            nf_loss /= self.n_dim * self.n_part
            opt_nf.zero_grad()
            self.manual_backward(nf_loss)
            opt_nf.step()
            self.log("logprob", nf_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        ### GAN PART
        if self.current_epoch >= self.train_nf / 2 or self.global_step == 1:
            if self.config["sched"] != None:
                sched_d.step()
            batch = batch.reshape(len(batch), self.n_part, self.n_dim)
            # batch = batch*((~mask).reshape(-1,30,1).int())
            batch[mask]=0
            fake = self.sampleandscale(batch, mask, scale=False)
            
            c_t=[None,None]
            c_f=[None,None]
            if self.config["momentum"]:
                c_t[1]=batch.reshape(len(batch),self.n_part,self.n_dim)[:,:,2].sum(1)
                c_f[1]=fake.reshape(len(batch),self.n_part,self.n_dim)[:,:,2].sum(1)
            if self.config["mass"]:
                c_t[0] = mass(batch,self.config["canonical"],)
                c_f[0] = mass(fake,self.config["canonical"],)
            pred_real = self.dis_net(batch, m=c_t[0],p=c_t[1], mask=mask,noise=noise)
            pred_fake = self.dis_net(fake.detach(),m=c_f[0],p=c_f[1], mask=mask,noise=noise)
            if self.wgan:
                # gradient_penalty = self.compute_gradient_penalty(self.dis_net, batch, fake.detach(), mask, 1)
                gradient_penalty = self.compute_gradient_penalty2(batch, fake.detach(), pred_real, pred_fake,None)
                self.log("gradient penalty", gradient_penalty, logger=True)
                d_loss = -torch.mean(pred_real.view(-1)) + torch.mean(pred_fake.view(-1)) + gradient_penalty
            else:
                target_real = torch.ones_like(pred_real)
                target_fake = torch.zeros_like(pred_fake)
                pred = torch.vstack((pred_real, pred_fake))
                target = torch.vstack((target_real, target_fake))
                d_loss = nn.MSELoss()(pred, target).mean()
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            if self.global_step > 10:
                opt_d.step()
            else:
                opt_d.zero_grad()

            self.log("d_loss", d_loss, logger=True, prog_bar=True)
            if self.global_step < 2:
                print("passed test disc")
            
        if (self.current_epoch > self.train_nf and self.global_step % self.freq_d < 2) or self.global_step <= 3:
            opt_g.zero_grad()
            fake = self.sampleandscale(batch, mask, scale=False)#~(mask.bool())
            c_f=[None,None]
            if self.config["momentum"]:
                c_f[1]=fake.reshape(len(batch),self.n_part,3)[:,:,2].sum(1)
            if self.config["mass"]:
                c_f[0] = mass(fake,self.config["canonical"],)
            pred_fake = self.dis_net(fake, m=c_f[0],p=c_f[1], mask=mask,noise=noise)

            target_real = torch.ones_like(pred_fake)
            if self.wgan:
                g_loss = -torch.mean(pred_fake.view(-1))
            else:
                g_loss = nn.MSELoss()((pred_fake.view(-1)), target_real.view(-1))
            self.manual_backward(g_loss)
            if self.global_step > 10:
                opt_g.step()
            else:
                opt_g.zero_grad()
            self.log("g_loss", g_loss, logger=True, prog_bar=True)
            if self.config["sched"]!=None:
                sched_g.step()
            if self.global_step < 3:
                print("passed test gen")
            # Control plot train
            if self.current_epoch % 5 == 0 and self.current_epoch > self.train_nf / 2:
                self.plot.plot_scores(pred_real,pred_fake,train=True,step=self.global_step)
                

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        mask = batch[:, 90:].cpu().bool()
        batch = batch[:, :90].cpu()
        if  self.config["context_features"]==1:
            c=mask.clone()
            c = (~c).sum(axis=1).reshape(-1,1).float()
        elif  self.config["context_features"]==2:
            c=mask.clone()
            c = torch.cat(( mass(batch).reshape(-1,1),(~c).sum(axis=1).reshape(-1,1)),dim=1).float()                
        else: 
            c=None
        self.dis_net.train()
        self.gen_net.train()
        self.flow.train()
        mask_test=self.sample_n(mask)
        batch = batch.to("cpu")
        self.flow = self.flow.to("cpu")
        self.dis_net = self.dis_net.cpu()
        self.gen_net = self.gen_net.cpu()
        
        with torch.no_grad():
            logprob = -self.flow.log_prob(batch,context=c).mean() / 90
            batch = batch.reshape(len(batch),30,3)
            gen, fake_scaled, true_scaled, z_scaled = self.sampleandscale(batch,mask_test, scale=True)#mask_test
            batch[mask]=0
            c_t=[None,None]
            c_f=[None,None]
            if self.config["momentum"]:
                c_t[1]=batch.reshape(len(batch),self.n_part,3)[:,:,2].sum(1)
                c_f[1]=gen.reshape(len(batch),self.n_part,3)[:,:,2].sum(1)
            if self.config["mass"]:
                c_t[0] = mass(batch,self.config["canonical"],)
                c_f[0] = mass(gen,self.config["canonical"],)
            scores_real = self.sig(self.dis_net(batch, m=c_t[0],p=c_t[1], mask=mask))
            scores_fake = self.sig(self.dis_net(gen,m=c_f[0],p=c_f[1], mask=mask))
            
        
        true_scaled[mask]=0
        fake_scaled[mask_test] = 0
        z_scaled[mask_test]  = 0
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        m_t = mass(true_scaled,self.config["canonical"])
        m_gen = mass(z_scaled, self.config["canonical"])
        m_c = mass(fake_scaled, self.config["canonical"])
        
        for i in range(30):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = 0
            z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        # Some metrics we track
        cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)
        cov_nf, mmd_nf = cov_mmd(true_scaled,z_scaled,  use_tqdm=False)
        try:
            
            fpndv = fpnd(fake_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.config["parton"])
        except:
            fpndv = 1000

        w1m_ = w1m(fake_scaled, true_scaled)[0]
        w1p_ = w1p(fake_scaled, true_scaled)[0]
        w1efp_ = w1efp(fake_scaled, true_scaled)[0]

        if fpndv<0.15 and w1m_<0.001 and not self.refinement:
            _,opt_d,opt_g=self.optimizers()
            for g in opt_d.param_groups:
                g['lr'] *= 0.01
            for g in opt_g.param_groups:
                g['lr'] *= 0.01
            self.refinement=True
        self.w1ms.append(w1m_)
        self.fpnds.append(fpndv)
        if (np.array([self.fpnds])[-4:] > 4).all() and self.current_epoch > self.config["max_epochs"]/2 and not self.config["bullshitbingo2"] or (np.array([self.w1ms])[-4:] > 0.006).all() and self.current_epoch > self.config["max_epochs"]/2 and not self.config["bullshitbingo2"]:
            print("no convergence, stop training")
            raise

        temp = {"val_logprob": float(logprob.numpy()),"val_fpnd": fpndv,"val_mmd": mmd,"val_cov": cov,"val_w1m": w1m_,
                "val_w1efp": w1efp_,"val_w1p": w1p_,"step": self.global_step,}
        print("epoch {}: ".format(self.current_epoch), temp)
        if self.hyperopt and self.global_step > 3:
            try:
                self._results(temp)
            except:
                print("error in results")
            summary = self._summary(temp)
        self.log("hp_metric", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1m", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p", w1p_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1efp", w1efp_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_logprob", logprob, prog_bar=True, logger=True)
        self.log("val_cov", cov, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_cov_nf", cov_nf, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_fpnd", fpndv, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_mmd", mmd, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_mmd_nf", mmd_nf, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.plot = plotting(model=self,gen=fake_scaled.reshape(-1,self.n_part*self.n_dim),true=true_scaled.reshape(-1,self.n_part*self.n_dim),config=self.config,step=self.global_step,
            logger=self.logger.experiment,p=self.config["parton"])
        self.plot.plot_scores(scores_real,scores_fake,train=False,step=self.global_step)
        self.plot.plot_mom(self.global_step)
        try:
            self.plot.plot_mass(m=m_gen.cpu().numpy(), m_t=m_t.cpu().numpy(), save=None, bins=50, quantile=True, plot_vline=False)
            # self.plot.plot_2d(save=True)
        #     self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,
        #                          m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc()
        self.flow = self.flow.to("cuda")
        self.gen_net = self.gen_net.to("cuda")
        self.dis_net = self.dis_net.to("cuda")
