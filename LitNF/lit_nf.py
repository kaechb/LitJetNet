from distutils.command.config import config
import os
import time
import traceback

import matplotlib.pyplot as plt
import nflows as nf
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p
from nflows.flows import base
from nflows.nn import nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_random_binary_mask
from torch import nn
from torch.nn import functional as FF
from torch.nn.functional import leaky_relu, sigmoid
from torch.autograd import Variable
import torch.autograd as autograd

from helpers import CosineWarmupScheduler, mass
from plotting import *


class Gen(nn.Module):
    def __init__(
        self,
        n_dim=3,
        l_dim=10,
        hidden=300,
        num_layers=3,
        num_heads=1,
        n_part=5,
        fc=False,
        dropout=0.5,
        no_hidden=True,
        norm=False
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.no_hidden = no_hidden
        self.fc = fc
        if fc:
            self.l_dim *= n_part
            self.embbed_flat = nn.Linear(n_dim * n_part, l_dim)
            self.flat_hidden = nn.Linear(l_dim, hidden)
            self.flat_hidden2 = nn.Linear(hidden, hidden)
            self.flat_hidden3 = nn.Linear(hidden, hidden)
            self.flat_out = nn.Linear(hidden, n_dim * n_part)
        else:
            self.embbed = nn.Linear(n_dim, l_dim)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=l_dim,
                    nhead=num_heads,
                    batch_first=True,
                    norm_first=norm,
                    dim_feedforward=hidden,
                    dropout=dropout,
                ),
                num_layers=num_layers,
            )
            self.hidden = nn.Linear(l_dim, hidden)
            self.hidden2 = nn.Linear(hidden, hidden)
            self.hidden3 = nn.Linear(hidden, hidden)
            self.dropout = nn.Dropout(dropout / 2)
            self.out = nn.Linear(hidden, n_dim)
            self.out2 = nn.Linear(l_dim, n_dim)

            self.out_flat = nn.Linear(hidden, n_dim * n_part)

    def forward(self, x, mask=None):

        if self.fc:
            x = x.reshape(len(x), self.n_part * self.n_dim)
            x = self.embbed_flat(x)
            x = leaky_relu(self.flat_hidden(x))
            #             x = self.dropout(x)
            x = self.flat_out(x)
            x = x.reshape(len(x), self.n_part, self.n_dim)
        else:
            x = self.embbed(x)
            x = self.encoder(x, src_key_padding_mask=mask)
            if not self.no_hidden == True:

                x = leaky_relu(self.hidden(x))
                x = self.dropout(x)
                x = leaky_relu(self.hidden2(x))
                x = self.dropout(x)
                x = self.out(x)
            elif self.no_hidden == "more":
                x = leaky_relu(self.hidden(x))
                x = self.dropout(x)
                x = leaky_relu(self.hidden2(x))
                x = self.dropout(x)
                x = leaky_relu(self.hidden3(x))
                x = self.dropout(x)

            else:
                x = leaky_relu(x)
                x = self.out2(x)
        return x


class Disc(nn.Module):
    def __init__(
        self,
        n_dim=3,
        l_dim=10,
        hidden=300,
        num_layers=3,
        num_heads=1,
        n_part=2,
        fc=False,
        dropout=0.5,
        mass=False,
        clf=False,
        norm=False
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        #         l_dim=n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.fc = fc
        self.clf = clf

        if fc:
            self.l_dim *= n_part
            self.embbed_flat = nn.Linear(n_dim * n_part, l_dim)
            self.flat_hidden = nn.Linear(l_dim, hidden)
            self.flat_hidden2 = nn.Linear(hidden, hidden)
            self.flat_hidden3 = nn.Linear(hidden, hidden)
            self.flat_out = nn.Linear(hidden, 1)
        else:
            self.embbed = nn.Linear(n_dim, l_dim)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.l_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden,
                    dropout=dropout,
                    norm_first=norm,
                    activation=lambda x: leaky_relu(x, 0.2),
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
            self.hidden = nn.Linear(l_dim + int(mass), 2 * hidden)
            self.hidden2 = nn.Linear(2 * hidden, hidden)
            self.out = nn.Linear(hidden, 1)

    def forward(self, x, m=None, mask=None):

        if self.fc == True:
            x = x.reshape(len(x), self.n_dim * self.n_part)
            x = self.embbed_flat(x)
            x = leaky_relu(self.flat_hidden(x), 0.2)
            x = leaky_relu(self.flat_hidden2(x), 0.2)
            x = self.flat_out(x)
        else:
            x = self.embbed(x)
            if self.clf:
                x = torch.concat((torch.ones_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
                mask = torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device)

                x = self.encoder(x, src_key_padding_mask=mask)
                x = x[:, 0, :]
            else:
                x = self.encoder(x, src_key_padding_mask=mask)
                x = torch.sum(x, axis=1)
            if m is not None:
                x = torch.concat((m.reshape(len(x), 1), x), axis=1)
            x = leaky_relu(self.hidden(x), 0.2)
            x = leaky_relu(self.hidden2(x), 0.2)
            x = self.out(x)
            x = x
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

    def __init__(self, config, hyperopt, num_batches):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.hyperopt = True

        self.start = time.time()
        # self.batch_size=batch_size
        # print(batch_size)
        self.config = config
        self.automatic_optimization = False
        self.freq_d = config["freq"]

        self.wgan = config["wgan"]
        # Metrics to track during the training
        self.metrics = {
            "val_w1p": [],
            "val_w1m": [],
            "val_w1efp": [],
            "val_cov": [],
            "val_mmd": [],
            "val_fpnd": [],
            "val_logprob": [],
            "step": [],
        }
        # Loss function of the Normalizing flows
        self.logprobs = []
        self.n_part = config["n_part"]
        # self.hparams.update(config)
        self.save_hyperparameters()
        self.flows = []
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.alpha = 1
        self.num_batches = int(num_batches)
        self.build_flow()
        self.gen_net = Gen(
            n_dim=self.n_dim,
            hidden=config["hidden"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            no_hidden=config["no_hidden"],
            fc=config["fc"],
            n_part=config["n_part"],
            l_dim=config["l_dim"],
            num_heads=config["heads"],
            norm=config["norm"]
        ).cuda()
        self.dis_net = Disc(
            n_dim=self.n_dim,
            hidden=config["hidden"],
            l_dim=config["l_dim"],
            num_layers=config["num_layers"],
            mass=self.config["mass"],
            num_heads=config["heads"],
            fc=config["fc"],
            n_part=config["n_part"],
            dropout=config["dropout"],
            clf=config["clf"],
            norm=config["norm"]
        ).cuda()
        self.sig = nn.Sigmoid()
        self.fpnds = []
        self.df = pd.DataFrame()
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        for p in self.gen_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        self.gen_net.out.weight.data.uniform_(-0.01,0.01)
        self.gen_net.out.bias.data.fill_(0)
        self.gen_net.out2.weight.data.uniform_(-0.01,0.01)
        self.gen_net.out2.bias.data.fill_(0)
        self.nf_train = True
        self.train_nf = config["max_epochs"] // config["frac_pretrain"]

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module
    
    def plot_mass(self,m_f,m_t=None,postfix=""):
        fig=plt.figure()
        if not m_t==None:
            _,bins,_=plt.hist(m_t.cpu().detach().numpy(),bins=30,label="True",histtype="step")
            plt.hist(m_f.cpu().detach().numpy(),bins=bins,label="Fake",histtype="step")
        else:
            plt.hist(m_f[m_f<500].cpu().detach().numpy(),bins=30,label="True",histtype="step")
        plt.legend()
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        plt.yscale("log")
        self.logger.experiment.add_figure("train_mass"+postfix, fig, global_step=self.current_epoch)
        plt.close()
    
    def plot_class(self,pred_real,pred_fake,mask):
        fig, ax = plt.subplots()
        ax.hist(pred_real.detach().cpu().numpy(), label="real", bins=np.linspace(0, 1, 30) if not self.wgan else 30, histtype="step")
        ax.hist(pred_fake.detach().cpu().numpy(), label="fake", bins=np.linspace(0, 1, 30) if not self.wgan else 30, histtype="step")
        ax.hist(pred_real[mask.sum(1)].detach().cpu().numpy(), label="real<30", bins=np.linspace(0, 1, 30) if not self.wgan else 30, histtype="step")
        ax.hist(pred_fake[mask.sum(1)].detach().cpu().numpy(), label="fake<30", bins=np.linspace(0, 1, 30) if not self.wgan else 30, histtype="step")
        ax.legend()
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        self.logger.experiment.add_figure("class_train", fig, global_step=self.current_epoch)
        plt.close()

    def on_after_backward(self) -> None:
        """This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues"""
        valid_gradients = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            
            self.zero_grad()
            self.counter += 1
            if self.counter > 8:
                raise ValueError("5 nangrads in a row")
        else:
            self.counter = 0

    def build_flow(self):
        K = self.config["coupling_layers"]
        for i in range(K):
            """This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not"""
            mask = create_random_binary_mask(self.n_dim * self.n_part)
            self.flows += [
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet,
                    tails="linear",
                    tail_bound=self.config["tail_bound"],
                    num_bins=self.config["bins"],)]
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim * self.n_part])
        # Creates working flow model from the list of layer modules
        self.flows = CompositeTransform(self.flows)
        # Construct flow model
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)

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
            max_iter_d = (self.config["max_epochs"]) * self.num_batches
            max_iter_g = (self.config["max_epochs"] - self.train_nf//2) * self.num_batches // self.freq_d
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=15 * self.num_batches, max_iters=max_iter_d)
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=15 * self.num_batches // self.freq_d, max_iters=max_iter_g)
        else:
            lr_scheduler_nf = None
            lr_scheduler_d = None
            lr_scheduler_g = None
        if self.config["sched"] != None:
            return [opt_nf, opt_d, opt_g], [lr_scheduler_nf, lr_scheduler_d, lr_scheduler_g]
        else:
            return [opt_nf, opt_d, opt_g]

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

    def scale(self, x,m): 
        x = x.reshape(len(x), self.n_part, self.n_dim)
        # self.data_module.scaler = self.data_module.scaler.to(x.device)
        for i in range(30):
            if self.config["quantile"]:
                x[~m[:,i] ,i, 2] = torch.tensor(self.data_module.ptscalers[i].inverse_transform(x[~m[:,i] ,i, 2].cpu().numpy().reshape(-1,1)).reshape(-1)).to(x.device)
                x[~m[:,i] ,i, :2] = self.data_module.scalers[i].inverse_transform(x[~m[:,i] ,i, :2].float())
            else:
                x[~m[:,i],i,:]= self.data_module.scalers[i].inverse_transform(x[~m[:,i],i,:])
        return x

    def sampleandscale(self, batch, mask, mask_test=None, scale=False):
        """This is a helper function that samples from the flow (i.e. generates a new sample)
        and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
        on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
        because calculating the mass is a non linear transformation and does not commute with the mass calculation"""
        mask=mask.bool()
        if mask_test!=None:
            mask_test=mask_test.bool()
        else:
            mask_test=mask
        with torch.no_grad():
            z = self.flow.sample(len(batch) if not self.config["context_features"] else 1, context=None if not self.config["context_features"]
                                else (~mask_test).sum(1).float().reshape(-1,1)).reshape(len(batch), self.n_part, self.n_dim).detach()
        fake = z + self.gen_net(z, mask=mask_test)  # (1-self.alpha)*
        fake = fake.reshape(len(batch), self.n_part, self.n_dim)
        if scale:
            fake_scaled, z_scaled, true = (self.scale(fake.detach().clone(),mask_test).to(batch.device),
                                        self.scale(z.detach().clone(),mask_test).to(batch.device), 
                                        self.scale(batch.detach().clone(),mask).to(batch.device))
            true = true * (~mask.reshape(len(batch), self.n_part, 1))
            z_scaled = z_scaled * (~mask_test.reshape(len(batch), self.n_part, 1))
            fake_scaled = fake_scaled * (~mask_test.reshape(len(batch), self.n_part, 1))
            return fake.to(batch.device), fake_scaled, true, z_scaled
        else:
            return fake

    def train_wgan(self,  batch,mask,opt_d,sched_d=None):
        batch = batch.reshape(len(batch), self.n_part, self.n_dim)
        fake = self.sampleandscale(batch, mask, scale=False)
        if self.config["mass"]:
            m_t = mass(batch, mask=~mask, canonical=self.config["canonical"])
            m_f = mass(fake, mask=~mask, canonical=self.config["canonical"])
        pred_real = self.dis_net(batch, None if not self.config["mass"] else m_t, mask=mask)
        pred_fake = self.dis_net(fake.detach(), None if not self.config["mass"] else m_f.detach(), mask=mask)
        gradient_penalty = self.compute_gradient_penalty(batch, fake.detach(),mask)
        self.log("gradient penalty", gradient_penalty, logger=True)
        d_loss = -torch.mean(pred_real.view(-1)) + torch.mean(pred_fake.view(-1))
        self.log("d_loss", d_loss, logger=True, prog_bar=True)
        d_loss += gradient_penalty
        self.dis_net.zero_grad()
        self.manual_backward(d_loss)
        if self.global_step > 1:
            opt_d.step()
        if self.global_step == 2:
            print("passed test disc")
        if self.current_epoch%10==0 and self.config["mass"]:
            self.plot_mass(m_t,m_f)
        return pred_real,pred_fake

    def train_disc(self,  batch,mask,opt_d,sched_d=None):  
        batch = batch.reshape(len(batch), self.n_part, self.n_dim)
        fake = self.sampleandscale(batch, mask, scale=False)
        fake = fake.detach()
        if self.config["mass"]:
            m_t = mass(batch, mask=~mask, canonical=self.config["canonical"])
            m_f = mass(fake, mask=~mask, canonical=self.config["canonical"])
        pred_real = self.dis_net(batch.detach(), None if not self.config["mass"] else m_t.detach(), mask=mask)
        pred_fake = self.dis_net(fake.detach(), None if not self.config["mass"] else m_f.detach(), mask=mask)
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)
        pred = torch.vstack((pred_real, pred_fake))
        target = torch.vstack((target_real, target_fake))
        d_loss = nn.MSELoss()(pred, target).mean()
        self.log("d_loss", d_loss, logger=True, prog_bar=True)
        self.dis_net.zero_grad()
        self.manual_backward(d_loss)
        if self.global_step > 0:
            opt_d.step()
        else:
            opt_d.zero_grad()
        if self.current_epoch%10==0 and self.config["mass"]:
            self.plot_mass(m_t,m_f)
        return pred_real,pred_fake

    def train_gen(self,batch,mask,opt_g,sched_g=None):
        fake = self.sampleandscale(batch, mask, scale=False)
        m_f = mass(fake, mask=~mask, canonical=self.config["canonical"])
        pred_fake = self.dis_net(fake, None if not self.config["mass"] else m_f, mask=mask)
        if self.wgan:
            g_loss = -torch.mean(pred_fake.view(-1))
        else:
            target_real = torch.ones_like(pred_fake)
            g_loss = nn.MSELoss()((pred_fake.view(-1)), target_real.view(-1))
        self.gen_net.zero_grad()
        self.manual_backward(g_loss)
        if self.global_step > 10:
            opt_g.step()
        else:
            opt_g.zero_grad()
        self.log("g_loss", g_loss, logger=True, prog_bar=True)

        if self.global_step == 3:
            print("passed test gen")
        if self.current_epoch%10==0:
            self.plot_mass(m_f,postfix="2")

    def compute_gradient_penalty(self,real, fake, mask):
        """Gradient of the critic's scores with respect to mixes of real and fake images.
        Args:
        real: a batch of real images
        fake: a batch of fake images
        Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
        """
        # Mix the images together
        epsilon = torch.rand(len(real), 1, 1, device=real.device, requires_grad=True)
        mixed_images = real * epsilon + fake * (1 - epsilon)
        mixed_mass = mass(mixed_images,mask=mask)
        # Calculate the critic's scores on the mixed images
        mixed_scores = self.dis_net(mixed_images, None if not self.config["mass"] else mixed_mass,mask)
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            #### START CODE HERE ####
            inputs = mixed_images,
            outputs = mixed_scores,
            #### END CODE HERE ####
            # These other parameters have to do with how the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores), 
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        # Penalize the mean squared distance of the gradient norms from 1
        gp = torch.mean(torch.square(gradient_norm - 1))
        return gp

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
        self.metrics["step"].append(self.current_epoch)
        self.df = self.df.append(pd.DataFrame([temp], index=[self.current_epoch]))
        self.df.to_csv(self.logger.log_dir + "result.csv", index_label=["index"])

    def training_step(self, batch, batch_idx):
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        mask = batch[:, 90:].bool()
        batch = batch[:, :90]
        opt_nf, opt_d, opt_g = self.optimizers()
        if self.config["sched"]:
            sched_nf, sched_d, sched_g = self.lr_schedulers()
        # ### NF PART
        if self.config["sched"] != None:
            self.log("lr_g", sched_g.get_last_lr()[-1], logger=True, on_epoch=True)
            self.log("lr_nf", sched_nf.get_last_lr()[-1], logger=True, on_epoch=True)
            self.log("lr_d", sched_d.get_last_lr()[-1], logger=True, on_epoch=True)

        # Only train nf for first few epochs
        if self.current_epoch < self.train_nf:
            if self.config["sched"] != None:
                sched_nf.step()
            nf_loss = -self.flow.to(self.device).log_prob(batch,context=(~mask).sum(1).reshape(-1,1).float() if self.config["context_features"] else None).mean()
            nf_loss /= self.n_dim * self.n_part
            self.flow.zero_grad()
            self.manual_backward(nf_loss)
            opt_nf.step()
            self.log("logprob", nf_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # GAN PART

        if self.config["sched"]:
            sched_d.step()
        if self.wgan:
            pred_real,pred_fake=self.train_wgan(batch=batch,mask=mask,opt_d=opt_d,sched_d=sched_d if self.config["sched"] else None )
        else:
            pred_real,pred_fake=self.train_disc(batch=batch,mask=mask,opt_d=opt_d,sched_d=sched_d if self.config["sched"] else None )
        if (self.current_epoch > self.train_nf//2  and self.global_step % self.freq_d < 2) or self.global_step <= 3:
            if self.config["sched"]:
                sched_g.step()
            self.train_gen(batch=batch,mask=mask,opt_g=opt_g,sched_g=sched_g if self.config["sched"] else None )
        # Control plot train
        if self.current_epoch % 5 == 0 and self.current_epoch > self.train_nf / 2:
            self.plot_class(pred_real=pred_real,pred_fake=pred_fake,mask=mask)

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""

        mask = batch[:, 90:].cpu().bool()
        mask_test = self.sample_n(mask).bool()

        batch = batch[:, :90].cpu()
        self.dis_net.train()
        self.gen_net.train()
        self.flow.train()
        # self.data_module.scaler.to("cpu")
        batch = batch.to("cpu")
        self.flow = self.flow.to("cpu")
        self.dis_net = self.dis_net.cpu()
        self.gen_net = self.gen_net.cpu()

        with torch.no_grad():
            gen, fake_scaled, true_scaled, z_scaled = self.sampleandscale(batch,mask, mask_test, scale=True)
            if self.config["mass"]:
                m_t = mass(batch, mask=~mask, canonical=self.config["canonical"])
                m_f = mass(gen, mask=~mask_test, canonical=self.config["canonical"])
                fig=plt.figure()
                _,bins,_=plt.hist(m_t.detach().numpy(),bins=30,label="True")
                plt.hist(m_f.cpu().detach().numpy(),bins=bins,label="Fake")
                plt.legend()
                self.logger.experiment.add_figure("train_mass", fig, global_step=self.current_epoch)
                plt.close()
        pred_real = self.dis_net(batch.reshape(len(batch), self.n_part, self.n_dim), None if not self.config["mass"] else m_t, mask=mask)
        pred_fake = self.dis_net(gen, None if not self.config["mass"] else m_f, mask=mask_test)
        if self.wgan:
            d_loss = -torch.mean(pred_real.view(-1)) + torch.mean(pred_fake.view(-1))
            g_loss = -torch.mean(pred_fake.view(-1))
        else:
            target_real = torch.ones_like(pred_real)
            target_fake = torch.zeros_like(pred_fake)
            pred = torch.vstack((pred_real, pred_fake))
            target = torch.vstack((target_real, target_fake))
            d_loss = nn.MSELoss()(pred, target).mean()
            g_loss = nn.MSELoss()((pred_fake.view(-1)), target_real.view(-1))
        true_scaled=true_scaled.reshape(-1,30,3)*(~mask).reshape(-1,30,1)
        fake_scaled=fake_scaled.reshape(-1,30,3)*(~mask_test).reshape(-1,30,1)
        true_scaled, fake_scaled, z_scaled = (true_scaled.reshape(-1, 90), fake_scaled.reshape(-1, 90), z_scaled.reshape(-1, 90))
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        m_t = mass(true_scaled, canonical=self.config["canonical"], mask=~mask).cpu()
        m_gen = mass(z_scaled, mask=~mask_test, canonical=self.config["canonical"]).cpu()
        m_c = mass(fake_scaled, mask=~mask_test, canonical=self.config["canonical"]).cpu()
        for i in range(30):
            i = 2 + 3 * i
            # gen[gen[:,i]<0,i]=0
            fake_scaled[fake_scaled[:, i] < 0, i] = 0
            true_scaled[true_scaled[:, i] < 0, i] = 0
        # metrics

        cov, mmd = cov_mmd(fake_scaled.reshape(-1, self.n_part, self.n_dim), true_scaled.reshape(-1, self.n_part, self.n_dim), use_tqdm=False)
        nfcov, nfmmd = cov_mmd(z_scaled.reshape(-1, self.n_part, self.n_dim), true_scaled.reshape(-1, self.n_part, self.n_dim), use_tqdm=False)
        try:
            fpndnf = fpnd(z_scaled.reshape(-1, self.n_part, self.n_dim).numpy(), use_tqdm=False, jet_type=self.config["parton"]) 
            fpndv = fpnd(fake_scaled.reshape(-1, self.n_part, self.n_dim).numpy(), use_tqdm=False, jet_type=self.config["parton"])
        except:
            fpndv = 1000
        self.fpnds.append(fpndv)
        if (np.array(self.fpnds)[-10:] > 5).all() and self.current_epoch > 200:
            print("fpnd to high, stop training")
            raise
        w1m_ = w1m(fake_scaled.reshape(len(batch), self.n_part, self.n_dim), true_scaled.reshape(len(batch), self.n_part, self.n_dim))[0]
        w1p_ = w1p(fake_scaled.reshape(len(batch), self.n_part, self.n_dim), true_scaled.reshape(len(batch), self.n_part, self.n_dim))[0]
        w1efp_ = w1efp(fake_scaled.reshape(len(batch), self.n_part, self.n_dim), true_scaled.reshape(len(batch), self.n_part, self.n_dim))[0]

        temp = {
            "val_fpnd": fpndv,
            "val_fpnd_nf": fpndnf,
            "val_mmd": mmd,
            "val_cov": cov,
            "val_w1m": w1m_,
            "val_w1efp": w1efp_,
            "val_w1p": w1p_,
            "step": self.global_step,
            "g_loss": float(g_loss.numpy()),
            "d_loss": float(d_loss.numpy()),
        }
        print("epoch {}: ".format(self.current_epoch), temp)
        if self.hyperopt and self.global_step > 3:

            if self.current_epoch < self.train_nf:
                with torch.no_grad():
                    logprob = -self.flow.log_prob(batch,context=(~mask).sum(1).reshape(-1,1).float() if self.config["context_features"] else None).mean() / 90
                self.log("val_logprob", logprob, prog_bar=True, logger=True)
                temp["val_logprob"] = float(logprob.numpy())
            try:
                self._results(temp)
            except:
                print("error in results")
            self._summary(temp)

        self.log("hp_metric", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1m", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p", w1p_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1efp", w1efp_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_cov", cov, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_fpnd", fpndv, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_fpnd_nf", fpndnf, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_mmd", mmd, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("d_loss", d_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.plot = plotting(
            model=self, gen=z_scaled, gen_corr=fake_scaled, true=true_scaled, config=self.config, step=self.global_step, logger=self.logger.experiment
        )
        try:
            self.plot.plot_mass(m=m_gen.cpu().numpy(), m_t=m_t.cpu().numpy(), m_c=m_c.cpu().numpy(), save=True, bins=50, quantile=True, plot_vline=False)
            self.plot.plot_class(pred_fake=pred_fake, pred_real=pred_real, bins=50, step=self.current_epoch)
            # self.plot.plot_2d(save=True)
        #             self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc()
        self.flow = self.flow.to("cuda")
        self.gen_net = self.gen_net.to("cuda")
        self.dis_net = self.dis_net.to("cuda")
