import os
import sys
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
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_random_binary_mask
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as FF
from torch.nn.functional import gelu, leaky_relu, sigmoid

from helpers import CosineWarmupScheduler, Scheduler
from nf import NF

sys.path.insert(1, "/home/kaechben/plots")
from plots import mass, plotting_point_cloud

from plotting import canonical
from transformer import *
from transformer import Disc, Gen


def weights_init(m):
    if m.__class__.__name__.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, clf=False):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.clf = clf

    def forward(self, x, src_mask, attn_mask=None):
        if self.clf:
            attn_mask = None
        else:
            attn_mask = attn_mask.view(-1, x.size(1), x.size(1))
        x = self.norm1(
            x
            + self.attention(
                query=x, key=x, value=x, key_padding_mask=src_mask, attn_mask=attn_mask
            )[0]
        )
        x = self.dropout1(x)
        x = self.norm2(x + self.ffn(x))
        x = self.dropout2(x)
        return x


class Disc(nn.Module):  #
    def __init__(
        self,
        n_dim=3,
        l_dim=25,
        hidden=512,
        num_layers=3,
        num_heads=5,
        n_part=2,
        dropout=0.5,
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim * num_heads
        self.n_part = n_part
        self.embbed = nn.Linear(n_dim, self.l_dim)
        self.hidden = nn.Linear(self.l_dim, 2 * hidden)
        self.hidden2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.pairembed = PairEmbed(
            n_dim,
            [l_dim * num_heads for i in range(num_layers - 2)] + [num_heads],
            normalize_input=False,
            activation="leaky",
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.l_dim,
                    ffn_hidden=hidden,
                    n_head=num_heads,
                    drop_prob=dropout,
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.clf_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.l_dim,
                    ffn_hidden=hidden,
                    n_head=num_heads,
                    drop_prob=dropout,
                    clf=True,
                )
                for _ in range(2)
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.l_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, mask=None):
        # IN (batch,particles,features) OUT (batch,particles,features)
        #
        assert mask.dtype == torch.bool
        u = self.pairembed(x)
        x = self.embbed(x)

        for layer in self.layers:
            x = layer(x, src_mask=mask, attn_mask=u)
        cls_token = self.cls_token
        x = torch.concat((cls_token.expand(x.shape[0], 1, x.shape[-1]), x), axis=1)
        mask = (
            torch.concat(
                (torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1
            )
            .to(x.device)
            .bool()
        )
        for layer in self.clf_layers:
            x = layer(x, src_mask=mask, attn_mask=None)

        x = leaky_relu(self.hidden(x), 0.2)
        x = leaky_relu(self.hidden2(x), 0.2)
        x = self.out(x)
        return x


class Gen(nn.Module):  #
    def __init__(
        self,
        n_dim=3,
        l_dim=25,
        hidden=512,
        num_layers=3,
        num_heads=5,
        n_part=2,
        dropout=0.5,
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim * num_heads
        self.n_part = n_part
        self.embbed = nn.Linear(n_dim, self.l_dim)

        self.hidden = nn.Linear(self.l_dim, 2 * hidden)
        self.hidden2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.pairembed = PairEmbed(
            n_dim,
            [l_dim * num_heads for i in range(num_layers)] + [num_heads],
            normalize_input=False,
            activation="leaky",
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.l_dim,
                    ffn_hidden=hidden,
                    n_head=num_heads,
                    drop_prob=dropout,
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.out = nn.Linear(self.l_dim, n_dim)

    def forward(self, x, mask=None):
        # IN (batch,particles,features) OUT (batch,particles,features)
        #
        assert mask.dtype == torch.bool
        u = self.pairembed(x)
        x = self.embbed(x)
        for layer in self.layers:
            x = layer(x, src_mask=mask, attn_mask=u)
        return self.out(x)


class ParGan(pl.LightningModule):
    def __init__(self, config, num_batches, path="/"):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.start = time.time()
        if "momentum" not in config.keys():
            config["momentum"] = False
        if "affine_add" not in config.keys():
            config["affine_add"] = False
        self.config = config
        self.automatic_optimization = False
        self.freq_d = config["freq"]
        # Loss function of the Normalizing flows
        self.save_hyperparameters()
        self.counter = 0

        self.fpnds = []
        self.w1ms = []
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.n_current = self.n_part
        self.num_batches = int(num_batches)
        self.nf = config["flow_prior"]
        self.wgan_gen = config["wgan_gen"]
        if self.nf:
            self.flow = NF(config=config, num_batches=num_batches).load_from_checkpoint(
                config["load_ckpt"]
            )
        self.gen_net = (
            Gen().cuda()
        )  # nn.Sequential(*[nn.Linear(3,10),nn.GELU(),nn.Linear(10,10),nn.GELU(),nn.Linear(10,3)])#Gen(config).cuda()
        self.dis_net = Disc().cuda()
        self.sig = nn.Sigmoid()
        self.df = pd.DataFrame()
        self.dis_net.apply(weights_init)
        self.gen_net.apply(weights_init)
        self.g_loss = 1000
        self.d_loss = 1000

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def _summary(self, temp):
        self.summary_path = "/beegfs/desy/user/{}/{}/summary.csv".format(
            os.environ["USER"], self.config["name"]
        )
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
        self.df = pd.concat([self.df, pd.DataFrame([temp], index=[self.current_epoch])])
        self.df.to_csv(self.logger.log_dir + "result.csv", index_label=["index"])

    # def on_after_backward(self) -> None:
    #     """This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues"""
    #     valid_gradients = False
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
    #         if not valid_gradients:
    #             break
    #     if not valid_gradients:
    #         print("not valid grads", self.counter)
    #         self.zero_grad()
    #         self.counter += 1
    #         if self.counter > 5:raise ValueError("5 nangrads in a row")
    #         self.stop_train = True
    #     else:
    #         self.counter = 0

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        if self.config["opt"] == "Adam":
            opt_g = torch.optim.Adam(
                self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9)
            )
            opt_d = torch.optim.Adam(
                self.dis_net.parameters(), lr=self.config["lr_d"], betas=(0, 0.9)
            )
        elif self.config["opt"] == "mixed":
            opt_g = torch.optim.Adam(
                self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9)
            )
            opt_d = torch.optim.SGD(self.dis_net.parameters(), lr=self.config["lr_d"])
        else:
            opt_g = torch.optim.RMSprop(
                self.gen_net.parameters(), lr=self.config["lr_g"]
            )
            opt_d = torch.optim.RMSprop(
                self.dis_net.parameters(), lr=self.config["lr_d"]
            )
        if self.config["sched"] == "cosine":
            max_iter_d = (self.config["max_epochs"]) * self.num_batches
            max_iter_g = (
                (self.config["max_epochs"])
                * self.num_batches
                // max(self.freq_d - 1, 1)
            )
            lr_scheduler_d = CosineWarmupScheduler(
                opt_d,
                warmup=self.config["warmup"] * self.num_batches,
                max_iters=max_iter_d,
            )
            lr_scheduler_g = CosineWarmupScheduler(
                opt_g,
                warmup=self.config["warmup"] * self.num_batches,
                max_iters=max_iter_g,
            )

        elif self.config["sched"] == "cosine2":
            max_iter_d = (self.config["max_epochs"]) * self.num_batches
            max_iter_g = (
                (self.config["max_epochs"])
                * self.num_batches
                // max(self.freq_d - 1, 1)
            )
            lr_scheduler_d = CosineWarmupScheduler(
                opt_d,
                warmup=self.config["warmup"] * self.num_batches,
                max_iters=max_iter_d // 3,
            )  # 15,150 // 3
            lr_scheduler_g = CosineWarmupScheduler(
                opt_g,
                warmup=self.config["warmup"]
                * self.num_batches
                // max(self.freq_d - 1, 1),
                max_iters=max_iter_g // 3,
            )  #  // 3
        elif self.config["sched"] == "linear":
            max_iter_d = (self.config["max_epochs"]) * self.num_batches
            max_iter_g = (
                (self.config["max_epochs"])
                * self.num_batches
                // max(1, (self.freq_d - 1))
            )
            lr_scheduler_d = Scheduler(
                opt_d,
                dim_embed=self.config["l_dim"],
                warmup_steps=self.config["warmup"] * self.num_batches,
            )  # 15,150 // 3
            lr_scheduler_g = Scheduler(
                opt_g,
                dim_embed=self.config["l_dim"],
                warmup_steps=self.config["warmup"]
                * self.num_batches
                // max(1, (self.freq_d - 1)),
            )  #  // 3
        else:
            lr_scheduler_d = None
            lr_scheduler_g = None
        if self.config["sched"] != None:
            return [opt_d, opt_g], [lr_scheduler_d, lr_scheduler_g]
        else:
            return [opt_d, opt_g]

    def sample_n(self, mask):
        # Samples a mask where the zero padded particles are True, rest False
        mask_test = torch.ones_like(mask)
        n, counts = np.unique(self.data_module.n, return_counts=True)
        counts_prob = torch.tensor(counts / len(self.data_module.n))
        n_test = n[
            torch.multinomial(counts_prob, replacement=True, num_samples=(len(mask)))
        ]
        indices = torch.arange(30, device=mask.device)
        mask_test = indices.view(1, -1) < torch.tensor(n_test).view(-1, 1)
        mask_test = ~mask_test.bool()
        return mask_test

    def sampleandscale(self, batch, mask):
        """This is a helper function that samples from the flow (i.e. generates a new sample)
        and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
        on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
        because calculating the mass is a non linear transformation and does not commute with the mass calculation"""
        assert mask.dtype == torch.bool
        # empty=torch.zeros_like(batch).reshape(len(batch)*self.n_part,self.n_dim)
        # flat_batch=batch.reshape(len(batch)*self.n_part,self.n_dim)
        # zero_indices=(flat_batch!=0).all(axis=1)

        # c=c[zero_indices]
        # batch_temp=batch.unsqueeze(-1).repeat(1,1,1,30)
        fake, mask = self.sample(mask)
        fake_scaled = fake.clone()
        true = batch.clone()
        self.data_module.scaler = self.data_module.scaler.to(batch.device)
        fake_scaled = self.data_module.scaler.inverse_transform(fake_scaled)
        true = self.data_module.scaler.inverse_transform(true)
        fake_scaled[mask] = 0
        return fake, fake_scaled, true

    def sample(self, mask):
        # empty=torch.zeros((len(c),self.n_dim),device=c.device)
        # rest=torch.ones((len(batch),self.n_part),device=mask.device).bool()
        # rest[:,:self.n_current]=torch.zeros((len(batch),self.n_current),device=mask.device).bool()
        # mask=mask & rest
        # self.flow.eval()
        # for param in self.flow.parameters():
        #     param.requires_grad = False
        # worked=False
        # counter=0
        # while not worked:
        #     try:
        #         fake = self.flow.sample(1,c[indices])
        #         worked=True
        #     except:
        #         counter+=1
        #         if counter>10:
        #             print("10 errors while sampling, abort")
        #             raise

        # empty[indices,:]=fake.reshape(-1,self.n_dim
        # )

        if self.nf:
            with torch.no_grad():
                z = self.flow.flow.sample(len(mask) * self.n_current).reshape(
                    -1, self.n_current, self.n_dim
                )
            fake = z + self.gen_net(z, mask)  # )
        else:
            noise = trunc_normal_(
                torch.normal(
                    0, 1, (len(mask), self.n_current, self.n_dim), device=mask.device
                ),
                std=1,
            )
            fake = self.gen_net(noise, mask)  # ,mask)
            # fake[:,:self.n_current,:]=empty.reshape(-1,self.n_current,self.n_dim)
        return fake, mask

    def critic(self, batch, fake, mask, gen=False):
        if gen:
            pred = self.dis_net(
                fake.reshape(len(batch), self.n_part, self.n_dim), mask=mask
            )
            target = torch.ones_like(pred)
        else:
            pred_real = self.dis_net(batch, mask=mask)
            pred_fake = self.dis_net(fake.detach(), mask=mask)
            target_real = torch.ones_like(pred_real)
            target_fake = torch.zeros_like(pred_fake)
            pred = torch.vstack((pred_real, pred_fake))
            target = torch.vstack((target_real, target_fake))
        d_loss = nn.MSELoss()(pred, target).mean()

        if gen:
            if self.wgan_gen:
                return -pred.mean()
            else:
                return d_loss
        else:
            return (
                d_loss,
                self.sig(pred_real.detach()).cpu().numpy(),
                self.sig(pred_fake.detach()).cpu().numpy(),
            )

    def training_step(self, batch, batch_idx):
        mask = batch[:, 90:].bool()
        batch = batch[:, :90].reshape(len(batch), self.n_part, self.n_dim)
        # batch[mask]=0
        opt_d, opt_g = self.optimizers()
        if self.config["sched"]:
            sched_d, sched_g = self.lr_schedulers()
        if self.config["sched"] != None:
            self.log(
                "lr_g",
                sched_g.get_last_lr()[-1],
                logger=True,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "lr_d",
                sched_d.get_last_lr()[-1],
                logger=True,
                on_epoch=True,
                on_step=False,
            )
        ### GAN PART
        if self.config["sched"] != None:
            sched_d.step()

        fake, mask = self.sample(mask)
        d_loss, pred_t, pred_f = self.critic(batch, fake.detach(), mask, gen=False)

        # if self.current_epoch % 5 == 1:
        #         self.plot.plot_scores(pred_t,pred_f,train=True,step=self.global_step)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log(
            "d_loss", d_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        # if d_loss<0.15 and self.reset_counter>1000 and not self.train_g:
        #     print("start training with {} particles".format(self.n_current))
        #     self.train_g=True
        # else:
        #     self.reset_counter+=1
        # # if self.train_g:
        # if self.global_step % self.freq_d < 2:
        if self.global_step % self.freq_d == 0:
            fake, mask = self.sample(mask)
            if self.config["sched"] != None:
                sched_g.step()
            g_loss = self.critic(batch, fake, mask, gen=True)

            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            self.log(
                "g_loss",
                g_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
            if self.d_loss < 0.25:
                self.freq_d = np.max(1, self.freq_d - 1, 1)
            if self.g_loss < 0.25:
                self.freq_q = np.max(1, self.freq_d + 1, 1)
        if self.global_step % 500 < 2:
            _, pred_real, pred_fake = self.critic(batch, fake, mask, gen=False)
            self.plot.plot_scores(
                pred_real.reshape(-1),
                pred_fake.reshape(-1),
                train=False,
                step=self.current_epoch,
            )
        # self.losses_g[self.g_counter%5]=g_loss.cpu().detach().numpy()
        # self.g_counter+=1

        # if np.mean(self.losses_g)<0.3 and self.n_current<self.n_part :

        #     self.plot = plotting(model=self,gen=fake[:,:self.n_current,:self.n_dim].detach().cpu(),true=batch[:,:self.n_current,:self.n_dim].detach().cpu(),config=self.config,step=self.global_step,n=self.n_current,
        #         logger=self.logger.experiment,p=self.config["parton"])
        #     try:

        #         self.plot.plot_mass(save=None, bins=10)
        #         self.plot.plot_scores(pred_t,pred_f,True,self.global_step)
        #     except:
        #         print("error while plotting pre-increase")
        #         traceback.print_exc()
        #     self.n_current+=1
        #     print("increased number particles to ",self.n_current)
        #     self.reset_counter=0
        #     self.train_g=False

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        mask = batch[:, 90:].cpu().bool()
        # rest=torch.ones((len(batch),self.n_part),device=mask.device).bool()
        # rest[:,:self.n_current]=torch.zeros((len(batch),self.n_current),device=mask.device).bool()
        # mask=mask | rest
        batch = batch[:, :90].cpu().reshape(len(mask), self.n_current, self.n_dim)

        # empty = torch.zeros_like(batch)
        self.gen_net = self.gen_net.to("cpu")
        self.dis_net = self.dis_net.to("cpu")
        if self.nf:
            self.flow.eval()
            self.flow = self.flow.to("cpu")
        batch = batch.to("cpu")
        fake, fake_scaled, true_scaled = self.sampleandscale(batch, mask)
        m_t = mass(true_scaled[:, : self.n_current, : self.n_dim])
        m_c = mass(fake_scaled[:, : self.n_current, : self.n_dim])

        for i in range(self.n_part):
            fake_scaled[fake_scaled[:, i, 2] < 0, i, 2] = 0
            true_scaled[true_scaled[:, i, 2] < 0, i, 2] = 0
        # Some metrics we track
        try:
            cov, mmd = cov_mmd(
                true_scaled[:, : self.n_current],
                fake_scaled[:, : self.n_current],
                use_tqdm=False,
            )
        except:
            print("error cov,mmd")
            cov = -1
            mmd = -1
        try:

            fpndv = fpnd(
                fake_scaled[:50000, :].numpy(),
                use_tqdm=False,
                jet_type=self.config["parton"],
            )
        except:
            fpndv = 1000

        w1m_ = w1m(fake_scaled[:, : self.n_current], true_scaled[:, : self.n_current])[
            0
        ]
        w1p_ = w1p(fake_scaled[:, : self.n_current], true_scaled[:, : self.n_current])[
            0
        ]
        w1efp_ = w1efp(
            fake_scaled[:, : self.n_current], true_scaled[:, : self.n_current]
        )[0]

        self.w1ms.append(w1m_)
        self.fpnds.append(fpndv)
        temp = {
            "val_fpnd": fpndv,
            "val_mmd": mmd,
            "val_cov": cov,
            "val_w1m": w1m_,
            "val_w1efp": w1efp_,
            "val_w1p": w1p_,
            "step": self.global_step,
        }
        print("epoch {}: ".format(self.current_epoch), temp)
        if self.global_step > 3:
            try:
                self._results(temp)
            except:
                print("error in results")
            if (fpndv < self.fpnds).all():
                summary = self._summary(temp)

        self.log(
            "hp_metric", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_w1m", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_w1p", w1p_, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_w1efp",
            w1efp_,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_cov", cov, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log(
            "val_fpnd", fpndv, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log(
            "val_mmd", mmd, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.plot = plotting_point_cloud(
            model=self,
            gen=fake_scaled[:, : self.n_current, : self.n_dim],
            true=true_scaled[:, : self.n_current, : self.n_dim],
            config=self.config,
            step=self.global_step,
            n=self.n_current,
            logger=self.logger.experiment,
            p=self.config["parton"],
        )
        self.plot.plot_mom(self.global_step)
        try:

            self.plot.plot_mass(save=None, bins=50)
            _, pred_real, pred_fake = self.critic(batch, fake, mask)
            self.plot.plot_scores(
                pred_real.reshape(-1),
                pred_fake.reshape(-1),
                train=False,
                step=self.current_epoch,
            )
            # self.plot.plot_2d(save=True)
        #     self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,
        #                          m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc()
        if self.nf:
            self.flow = self.flow.to("cuda")
        self.dis_net = self.dis_net.to("cuda")
        self.gen_net = self.gen_net.to("cuda")
        print("Version:", self.logger.version)
