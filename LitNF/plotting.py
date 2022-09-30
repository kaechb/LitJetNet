print("thx max")
import matplotlib.pyplot as plt
import os
import hist
import mplhep as hep
import torch
import numpy as np
import hist
from hist import Hist
import traceback
import pandas as pd
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helpers import *
import os
from scipy import stats
import datetime
import pandas as pd
import traceback
import time
from torch import nn

gen_step = 0

# train mode
import nflows as nf
from nflows.utils.torchutils import create_random_binary_mask
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.nn import nets
from nflows.flows.base import Flow
from nflows.flows import base
from nflows.transforms.coupling import *
from nflows.transforms.autoregressive import *

from pytorch_lightning.callbacks import ModelCheckpoint

# from comet_ml import Experiment

import pytorch_lightning as pl
import os

# from plotting import plotting
from torch.nn import functional as FF
import traceback
import os

import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, ExponentialLR
import torch
from torch import nn
from torch.nn import functional as FF
import numpy as np
from jetnet.evaluation import w1p, w1efp, w1m, cov_mmd, fpnd
import mplhep as hep
import hist
from hist import Hist
from pytorch_lightning.loggers import TensorBoardLogger
from collections import OrderedDict

from helpers import *

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import time

print("good boy")
# from torch.nn import MultiheadAttention,TransformerEncoder,TransformerEncoderLayer
def mass(p, canonical=False):
    if len(p.shape)!=3:
        n_dim = p.shape[1]
        p = p.reshape(-1, n_dim // 3, 3)
    if canonical:
       
        px = p[:, :, 0]
        py = p[:, :, 1]
        pz = p[:, :, 2]

    else:

        px = torch.cos(p[:, :, 1]) * p[:, :, 2]
        py = torch.sin(p[:, :, 1]) * p[:, :, 2]
        pz = torch.sinh(p[:, :, 0]) * p[:, :, 2]
    px = torch.clamp(px, min=-100, max=100)
    py = torch.clamp(py, min=-100, max=100)
    pz = torch.clamp(pz, min=-100, max=100)
    E = torch.sqrt(px**2 + py**2 + pz**2)
    E = E.sum(axis=1) ** 2
    p = px.sum(axis=1) ** 2 + py.sum(axis=1) ** 2 + pz.sum(axis=1) ** 2
    m2 = E - p
    # if m2.isnan().any():
    #     print("px:{} py:{} pz:{} ".format(px.abs().max(),py.abs().max(),pz.abs().max()))
    # assert m2.isnan().sum()==0
    return torch.sqrt(torch.max(m2, torch.zeros(len(E)).to(E.device)))


font = {"family": "normal", "weight": "bold", "size": 12}

matplotlib.rc("font", **font)


class plotting:
    """This is a class that takes care of  plotting steps in the script,
    It is initialized with the following arguments:
    true=the simulated data, note that it needs to be scaled
    gen= Generated data , needs to be scaled
    step=The current step of the training, this is need for tensorboard
    model=the model that is trained, a bit of an overkill as it is only used to access the losses
    config=the config used for training
    logger=The logger used for tensorboard logging"""

    def __init__(
        self,
        true,
        gen,
        gen_corr,
        config,
        step,
        model=None,
        logger=None,
    ):
        self.config = model.config
        self.n_dim = self.config["n_dim"]
        self.gen = gen.numpy().reshape(len(gen),90)
        self.gen_corr = gen_corr.numpy().reshape(len(gen_corr),90)
        self.test_set = true.numpy().reshape(len(true),90)
        self.step = step
        self.model = model
        if logger is not None:
            self.summary = logger

    def plot_2d(self, save=False):
        # This creates a 2D histogram of the inclusive distribution for all 3 feature combinations
        # Inclusive means that is the distribution of pt of all particles per jet and sample
        # if save, the histograms are logged to tensorboard otherwise they are shown
        data = self.test_set[:, : self.n_dim].reshape(-1, 3)
        gen = self.gen[:, : self.n_dim].reshape(-1, 3)
        labels = [r"$\eta^{rel}$", r"$\phi^{rel}$", r"$p_T^{rel}$"]
        names = ["eta", "p3hi", "pt"]
        for index in [[0, 1], [0, 2], [1, 2]]:

            fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
            _, x, y, _ = ax[0].hist2d(data[:, index[0]], data[:, index[1]], bins=30)

            if index[1] == 2:

                y = np.logspace(np.log(y[0]), np.log(y[-1]), len(y))
                ax[0].hist2d(data[:, index[0]], gen[:, index[1]], bins=[x, y])
            ax[1].hist2d(gen[:, index[0]], gen[:, index[1]], bins=[x, y])
            plt.tight_layout(pad=2)
            ax[0].set_xlabel(labels[index[0]])
            ax[0].set_ylabel(labels[index[1]])

            ax[0].set_title("Data")
            ax[1].set_xlabel(labels[index[0]])
            ax[1].set_ylabel(labels[index[1]])

            ax[1].set_title("Gen")

            if save:
                self.summary.add_figure(
                    "2d{}-{}".format(names[index[0]], names[index[1]]),
                    fig,
                    global_step=self.step,
                )

                # self.summary.close()
            else:
                plt.show()

    def plot_mass(
        self, m, m_t, m_c, save=False, quantile=False, bins=15, plot_vline=True
    ):
        # This creates a histogram of the inclusive distributions and calculates the mass of each jet
        # and creates a histogram of that
        # if save, the histograms are logged to tensorboard otherwise they are shown
        # if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i = 0

        for v, name in zip(
            ["eta", "phi", "pt", "m"],
            [r"$\eta^{rel}$", r"$\phi^{rel}$", r"$p_T^{rel}$", r"$m_T^{rel}$"],
        ):

            if v != "m":
                a = np.quantile(self.test_set[:, i], 0.001)
                b = np.quantile(self.test_set[:, i], 0.999)
                h = hist.Hist(hist.axis.Regular(bins, a, b))
                h2 = hist.Hist(hist.axis.Regular(bins, a, b))
                h3 = hist.Hist(hist.axis.Regular(bins, a, b))

                h.fill(self.test_set[:, i])
                h2.fill(self.gen[:, i])
                h3.fill(self.gen_corr[:, i])
                i += 1
            else:
                a = np.quantile(m_t, 0.001)
                b = np.quantile(m_t, 0.999)
                h = hist.Hist(hist.axis.Regular(bins, a, b))
                h2 = hist.Hist(hist.axis.Regular(bins, a, b))
                h3 = hist.Hist(hist.axis.Regular(bins, a, b))
                bins = h.axes[0].edges
                h.fill(m_t)
                h2.fill(m)
                h3.fill(m_c)
            fig, ax = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 8)
            )

            #             hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            try:
                main_ax_artists, sublot_ax_arists = h.plot_ratio(
                    h3,
                    ax_dict={"main_ax": ax[0], "ratio_ax": ax[1]},
                    rp_ylabel=r"Ratio",
                    rp_num_label="MC Simulated",
                    rp_denom_label="Flow Generated+Corrected",
                    rp_uncert_draw_type="line",  # line or bar
                )

                h2.plot(ax=ax[0], label="Flow Generated")
                ax[0].set_xlabel("")
                #                 if quantile and v=="m" and plot_vline:
                #                     ax[0].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile Flow Generated",hatch="/")
                #                     ax[0].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile MC Simulated')

                ax[1].set_ylim(0.25, 2)
                ax[0].set_xlim(a, b)
                ax[0].legend()
                ax[1].set_xlim(a, b)
            #                 if v!="m":
            #                     ax[0].legend(["Flow Generated","MC Simulated"])
            #                 elif plot_vline:
            #                     ax[0].legend(["Flow Generated","MC Simulated"] )
            except:
                print("mass plot failed reverting to simple plot mass bins")
                traceback.print_exc()
                plt.close()
                plt.figure()
                _, b, _ = plt.hist(m_t, 15, label="Sim", alpha=0.5)
                plt.hist(m, b, label="Gen", alpha=0.5)
                plt.legend()
            # hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            ax[1].set_xlabel(name, fontsize=28)
            ax[0].set_ylabel("Counts", fontsize=28)
            ax[1].set_ylabel("Ratio", fontsize=28)

            plt.tight_layout(pad=2)
            if save:
                if v != "m":
                    self.summary.add_figure("inclusive" + v, fig, self.step)
                else:
                    self.summary.add_figure("jet_mass", fig, self.step)
            #             print("added figure")
            #             self.summary.close()
            else:
                plt.show()

    def plot_correlations(self, save=True):
        # Plots correlations between all particles for i=0 eta,i=1 phi,i=2 pt
        self.plot_corr(i=0, save=save)
        self.plot_corr(i=1, save=save)
        self.plot_corr(i=2, save=save)

    def plot_corr(
        self, i=0, names=["$\eta^{rel}$", "$\phi^{rel}$", "$p_T$"], save=True
    ):
        if i == 2:
            c = 1
        else:
            c = 0.25
        df_g = pd.DataFrame(self.gen[:, : self.n_dim][:, range(i, 90, 3)])
        df_h = pd.DataFrame(self.test_set[:, : self.n_dim][:, range(i, 90, 3)])

        fig, ax = plt.subplots(ncols=2, figsize=(15, 7.5))
        corr_g = ax[0].matshow(df_g.corr())
        corr_g.set_clim(-c, c)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(corr_g, cax=cax)

        corr_h = ax[1].matshow(df_h.corr())
        corr_h.set_clim(-c, c)
        divider = make_axes_locatable(ax[1])

        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(corr_h, cax=cax2)
        plt.suptitle("{} Correlation between Particles".format(names[i]), fontsize=38)
        ax[0].set_title("Flow Generated", fontsize=34)
        ax[1].set_title("MC Simulated", fontsize=28)
        ax[0].set_xlabel("Particles", fontsize=28)
        ax[0].set_ylabel("Particles", fontsize=28)
        ax[1].set_xlabel("Particles", fontsize=28)
        ax[1].set_ylabel("Particles", fontsize=28)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        if save:
            title = ["corr_eta", "corr_phi", "corr_pt"]
            self.summary.add_figure(title[i], fig, self.step)

        #             self.summary.close()
        else:
            plt.show()
    def plot_scores(self,pred_real,pred_fake,train,step):
        fig, ax = plt.subplots()
        _,bins,_=ax.hist(pred_real.detach().cpu().numpy(), label="MC Simulated", bins=np.linspace(-0.1,1.1,100), histtype="step")
        ax.hist(pred_fake.detach().cpu().numpy(), label="ML Generated", bins=bins, histtype="step")
        ax.legend()
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        self.summary.add_figure("class_train" if train else "class_val", fig, global_step=step)