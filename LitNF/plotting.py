
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
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

# from torch.nn import MultiheadAttention,TransformerEncoder,TransformerEncoderLayer
def mass(p, canonical=False):
    if len(p.shape)==2:
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


class plotting():
    '''This is a class that takes care of  plotting steps in the script,
        It is initialized with the following arguments:
        true=the simulated data, note that it needs to be scaled
        gen= Generated data , needs to be scaled
        step=The current step of the training, this is need for tensorboard
        model=the model that is trained, a bit of an overkill as it is only used to access the losses
        config=the config used for training
        logger=The logger used for tensorboard logging'''
    def __init__(self,true,gen,config,p,step=None,model=None,logger=None,weight=1,nf=None):
        self.config=model.config
        self.n_dim=self.config["n_dim"]
        self.gen=gen
        self.test_set=true
        self.step=step
        self.model=model
        self.p=p
        self.n_part=config["n_part"]
        self.nf=nf
        self.n_dim=config["n_dim"]
        self.weight=weight
        if logger is not None:
            self.summary=logger
    def plot_mass_only(self,m,m_t,bins=15):
        fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(6,8))
        a=min(np.quantile(m_t,0.001),np.quantile(m,0.001))
        b=max(np.quantile(m_t,0.999),np.quantile(m,0.999))
        a=np.quantile(m_t,0.001)
        b=np.quantile(m_t,0.999)
        h=hist.Hist(hist.axis.Regular(bins,a,b))
        h2=hist.Hist(hist.axis.Regular(bins,a,b))
        bins = h.axes[0].edges
        h.fill(m)#,weight=1/self.weight)
        h2.fill(m_t)
            
            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )

        main_ax_artists, sublot_ax_arists = h.plot_ratio(
            h2,
            ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
            rp_ylabel=r"Ratio",
            rp_num_label="Generated",
            rp_denom_label="Ground Truth",
            rp_uncert_draw_type="line",  # line or bar
        )
        ax[0].set_xlabel("")
#                 if quantile and v=="m" and plot_vline:
#                     ax[0,k].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile gen",hatch="/")
#                     ax[0,k].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile train')

        ax[1].set_ylim(0.25,2)
        ax[0].set_xlim(a,b)
        ax[1].set_xlabel("$m_T$",fontsize=22)
        ax[1].set_xlim(a,b)
        ax[0].set_ylabel("Counts" ,fontsize=18)
        ax[1].set_ylabel("Ratio",fontsize=18)
        plt.close()
        # plt.savefig("{}_mass".format(self.p))
        #plt.show()
        
    def plot_marginals(self,ith=None,title=None,save=None):
        #This plots the marginal distribution for simulation and generation
        #Note that this is the data the model sees during training as input to model in the NF
        #This is the distribution of one of [eta,phi,pt] of one particle of the n particles per jet: for example the pt of the 3rd particle
        #if save, the histograms are logged to tensorboard otherwise they are shown
        
        # plt.switch_backend('agg')
        i=str(ith)

        name,label=["eta","phi","pt"],['${{\eta}}^{{\\tt rel}}_{{{}}}$'.format(ith+1),"${{\phi}}^{{\\tt rel}}_{{{}}}$".format(ith+1),"${{p^{{\\tt rel}}_{{T,{}}}}}$".format(ith+1)]
        fig,ax=plt.subplots(2,3,gridspec_kw={'height_ratios': [3, 1]},figsize=(18,6))
        particles=[3*ith,3*ith+1,3*ith+2]
        pre=""
        if ith!=0:
            pre=str(ith+1)+"."
        plt.suptitle(pre+" Hardest Particle",fontweight="bold",fontsize=18)
        k=0
        for i in particles:


            ax_temp=ax[:,k]
           
            a=np.quantile(self.test_set[:,i].numpy(),0)
            b=np.quantile(self.test_set[:,i].numpy(),1)

            h=hist.Hist(hist.axis.Regular(15,a,b,label=label[i%3],underflow=False,overflow=False))
            h2=hist.Hist(hist.axis.Regular(15,a,b,label=label[i%3],underflow=False,overflow=False))
            h.fill(self.gen[:,i].numpy())
            h2.fill(self.test_set[:,i].numpy())
            
            plt.tight_layout()
            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0,k] )
       
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax_temp[0],"ratio_ax":ax_temp[1]},
                rp_ylabel=r"Ratio",
#                 rp_xlabel=label[i%3],
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            
            
            ax_temp[0].set_xlabel("")
            ax_temp[1].set_ylim(0.25,2)
            ax_temp[0].set_xlim(a,b)
            ax_temp[1].set_xlim(a,b)
            ax_temp[1].set_xlabel(label[i%3],fontsize=22)
            ax_temp[0].set_ylabel("Counts" ,fontsize=18)
            ax_temp[1].set_ylabel("Ratio",fontsize=18)
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].patches[1].set_fc("orange")
            ax[0,k].patches[1].set_alpha(0.3) 
            ax[0,k].get_legend().remove()
            #plt.tight_layout(pad=2)
            k+=1
        ax[0,-1].legend(loc="best",fontsize=18)  
        plt.close()
        # if not save==None:
        #     plt.savefig(save+str(ith)+".pdf",format="pdf")
        #plt.show()


   
        
    def oversample(self,m,m_t,weight,save=None,quantile=False,bins=15,plot_vline=False,title="",leg=-2):
        #This creates a histogram of the inclusive distributions and calculates the mass of each jet
        #and creates a histogram of that
        #if save, the histograms are logged to tensorboard otherwise they are shown
        #if quantile, this also creates a histogram of a subsample of the generated data, 
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i=0
        k=0
        fig,ax=plt.subplots(2,4,gridspec_kw={'height_ratios': [3, 1]},figsize=(20,5))
        plt.suptitle(save)
        for v,name in zip(["eta","phi","pt","m"],[r"$\eta^{rel}$",r"$\phi^{rel}$",r"$p_T^{rel}$",r"$m^{rel}$"]):
            
            if v!="m":
                a=min(np.quantile(self.gen[:,i],0.001),np.quantile(self.test_set[:,i],0.001))
                b=max(np.quantile(self.gen[:,i],0.999),np.quantile(self.test_set[:,i],0.999))     
                
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                h.fill(self.gen[:,i],weight=1/weight)
                
                h2.fill(self.test_set[:,i])
                i+=1
            else:
                a=min(np.quantile(m_t,0.001),np.quantile(m,0.001))
                b=max(np.quantile(m_t,0.999),np.quantile(m,0.999))
                a=np.quantile(m_t,0.001)
                b=np.quantile(m_t,0.999)
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                
                h.fill(m,weight=1/weight)#,weight=1/self.weight)
                h2.fill(m_t)
            
            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
        
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0,k],"ratio_ax":ax[1,k]},
                rp_ylabel=r"Ratio",
                rp_num_label=r"Generated$\times{}$".format(1./weight),
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0,k].set_xlabel("")
#                 if quantile and v=="m" and plot_vline:
#                     ax[0,k].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile gen",hatch="/")
#                     ax[0,k].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile train')


            ax[1,k].set_ylim(0.25,2)
            ax[0,k].set_xlim(a,b)
            ax[1,k].set_xlabel(name,fontsize=18)
            ax[1,k].set_xlim(a,b)
            ax[0,k].set_ylabel("Counts" ,fontsize=18)
            ax[1,k].set_ylabel("Ratio",fontsize=18)
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].patches[1].set_fc("orange")
            ax[0,k].patches[1].set_alpha(0.3) 
            ax[0,k].get_legend().remove()
            plt.tight_layout(pad=1)
            k+=1
        ax[0,leg].legend(loc="best",fontsize=15) 
        # if not save==None:   
        #         plt.savefig(save+".pdf",format="pdf")

            
    def plot_mass(self,m,m_t,save=None,quantile=False,bins=15,plot_vline=False,title="",leg=-1):
        #This creates a histogram of the inclusive distributions and calculates the mass of each jet
        #and creates a histogram of that
        #if save, the histograms are logged to tensorboard otherwise they are shown
        #if quantile, this also creates a histogram of a subsample of the generated data, 
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i=0
        k=0
        fig,ax=plt.subplots(2,4,gridspec_kw={'height_ratios': [3, 1]},figsize=(24,6))
        plt.suptitle("All Particles",fontweight="bold",fontsize=18)
        for v,name in zip(["eta","phi","pt","m"],[r"$\eta^{\tt rel}$",r"$\phi^{\tt rel}$",r"$p_T^{\tt rel}$",r"$m^{\tt rel}$"]):
            
            if v!="m":
                a=min(np.quantile(self.gen[:,i],0.001),np.quantile(self.test_set[:,i],0.001))
                b=max(np.quantile(self.gen[:,i],0.999),np.quantile(self.test_set[:,i],0.999))     
                temp=self.test_set[:,i].numpy()
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                h.fill(self.gen[:,i])
                h2.fill(self.test_set[:,i])
                i+=1
            else:
                a=min(np.quantile(m_t,0.001),np.quantile(m,0.001))
                b=max(np.quantile(m_t,0.999),np.quantile(m,0.999))
                a=np.quantile(m_t,0.001)
                b=np.quantile(m_t,0.999)
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                #bins = h.axes[0].edges
                h.fill(m)#,weight=1/self.weight)
                h2.fill(m_t)
                temp=m_t
            
            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
        
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0,k],"ratio_ax":ax[1,k]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0,k].set_xlabel("")
            

            # ax[0,k].patches[1].set_fc("orange")
            # ax[0,k].patches[1].set_alpha(0.5)
#                 if quantile and v=="m" and plot_vline:
#                     ax[0,k].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile gen",hatch="/")
#                     ax[0,k].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile train')

            #ax[0,k].hist(temp,bins=bins,color="orange",alpha=0.5)  
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].patches[1].set_fc("orange")
            ax[0,k].patches[1].set_alpha(0.3) 
            ax[1,k].set_ylim(0.25,2)
            ax[0,k].set_xlim(a,b)
            ax[1,k].set_xlabel(name,fontsize=22)
            ax[1,k].set_xlim(a,b)
            ax[0,k].set_ylabel("Counts", fontsize=18)
            ax[1,k].set_ylabel("Ratio",fontsize=18)
            ax[0,k].get_legend().remove()
            k+=1
#                 if plot_vline:
#                        ax[0,k].legend(["Generated","Training","10% quantile Gen","10% quantile Sim"] )
#                 else:
#                       ax[0,k].legend(["Flow Generated","MC Simulated"] )
            
            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            
#             plt.xlabel(name)
        
        ax[0,leg].legend(loc="best",fontsize=18)  
        plt.tight_layout(pad=1)
        # if not save==None:
        #     plt.savefig(save+".pdf",format="pdf")
        self.summary.add_figure("inclusive", fig, self.step)
        plt.close()

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
        #if save:
        title = ["corr_eta", "corr_phi", "corr_pt"]
        self.summary.add_figure(title[i], fig, self.step)
        plt.close()
        #             self.summary.close()
        # else:
        #     plt.show()
    def plot_scores(self,pred_real,pred_fake,train,step):
        fig, ax = plt.subplots()
        bins=np.linspace(0,1,100)
        ax.hist(pred_fake.detach().cpu().numpy(), label="Generated", bins=bins, histtype="step")
        ax.hist(pred_real.detach().cpu().numpy(), label="Ground Truth", bins=bins, histtype="stepfilled",alpha=0.3)
        ax.legend()
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        self.summary.add_figure("class_train" if train else "class_val", fig, global_step=step)
        plt.close()
    def plot_mom(self,step):
        fig, ax = plt.subplots()
        bins=np.linspace(0.7,1.4,30)
        ax.hist(self.gen.reshape(len(self.gen),self.n_part,3)[:,:,2].sum(1).detach().cpu().numpy(), label="Generated", bins=bins, histtype="step",alpha=1)
        
        ax.hist(self.test_set.reshape(len(self.test_set),self.n_part,3)[:,:,2].sum(1).detach().cpu().numpy(), label="Ground Truth", bins=bins, histtype="stepfilled",alpha=0.3)
        ax.hist(self.nf.reshape(len(self.nf),self.n_part,3)[:,:,2].sum(1).detach().cpu().numpy(), label="NF", bins=bins, histtype="step",alpha=1,linestyle="dashed")
        ax.legend()
        plt.ylabel("Counts")
        plt.xlabel("$\sum p_T^{rel}$")
        self.summary.add_figure("momentum_sum", fig, global_step=step)
        plt.close()