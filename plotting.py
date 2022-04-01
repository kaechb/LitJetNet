import matplotlib.pyplot as plt
import os
import hist
import mplhep as hep
import torch
import numpy as np
import hist
from hist import Hist
import traceback
class plotting():
    def __init__(self,true,gen,config,step,model=None,logger=None,):
        self.config=model.config
        self.n_dim=self.config["n_dim"]
        self.gen=gen
        self.test_set=true
        self.step=step
        self.model=model
        if logger is not None:
            self.summary=logger
       
    def plot_marginals(self,save=False):
        plt.switch_backend('agg')
        
#         if save:
#             pass
#             self.plot_path_vars=self.plot_path+"/vars"
#             os.makedirs(self.plot_path_vars,exist_ok=True)
        name,label=["eta","phi","pt"],[r"$\eta^{rel}$",r"$\phi$",r"$p_T^{rel}$"]

        for i in range(self.n_dim):
            a=np.quantile(self.test_set[:,i].numpy(),0)
            b=np.quantile(self.test_set[:,i].numpy(),1)

            h=hist.Hist(hist.axis.Regular(15,a,b))
            h2=hist.Hist(hist.axis.Regular(15,a,b))
            h.fill(self.gen[:,i].numpy())
            h2.fill(self.test_set[:,i].numpy())
            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
            plt.tight_layout()
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
       
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                rp_num_label="Generated",
                rp_denom_label="Data",
                rp_uncert_draw_type="line",  # line or bar
            )

            ax[0].set_xlabel("")
            ax[1].set_ylim(0.25,2)
            ax[0].set_xlim(a,b)
            ax[1].set_xlim(a,b)
            plt.xlabel(label[i%3])
            plt.tight_layout(pad=2)
            if save:
                self.summary.add_figure("jet{}_{}".format(i//3+1,name[i%3]),fig,global_step=self.step)
                self.summary.close()
#                 plt.savefig("{}/jet{}_{}.png".format(self.plot_path_vars,int(i/3)+1,name[i%3]))
#                 ax[0].set_yscale('log')
#                 plt.savefig("{}/log_jet{}_{}.png".format(self.plot_path_vars,int(i/3)+1,name[i%3]))
#                 plt.close()
            else:
                plt.show()
    def plot_2d(self,save=False):
        data=self.test_set[:,:self.n_dim].reshape(-1,3).numpy()
        gen=self.gen[:,:self.n_dim].reshape(-1,3).numpy()
        labels=[r"$\eta^{rel}$",r"$\phi$",r"$p_T^{rel}$"]
        names=["eta","phi","pt","m"]
        for index in [[0,1],[0,2],[1,2]]:
            
            fig,ax=plt.subplots(ncols=2,figsize=(16, 8))
            _,x,y,_=ax[0].hist2d(data[:,index[0]],data[:,index[1]],bins=30)
            
            if index[1]==2:
                
                y = np.logspace(np.log(y[0]),np.log(y[-1]),len(y))
                ax[0].hist2d(data[:,index[0]],gen[:,index[1]],bins=[x,y])
            ax[1].hist2d(gen[:,index[0]],gen[:,index[1]],bins=[x,y])
            plt.tight_layout(pad=2)
            ax[0].set_xlabel( labels[index[0]])
            ax[0].set_ylabel( labels[index[1]])
            
            ax[0].set_title("Data")
            ax[1].set_xlabel( labels[index[0]])
            ax[1].set_ylabel( labels[index[1]])
            
            ax[1].set_title("Gen")
           
            if save:
                self.summary.add_figure("2d{}-{}".format(names[index[0]],names[index[1]]),fig,global_step=self.step)
                
                self.summary.close()
            else:
                plt.show()
 
        
    def plot_mass(self,save=False,quantile=False):
        i=0

        m=self.gen[:,self.n_dim].numpy()
        m_t=self.test_set[:,-1].numpy()
        gen=self.gen[:,:self.n_dim].reshape(-1,3).numpy()
        for v,name in zip(["eta","phi","pt","m"],[r"$\eta^{rel}$",r"$\phi$",r"$p_T^{rel}$",r"$m_T^{rel}$"]):
            
            if v!="m":
                a=min(np.quantile(self.gen[:,i],0.001),np.quantile(self.test_set[:,i],0.001))
                b=max(np.quantile(self.gen[:,i],0.999),np.quantile(self.test_set[:,i],0.999))     
                h=hist.Hist(hist.axis.Regular(15,a,b))
                h2=hist.Hist(hist.axis.Regular(15,a,b))
                h.fill(self.gen[:,i])
                h2.fill(self.test_set[:,i])
                i+=1
            else:
                a=np.quantile(m_t,0.001)
                b=np.quantile(m_t,0.999)
                h=hist.Hist(hist.axis.Regular(15,a,b))
                h2=hist.Hist(hist.axis.Regular(15,a,b))
                bins = h.axes[0].edges
                h.fill(m)
                h2.fill(m_t)
           

            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(8,8))
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )

            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                rp_num_label="Generated",
                rp_denom_label="MC Simulation",
                rp_uncert_draw_type="line",  # line or bar
            )
           
           
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            if quantile and v=="m":
                ax[0].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile gen",hatch="/")
                ax[0].vlines(np.quantile(m,0.1),0,np.max(h[:]),color="red",label='10% quantile train')
            ax[0].set_xlabel("")
            ax[1].set_ylim(0.25,2)
            ax[0].set_xlim(a,b)
            ax[1].set_xlim(a,b)
            plt.xlabel(name)
            plt.tight_layout(pad=2)
            ax[0].legend(["Generated","Training","10% quantile Gen","10% quantile Train"] )
            if save:
                if v!="m":
                     self.summary.add_figure("inclusive"+v,fig,self.step)
                else:
                    self.summary.add_figure("jet_mass",fig,self.step)
    #             print("added figure")
    #             self.summary.close()
            else:
                plt.show()
                plt.xlabel(name)

    def losses(self,save=False):
        fig=plt.figure()
        hep.cms.label("Private Work",data=None,lumi=None,year=None)
        plt.xlabel('step')
        plt.ylabel('loss')
        ln1=plt.plot(self.model.logprobs,label='log$(p_{gauss}(x_{data}))$')
        if self.config["n_mse"]<np.inf:
            plt.twinx()
            ln2=plt.plot(self.model.mlosses,label=r'mass mse $\times$ {}'.format(self.config["lambda"]),color='orange')
            plt.ylabel("MSE")
            plt.yscale("log")
            ln1+=ln2
        labs=[l.get_label() for l in ln1]
        plt.legend(ln1,labs)
        plt.tight_layout(pad=2)
        if save:
            self.summary.add_figure("losses",fig,self.step)
#             self.summary.close()
        else:
            plt.show()