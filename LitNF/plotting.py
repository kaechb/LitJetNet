import matplotlib.pyplot as plt
import os
import hist
import mplhep as hep
import torch
import numpy as np
import hist
from hist import Hist
import traceback
from helpers import mass
import pandas as pd
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
font = {'family' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
class plotting():
    '''This is a class that takes care of  plotting steps in the script,
        It is initialized with the following arguments:
        true=the simulated data, note that it needs to be scaled
        gen= Generated data , needs to be scaled
        step=The current step of the training, this is need for tensorboard
        model=the model that is trained, a bit of an overkill as it is only used to access the losses
        config=the config used for training
        logger=The logger used for tensorboard logging'''
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
        #This plots the marginal distribution for simulation and generation
        #Note that this is the data the model sees during training as input to model in the NF
        #This is the distribution of one of [eta,phi,pt] of one particle of the n particles per jet: for example the pt of the 3rd particle
        #if save, the histograms are logged to tensorboard otherwise they are shown
        plt.switch_backend('agg')
        name,label=["eta","phi","pt"],[r"$\eta^{rel}$",r"$\phi^{rel}$",r"$p_T^{rel}$"]

        for i in range(self.n_dim):
            a=np.quantile(self.test_set[:,i].numpy(),0)
            b=np.quantile(self.test_set[:,i].numpy(),1)

            h=hist.Hist(hist.axis.Regular(15,a,b))
            h2=hist.Hist(hist.axis.Regular(15,a,b))
            h.fill(self.gen[:,i].numpy())
            h2.fill(self.test_set[:,i].numpy())
            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(8,8))
            plt.tight_layout()
#             hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
       
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                rp_num_label="Flow Generated",
                rp_denom_label="MC Simulated",
                rp_uncert_draw_type="line"  # line or bar
            )

            ax[0].set_xlabel("")
            ax[1].set_xlabel(name,fontsize=28)
            ax[0].set_ylabel("Counts",fontsize=28)
            ax[1].set_ylabel("Ratio",fontsize=28)
            
            ax[1].set_ylim(0.25,2)
            ax[0].set_xlim(a,b)
            ax[1].set_xlim(a,b)
            plt.xlabel(label[i%3])
            plt.tight_layout(pad=2)
            if save:
                self.summary.add_figure("jet{}_{}".format(i//3+1,name[i%3]),fig,global_step=self.step)
                self.summary.close()
            else:
                plt.show()


    def plot_2d(self,save=False):
        #This creates a 2D histogram of the inclusive distribution for all 3 feature combinations
        #Inclusive means that is the distribution of pt of all particles per jet and sample
        #if save, the histograms are logged to tensorboard otherwise they are shown
        data=self.test_set[:,:self.n_dim].reshape(-1,3).numpy()
        gen=self.gen[:,:self.n_dim].reshape(-1,3).numpy()
        labels=[r"$\eta^{rel}$",r"$\phi^{rel}$",r"$p_T^{rel}$"]
        names=["eta","p3hi","pt"]
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
                
                # self.summary.close()
            else:
                plt.show()
 
        
    def plot_mass(self,m,m_t,save=False,quantile=False,bins=15,plot_vline=True):
        #This creates a histogram of the inclusive distributions and calculates the mass of each jet
        #and creates a histogram of that
        #if save, the histograms are logged to tensorboard otherwise they are shown
        #if quantile, this also creates a histogram of a subsample of the generated data, 
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i=0


        gen=self.gen[:,:self.n_dim].reshape(-1,3).numpy()
        for v,name in zip(["eta","phi","pt","m"],[r"$\eta^{rel}$",r"$\phi^{rel}$",r"$p_T^{rel}$",r"$m^{rel}$"]):
            
            if v!="m":
                a=np.quantile(self.test_set[:,i],0.001)
                b=np.quantile(self.test_set[:,i],0.999)
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                h.fill(self.gen[:,i])
                h2.fill(self.test_set[:,i])
                i+=1
            else:
                a=np.quantile(m_t,0.001)
                b=np.quantile(m_t,0.999)
                h=hist.Hist(hist.axis.Regular(bins,a,b))
                h2=hist.Hist(hist.axis.Regular(bins,a,b))
                bins = h.axes[0].edges
                h.fill(m)
                h2.fill(m_t)
            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(8,8))
#             hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            try:
                main_ax_artists, sublot_ax_arists = h.plot_ratio(
                    h2,
                    ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                    rp_ylabel=r"Ratio",
                    rp_num_label="Flow Generated",
                    rp_denom_label="MC Simulated",
                    rp_uncert_draw_type="line",  # line or bar
                )
                ax[0].set_xlabel("")
                if quantile and v=="m" and plot_vline:
                    ax[0].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile Flow Generated",hatch="/")
                    ax[0].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile MC Simulated')
                    
                ax[1].set_ylim(0.25,2)
                ax[0].set_xlim(a,b)
                
                ax[1].set_xlim(a,b)
                if v!="m":
                    ax[0].legend(["Flow Generated","MC Simulatied"])
                elif plot_vline:
                    ax[0].legend(["Flow Generated","MC Simulated"] )
            except:
                print("mass plot failed reverting to simple plot mass bins")
                plt.close()
                plt.figure()
                _,b,_=plt.hist(m_t,15,label="Sim",alpha=0.5)
                plt.hist(m,b,label="Gen",alpha=0.5)
                plt.legend()  
           # hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            ax[1].set_xlabel(name,fontsize=28)
            ax[0].set_ylabel("Counts",fontsize=28)
            ax[1].set_ylabel("Ratio",fontsize=28)
            
            plt.tight_layout(pad=2)
            if save:
                if v!="m":
                     self.summary.add_figure("inclusive"+v,fig,self.step)
                else:
                    self.summary.add_figure("jet_mass",fig,self.step)
    #             print("added figure")
    #             self.summary.close()
            else:
                plt.show()

    def losses(self,save=False):
        '''This plots the different losses vs epochs'''
        fig=plt.figure()
        hep.cms.label("Private Work",data=None,lumi=None,year=None)
        plt.xlabel('step')
        plt.ylabel('loss')
        ln1=plt.plot(self.model.logprobs,label='logprob')
        if "calc_massloss" in self.config.keys() and self.config["calc_massloss"]:
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
    def plot_scores(self,scores_T,scores_gen,save=False):
        '''This plots the score distribution of the discriminator evaluated on true (simulated) sample scores_T 
        and on generated samples scores_gen'''
        fig=plt.figure()
        a=min(np.quantile(scores_gen,0.001),np.quantile(scores_T,0.001))
        b=max(np.quantile(scores_gen,0.999),np.quantile(scores_T,0.999))
        hep.cms.label("Private Work",data=None,lumi=None,year=None)
        plt.xlabel('Score')
        plt.ylabel('Counts')
        plt.suptitle("Discriminator Prediction")
        _,b,_=plt.hist(scores_T,label="MC Sim",alpha=0.5,bins=np.linspace(a,b,30))
        plt.hist(scores_gen,b,label="Flow Gen",alpha=0.5)
        plt.legend()
        plt.tight_layout(pad=2)
        if save:
            self.summary.add_figure("scores",fig,self.step)
#             self.summary.close()
        else:
            plt.show()
    def plot_cond(self,save=False):
        
        data_module=self.model.data_module
        fig,ax=plt.subplots(1)
        hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax )
        c=torch.ones(N).reshape(-1,1)*(-1)
        gen=self.model.flow.to("cpu").sample(1,c).reshape(-1,90).to("cpu")
        gen=self.model.data_module.scaler.to("cpu").inverse_transform(torch.hstack((gen,torch.ones(N).reshape(-1,1))))
        gen_trafo=torch.clone(gen)
        m=mass(gen[:,:90])
        gen=self.data_module.scaler.transform(torch.hstack((gen[:,:90],m.reshape(-1,1))))
        m=gen[:,-1]
        plt.ylabel("Counts [a.u.]",fontsize=28)
        plt.xlabel(r"$m^{scaled}$ [a.u]",fontsize=28)
        _,b,_=plt.hist((self.data_module.data)[:,-1].numpy(),bins=100,label="Training Data",alpha=0.5)
        plt.hist(m.detach().numpy(),bins=b,label="$m_{cond}=%d$"%c.numpy()[0],alpha=0.5)
        plt.legend()
        plt.show()
        gamma=0.2
        gen_trafo=gen_trafo[:,:90].detach().numpy()[:N,:90]#
        test_set_trafo=data_module.scaler.inverse_transform(data_module.test_set[(data_module.test_set[:,-1]<gamma+c[0])
                                        &(data_module.test_set[:,-1]>c[0]-gamma)])[:N,:90].numpy()[:N,:]

        for v,name in zip(["eta","phi","pt"],[r"$\eta^{rel}_{tot}$",r"$\phi^{rel}_{tot}$",r"$p_{T,tot}^{rel}$"]):
            a=min(np.quantile(gen_trafo.reshape(-1,3)[:N,i],0.001),np.quantile(test_set_trafo.reshape(-1,3)[:N,i],0.001))
            b=max(np.quantile(gen_trafo.reshape(-1,3)[:N,i],0.999),np.quantile(test_set_trafo.reshape(-1,3)[:N,i],0.999))     
            h=hist.Hist(hist.axis.Regular(15,a,b))
            h2=hist.Hist(hist.axis.Regular(15,a,b))
            h.fill(gen_trafo.reshape(-1,3)[:N,i])
            h2.fill(test_set_trafo.reshape(-1,3)[:N,i])
            i+=1
            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(14,14))
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                rp_num_label="Generated $m^{scaled}_{cond}=%d\pm %1.2f$"%(c.numpy()[0],gamma),
                rp_denom_label="Simulated $m^{scaled}=%d\pm %1.2f$"%(c.numpy()[0],gamma),
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0].set_xlabel(None)
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            plt.xlabel(name)
            plt.tight_layout(pad=2)
        if save:
            self.summary.add_figure("scores",fig,self.step)
#             self.summary.close()
        else:
            plt.show()

    def plot_marg_cond(self,save=False):
        
        data_module=self.model.data_module
        fig,ax=plt.subplots(1)
        hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax )
        c=torch.ones(N).reshape(-1,1)*(-1)
        gen=self.model.flow.to("cpu").sample(1,c).reshape(-1,90).to("cpu")
        gen=self.model.data_module.scaler.to("cpu").inverse_transform(torch.hstack((gen,torch.ones(N).reshape(-1,1))))
        gen_trafo=torch.clone(gen)
        m=mass(gen[:,:90])
        gen=self.data_module.scaler.transform(torch.hstack((gen[:,:90],m.reshape(-1,1))))
        m=gen[:,-1]
        plt.ylabel("Counts [a.u.]")
        plt.xlabel(r"$m^{scaled}$ [a.u]")
        _,b,_=plt.hist((self.data_module.data)[:,-1].numpy(),bins=100,label="Training Data",alpha=0.5)
        plt.hist(m.detach().numpy(),bins=b,label="$m_{cond}=%d$"%c.numpy()[0],alpha=0.5)

        plt.legend()
        plt.show()

        gamma=0.2
        gen_trafo=gen_trafo[:,:90].detach().numpy()[:N,:90]#
        test_set_trafo=data_module.scaler.inverse_transform(data_module.test_set[(data_module.test_set[:,-1]<gamma+c[0])
                                        &(data_module.test_set[:,-1]>c[0]-gamma)])[:N,:90].numpy()[:N,:]
        i=0
        for v,name in zip(["eta","phi","pt"],[r"$\eta_1^{rel}$",r"$\phi_1^{rel}$",r"$p_{1,T}^{rel}$"]):
            a=min(np.quantile(gen_trafo[:N,i],0.001),np.quantile(test_set_trafo[:N,i],0.001))
            b=max(np.quantile(gen_trafo[:N,i],0.999),np.quantile(test_set_trafo[:N,i],0.999))     
            h=hist.Hist(hist.axis.Regular(15,a,b))
            h2=hist.Hist(hist.axis.Regular(15,a,b))
            h.fill(gen_trafo[:N,i])
            h2.fill(test_set_trafo[:N,i])
            i+=1
            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]},figsize=(14,14))
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                rp_num_label="Generated $m^{scaled}_{cond}=%d\pm %1.2f$"%(c.numpy()[0],gamma),
                rp_denom_label="Simulated $m^{scaled}=%d\pm %1.2f$"%(c.numpy()[0],gamma),
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0].set_xlabel(None)
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            plt.xlabel(name)
            plt.tight_layout(pad=2)
            if save:
                self.summary.add_figure("scores",fig,self.step)
    #             self.summary.close()
            else:
                plt.show()
    def plot_correlations(self,save=True):
        #Plots correlations between all particles for i=0 eta,i=1 phi,i=2 pt
        self.plot_corr(i=0,save=save)
        self.plot_corr(i=1,save=save)
        self.plot_corr(i=2,save=save)

    def plot_corr(self,i=0,names=["$\eta^{rel}$","$\phi^{rel}$","$p_T$"],save=True):
        if i==2:
            c=1
        else:
            c=.25
        df_g=pd.DataFrame(self.gen[:,:self.n_dim].detach().numpy()[:,range(i,90,3)])
        df_h=pd.DataFrame(self.test_set[:,:self.n_dim].detach().numpy()[:,range(i,90,3)])
        
        fig,ax=plt.subplots(ncols=2,figsize=(15,7.5))
        corr_g = ax[0].matshow(df_g.corr())
        corr_g.set_clim(-c,c)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar=fig.colorbar(corr_g,cax=cax)

        corr_h = ax[1].matshow(df_h.corr())
        corr_h.set_clim(-c,c)
        divider = make_axes_locatable(ax[1])

        cax2 = divider.append_axes('right', size='5%', pad=0.05)
        cbar=fig.colorbar(corr_h,cax=cax2)
        plt.suptitle("{} Correlation between Particles".format(names[i]),fontsize=38)
        ax[0].set_title("Flow Generated",fontsize=34)
        ax[1].set_title("MC Simulated",fontsize=28)
        ax[0].set_xlabel("Particles",fontsize=28)
        ax[0].set_ylabel("Particles",fontsize=28)
        ax[1].set_xlabel("Particles",fontsize=28)
        ax[1].set_ylabel("Particles",fontsize=28)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        if save:
                title=["corr_eta","corr_phi","corr_pt"]
                self.summary.add_figure(title[i],fig,self.step)
                
    #             self.summary.close()
        else:
                plt.show()
    def var_part(self,true,gen,true_n,gen_n,m_true,m_gen,form=2,save=True):
        labels=["$\eta^{rel}$","$\phi^{rel}$","$p^{rel}_T$","$m^{rel}$"]
        names=["eta","phi","pt","m"]
        n,counts=torch.unique(true_n,return_counts=True)
        for j in range(4):
            fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(15,15))

            k=-1
            ntemp=n[-form**2:]

            
            for i in list(ntemp)[::-1]: 
                k+=1
                i=int(i)

                if names[j]!="m":
                    a=np.quantile(self.test_set[true_n.reshape(-1)==i,:].reshape(-1,3)[:,j],0.001)
                    b=np.quantile(self.test_set[true_n.reshape(-1)==i,:].reshape(-1,3)[:,j],0.999)    
                    h=hist.Hist(hist.axis.Regular(15,a,b))
                    h2=hist.Hist(hist.axis.Regular(15,a,b))
                    bins = h.axes[0].edges

                    ax[k//form,k%form].legend()
                    h.fill(self.gen[gen_n.reshape(-1)==i,:].reshape(-1,3)[:,j])
                    h2.fill(self.test_set[true_n.reshape(-1)==i,:].reshape(-1,3)[:,j])
                    
                else:
                    a=np.quantile(m_true[true_n.reshape(-1)==i],0.001)
                    b=np.quantile(m_gen[gen_n.reshape(-1)==i],0.999)

                    h=hist.Hist(hist.axis.Regular(15,a,b))
                    h2=hist.Hist(hist.axis.Regular(15,a,b))
                    bins = h.axes[0].edges
                    h.fill(m_gen[gen_n.reshape(-1)==i])
                    h2.fill(m_true[true_n.reshape(-1)==i])
                    

                h.plot1d(    ax=ax[k//2,k%2])  # line or bar)
                h2.plot1d(    ax=ax[k//2,k%2])  # line or bar)
                ax[k//2,k%2].set_title("{} Distribution for jets with {} particles".format(labels[j],i))

                ax[k//2,k%2].set_xlabel(labels[j])

                ax[k//2,k%2].set_xlim(a,b)

                #plt.tight_layout(pad=2)

            if save:
                self.summary.add_figure("jet{}_{}_part".format(names[j],i),fig,global_step=self.step)
                self.summary.close()
            else:
                plt.show()