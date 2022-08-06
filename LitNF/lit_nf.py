
import traceback
import os
import nflows as nf
from nflows.utils.torchutils import create_random_binary_mask
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import *
from nflows.nn import nets
from nflows.flows.base import Flow
from nflows.flows import base
from nflows.transforms.coupling import *
from nflows.transforms.autoregressive import *
from particle_net import ParticleNet
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR,ReduceLROnPlateau,ExponentialLR
import torch
from torch import nn
from torch.nn import functional as FF
import numpy as np
from jetnet.evaluation import w1p, w1efp, w1m, cov_mmd,fpnd
import mplhep as hep
import hist
from hist import Hist
from pytorch_lightning.loggers import TensorBoardLogger
from collections import OrderedDict
from ray import tune
from helpers import *
from plotting import *
import pandas as pd
import os
from helpers import CosineWarmupScheduler
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import time
from torch.nn.functional import leaky_relu,sigmoid
class Gen(nn.Module):
    
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,n_part=5,fc=False,dropout=0.5):
        super().__init__()
        self.hidden_nodes=hidden
        self.n_dim=n_dim
        self.l_dim=l_dim
        self.n_part=n_part
        
       
        self.fc=fc
        if fc:
            self.l_dim*=n_part 
            self.embbed_flat=nn.Linear(n_dim*n_part,l_dim)
            self.flat_hidden=nn.Linear(l_dim,hidden)
            self.flat_hidden2=nn.Linear(hidden,hidden)
            self.flat_hidden3=nn.Linear(hidden,hidden)
            self.flat_out=nn.Linear(hidden,n_dim*n_part)
        else:
            self.embbed=nn.Linear(n_dim,l_dim)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=l_dim,nhead=num_heads,batch_first=True,norm_first=False
            ,dim_feedforward=hidden,dropout=dropout) ,num_layers=num_layers)
            self.hidden=nn.Linear(l_dim,hidden)
            self.dropout=nn.Dropout(0.5)
            self.out=nn.Linear(l_dim, n_dim)
            self.out_flat=nn.Linear(hidden,n_dim*n_part )
        
    def forward(self,x):

        if self.fc:
            x=x.reshape(len(x),self.n_part*self.n_dim)
            x=self.embbed_flat(x)
            x=leaky_relu(self.flat_hidden(x))
#             x = self.dropout(x)
            x=self.flat_out(x)
            x=x.reshape(len(x),self.n_part,self.n_dim)
        else:
            x=self.embbed(x)
            x=self.encoder(x)
#             x=leaky_relu(self.hidden(x))
            x=self.out(x)
        return x

class Disc(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,n_part=2,fc=False,dropout=0.5):
        super().__init__()
        self.hidden_nodes=hidden
        self.n_dim=n_dim
#         l_dim=n_dim
        self.l_dim=l_dim
        self.n_part=n_part
        self.fc=fc

        
        if fc:
            self.l_dim*=n_part 
            self.embbed_flat=nn.Linear(n_dim*n_part,l_dim)
            self.flat_hidden=nn.Linear(l_dim,hidden)
            self.flat_hidden2=nn.Linear(hidden,hidden)
            self.flat_hidden3=nn.Linear(hidden,hidden)
            self.flat_out=nn.Linear(hidden,1)
        else:
            self.embbed=nn.Linear(n_dim,l_dim)
            self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.l_dim,nhead=num_heads,dim_feedforward=hidden,dropout=dropout,norm_first=False,
                                       activation=lambda x:leaky_relu(x,0.2),batch_first=True) ,num_layers=num_layers)
            self.hidden=nn.Linear(l_dim,2*hidden)
            self.hidden2=nn.Linear(2*hidden,hidden)
            self.out=nn.Linear(hidden,1)

    def forward(self,x):

        if self.fc==True:
            x=x.reshape(len(x),self.n_dim*self.n_part)
            x=self.embbed_flat(x)
            x=leaky_relu(self.flat_hidden(x),0.2)
            x=leaky_relu(self.flat_hidden2(x),0.2)
            x=self.flat_out(x)
        else:
            x=self.embbed(x)
            x=torch.concat((torch.ones_like(x[:,0,:]).reshape(len(x),1,-1),x),axis=1)
            
            x=self.encoder(x)
            
#             x=torch.sum(x,axis=1)
            x=x[:,0,:]
            x=leaky_relu(self.hidden(x),0.2)
            
            x=leaky_relu(self.hidden2(x),0.2)
            
            x=self.out(x)
        return x
      
class TransGan(pl.LightningModule):
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
                hidden_features=self.config["network_nodes_nf"],
                context_features=c,
                num_blocks=self.config["network_layers_nf"],
                activation=self.config["activation"]  if "activation" in self.config.keys() else FF.relu,
                dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
                use_batch_norm=self.config["batchnorm"] if "batchnorm" in self.config.keys() else 0
        )

    def __init__(self,config,hyperopt):
        
        '''This initializes the model and its hyperparameters'''
        super().__init__()
        self.hyperopt=True
        self.config=config
        self.automatic_optimization=False
        self.freq_d=config["freq"]
        self.wgan=config["wgan"]
        #Metrics to track during the training
        self.metrics={"val_w1p":[],"val_w1m":[],"val_w1efp":[],"val_cov":[],"val_mmd":[],"val_fpnd":[],"val_logprob":[],"step":[]}
        #Loss function of the Normalizing flows
        self.logprobs=[]
        self.n_part=config["n_part"]
        self.hparams.update(config)
        self.save_hyperparameters()
        self.flows = []
        self.n_dim=self.config["n_dim"]
        self.n_part=config["n_part"]
        self.add_corr=config["corr"]
        self.alpha=1
        K=self.config["coupling_layers"]
        for i in range(K):
            '''This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not'''
           
            if self.config["autoreg"]:
                self.flows += [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
#                         random_mask=True,
                        features=self.n_dim,
                        hidden_features=128,
                        use_residual_blocks=True, 
                        tails='linear',
                        tail_bound=self.config["tail_bound"],
                        num_bins=self.config["bins"] )]
            else:
                mask=create_random_binary_mask(self.n_dim*self.n_part)            
                self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                            mask=mask,
                            transform_net_create_fn= self.create_resnet, 
                            tails='linear',
                            tail_bound=self.config["tail_bound"],
                            num_bins=self.config["bins"] )]

        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim*self.n_part])
        self.q_test =nf.distributions.normal.StandardNormal([self.n_dim*self.n_part])
        #Creates working flow model from the list of layer modules
        self.flows=CompositeTransform(self.flows)
        # Construct flow model
        self.flow_test= base.Flow(distribution=self.q_test, transform=self.flows)
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)

        self.gen_net = Gen(n_dim=self.n_dim,hidden=config["hidden"],num_layers=config["num_layers"],dropout=config["dropout"],
                           fc= config["fc"],n_part=config["n_part"],l_dim=config["l_dim"],num_heads=config["heads"]).cuda()
        self.dis_net = Disc(n_dim=self.n_dim,hidden=config["hidden"],l_dim=config["l_dim"],num_layers=config["num_layers"],
                            num_heads=config["heads"],fc=config["fc"],n_part=config["n_part"],dropout=config["dropout"]).cuda()
        self.sig=nn.Sigmoid()
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        self.d_train=True
    

    def load_datamodule(self,data_module):
        '''needed for lightning training to work, it just sets the dataloader for training and validation'''
        self.data_module=data_module
        
    def on_after_backward(self) -> None:
        '''This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues'''
        valid_gradients = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("not valid grads",self.counter)
            self.zero_grad()
            self.counter+=1
            if self.counter>5:
                raise ValueError('5 nangrads in a row')
        else:
            self.counter=0

    def sampleandscale(self,batch,scale=False):
        '''This is a helper function that samples from the flow (i.e. generates a new sample) 
            and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
            on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
            because calculating the mass is a non linear transformation and does not commute with the mass calculation''' 
        z=self.flow.sample(len(batch)).reshape(len(batch),self.n_part,self.n_dim)
        if self.add_corr: 
            fake=z+self.gen_net(z)#(1-self.alpha)*
            fake=fake.reshape(len(batch),self.n_part,self.n_dim)
        else:
            fake=self.gen_net(z)
        assert batch.device==fake.device

        self.data_module.scaler=self.data_module.scaler.to(batch.device)
        if scale:
            fake_scaled=self.data_module.scaler.inverse_transform(fake.reshape(len(batch),self.n_dim*self.n_part))
            z_scaled=self.data_module.scaler.inverse_transform(z.reshape(len(batch),self.n_dim*self.n_part))
            true=self.data_module.scaler.inverse_transform(batch)
            return fake,batch,z,fake_scaled,true,z_scaled
        else:
            return fake
        
    def configure_optimizers(self):
        self.batch_size=self.config["batch_size"]
        self.losses=[]
        
        #mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        if self.config["opt"]=="Adam":

            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.config["lr_g"],betas=(0,.9))
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.config["lr_d"],betas=(0,.9))
        elif self.config["opt"]=="AdamW":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.config["lr_g"],betas=(0,.9))
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.config["lr_d"],betas=(0,.9))
        else:
            opt_g = torch.optim.RMSprop(self.gen_net.parameters(), lr=self.config["lr_g"])
            opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.config["lr_d"])
        opt_nf = torch.optim.AdamW(self.flow.parameters(), lr=self.config["lr_nf"] )
        lr_scheduler_nf = CosineWarmupScheduler(opt_nf,warmup=1,max_iters=10000000*self.config["freq"]) 
        lr_scheduler_d =None if not self.config["sched"] else CosineWarmupScheduler(opt_d,warmup=300*self.config["freq"],max_iters=1500*self.config["freq"])
        lr_scheduler_g =None if not self.config["sched"] else CosineWarmupScheduler(opt_g,warmup=300,max_iters=1500)
        if self.config["sched"]:
            return  [opt_nf,opt_d,opt_g],[lr_scheduler_nf,lr_scheduler_d,lr_scheduler_g]
        else:
            return [opt_nf,opt_d,opt_g] 
    
    
    def compute_gradient_penalty(self,D, real_samples, fake_samples, phi):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0),1, 1))).to(real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D.train()(interpolates)
        fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
        return gradient_penalty


    def _summary(self,temp):
            first=False
            self.summary_path="/beegfs/desy/user/{}/{}/summary.csv".format(os.environ["USER"],self.config["name"])
            if self.global_step==0:
                self.start=time.time()
            if os.path.isfile(self.summary_path):
                
                summary=pd.read_csv(self.summary_path).set_index(["path_index"])
            else:
                print("summary not found")
                summary=pd.DataFrame()
                first=True
                
            summary.loc[self.logger.log_dir,self.config.keys()]=self.config.values()
            summary.loc[self.logger.log_dir,temp.keys()]=temp.values()
            summary.loc[self.logger.log_dir,"time"]=time.time()-self.start          
            summary.to_csv(self.summary_path,index_label=["path_index"])  
            return summary
    
    def _results(self):
        self.metrics["step"].append(self.global_step)
        self.df=pd.DataFrame.from_dict(self.metrics)
        self.df.to_csv(self.logger.log_dir+"result.csv",index_label=["index"])
    
   
    
    def training_step(self, batch, batch_idx):
        """training loop of the model, here all the data is passed forward to a gaussian
            This is the important part what is happening here. This is all the training we do """
        opt_nf,opt_d,opt_g=self.optimizers()
        if self.config["sched"]:
            sched_nf,sched_d,sched_g=self.lr_schedulers()

        pretrain=self.config["pretrain"]
        nf_loss=0
        d_loss_avg=0
        gradient_penalty=0

        ### NF PART
        if self.config["sched"]:
            sched_nf.step()
        if self.global_step<8000:
            nf_loss -=self.flow.to(self.device).log_prob(batch).mean()#c if self.config["context_features"] else None
            nf_loss/=(self.n_dim*self.n_part) 
            opt_nf.zero_grad()
            self.manual_backward(nf_loss)
            opt_nf.step()
            self.log("logprob", nf_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) 
        
        ### GAN PART
        batch=batch.reshape(len(batch),self.n_part,self.n_dim)
        fake=self.sampleandscale(batch,scale=False)
        if self.current_epoch>pretrain :
            pred_real=self.dis_net(batch)
            pred_fake=self.dis_net(fake.detach())
            if self.wgan:
                gradient_penalty =  self.compute_gradient_penalty(self.dis_net, batch, fake.detach(),1)
                self.log("gradient penalty",gradient_penalty,logger=True)
                d_loss=-torch.mean(pred_real.view(-1))+torch.mean(pred_fake.view(-1))+self.config["lambda"]*gradient_penalty
            else:               
                target_real=torch.ones_like(pred_real)
                target_fake=torch.zeros_like(pred_fake)
                pred=torch.vstack((pred_real,pred_fake))
                target=torch.vstack((target_real,target_fake))
                d_loss=nn.MSELoss()(pred,target).mean()
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            d_loss_avg+=d_loss.cpu().detach().numpy()-10*gradient_penalty.cpu().detach().numpy()
            self.log("d_loss",d_loss,logger=False,prog_bar=True)
            self.logger.experiment.add_scalars("d_losses",{"train_disc":d_loss_avg},global_step=self.global_step)
            if self.config["sched"]:
                sched_d.step()
        
        if self.current_epoch>(pretrain*2) and self.global_step%self.freq_d<2:
            opt_g.zero_grad()
            pred_fake=(self.dis_net(fake))
            target_real=torch.ones_like(pred_fake)
            if self.wgan:
                g_loss=-torch.mean(pred_fake.view(-1))
            else:
                g_loss=nn.MSELoss()((pred_fake.view(-1)),target_real.view(-1))
            self.manual_backward(g_loss)
            opt_g.step()
            self.logger.experiment.add_scalars("g_losses",{"train_gen":g_loss},global_step=self.global_step)
            self.log("g_loss",g_loss,logger=False,prog_bar=True)
            if self.config["sched"]:
                sched_g.step()

        # Control plot train
        if self.current_epoch%5==0 and self.current_epoch>pretrain :
            fig,ax=plt.subplots()
            ax.hist(pred_fake.detach().cpu().numpy(),label="fake",bins=np.linspace(0,1,30) if not self.wgan else 30,histtype='step')
            ax.hist(pred_real.detach().cpu().numpy(),label="real",bins=np.linspace(0,1,30) if not self.wgan else 30,histtype='step')
            ax.legend()
            plt.ylabel("Counts")
            plt.xlabel("Critic Score")
            self.logger.experiment.add_figure("class_train",fig,global_step=self.current_epoch)            
        
    
    def validation_step(self, batch, batch_idx):
        '''This calculates some important metrics on the hold out set (checking for overtraining)'''
        self.data_module.scaler.to("cpu")  
        batch=batch.to("cpu")
        self.flow=self.flow.to("cpu")
        self.dis_net=self.dis_net.cpu()
        self.gen_net=self.gen_net.cpu()
        c=None       
        with torch.no_grad():
            logprob=-self.flow.log_prob(batch).mean()/90

            gen,true,z,fake_scaled,true_scaled,z_scaled=self.sampleandscale(batch,scale=True)
            scores_fake=self.dis_net(gen)
            scores_real=self.dis_net(true.reshape(len(batch),self.n_part,self.n_dim))
            scores_nf=self.dis_net(z.reshape(len(batch),self.n_part,self.n_dim))
        bins=50
        if not self.wgan:
            bins=np.linspace(0,1,bins)
            scores_nf=nn.Sigmoid()(scores_nf)
            scores_real=nn.Sigmoid()(scores_real)
            scores_fake=nn.Sigmoid()(scores_fake)
            real_loss=nn.BCELoss()(scores_real,torch.ones_like(scores_nf))
            fake_loss_gan=nn.BCELoss()(scores_fake,torch.zeros_like(scores_nf))
            fake_loss_nf=nn.BCELoss()(scores_nf,torch.zeros_like(scores_nf))
            g_loss=nn.BCELoss()(scores_fake,torch.ones_like(scores_nf))
            num=2
            d_loss=(real_loss+fake_loss_gan)/num
        else:
            d_loss=-torch.mean(scores_real.view(-1))+torch.mean(scores_fake.view(-1))#+10*gradient_penalty
            g_loss=-torch.mean(scores_fake.view(-1))
        
        fig=plt.figure()
        _,bins,_=plt.hist(scores_nf.numpy(),bins=bins,label="nf",alpha=0.5)
        plt.hist(scores_real.numpy(),bins=bins,label="true",alpha=0.5)
        plt.hist(scores_fake.numpy(),bins=bins,label="corr",alpha=0.5)
        plt.xlabel("Critic Score")
        plt.ylabel("Counts")
        plt.legend()
        true_scaled,fake_scaled,z_scaled=true_scaled.reshape(-1,90),fake_scaled.reshape(-1,90),z_scaled.reshape(-1,90)
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        m_t=mass(true_scaled[:,:self.n_dim*self.n_part].to(self.device),self.config["canonical"]).cpu()
        m_gen=mass(z_scaled[:,:self.n_dim*self.n_part],self.config["canonical"]).cpu()
        m_c=mass(fake_scaled[:,:self.n_dim*self.n_part],self.config["canonical"]).cpu()
        for i in range(30):
            i=2+3*i
            # gen[gen[:,i]<0,i]=0
            fake_scaled[fake_scaled[:,i]<0,i]=0
            true_scaled[true_scaled[:,i]<0,i]=0
          #Some metrics we track
        cov,mmd=cov_mmd(fake_scaled.reshape(-1,self.n_part,self.n_dim),true_scaled.reshape(-1,self.n_part,self.n_dim),use_tqdm=False)
        try:
            fpndv=fpnd(fake_scaled.reshape(-1,self.n_part,self.n_dim).numpy(),use_tqdm=False,jet_type=self.config["parton"])
        except:
            fpndv=1000
        self.metrics["val_fpnd"].append(fpndv)
        self.metrics["val_logprob"].append(logprob)
        self.metrics["val_mmd"].append(mmd)
        self.metrics["val_cov"].append(cov)
        self.metrics["val_w1p"].append(w1p(fake_scaled.reshape(len(batch),self.n_part,self.n_dim),true_scaled.reshape(len(batch),self.n_part,self.n_dim)))
        self.metrics["val_w1m"].append(w1m(fake_scaled.reshape(len(batch),self.n_part,self.n_dim),true_scaled.reshape(len(batch),self.n_part,self.n_dim)))
        self.metrics["val_w1efp"].append(w1efp(fake_scaled.reshape(len(batch),self.n_part,self.n_dim),true_scaled.reshape(len(batch),self.n_part,self.n_dim)))
        
        
        temp={"val_logprob":logprob.numpy(),"val_fpnd":fpndv,"val_mmd":mmd,"val_cov":cov,"val_w1m":self.metrics["val_w1m"][-1][0],"val_w1efp":self.metrics["val_w1efp"][-1][0],"val_w1p":self.metrics["val_w1p"][-1][0],"step":self.global_step}
        
        print("step {}: ".format(self.global_step),temp)
        if self.hyperopt:
            self._results()
            summary=self._summary(temp)

        self.log("val_w1m",self.metrics["val_w1m"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1p",self.metrics["val_w1p"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_w1efp",self.metrics["val_w1efp"][-1][0],on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_logprob",logprob,prog_bar=True,logger=True)
        self.log("val_cov",cov,prog_bar=True,logger=True,on_step=False, on_epoch=True)
        self.log("val_fpnd",fpndv,prog_bar=True,logger=True,on_step=False, on_epoch=True)
        self.log("val_mmd",mmd,prog_bar=True,logger=True,on_step=False, on_epoch=True)

        self.plot=plotting(model=self,gen=z_scaled,gen_corr=fake_scaled,true=true_scaled,config=self.config,step=self.global_step,logger=self.logger.experiment)  
        try:
            self.plot.plot_mass(m=m_gen.cpu().numpy(),m_t=m_t.cpu().numpy(),m_c=m_c.cpu().numpy(),save=True,bins=15,quantile=True,plot_vline=False)
            self.plot.plot_2d(save=True)
#             self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc() 
        self.flow=self.flow.to("cuda")
        self.gen_net=self.gen_net.to("cuda")
        self.dis_net=self.dis_net.to("cuda")
    
