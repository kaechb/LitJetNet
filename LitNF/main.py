from pytorch_lightning.callbacks import ModelCheckpoint
# from comet_ml import Experiment

import pytorch_lightning as pl
import os
from plotting import plotting
from torch.nn import functional as FF
from lit_nf import LitNF
from jetnet_dataloader import JetNetDataloader
from helpers import *
from ray import tune
import ray
from pytorch_lightning.loggers import CometLogger
from ray.tune import CLIReporter
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import os
from scipy import stats
import datetime
import pandas as pd
import traceback
import time
def train(config, hyperopt=False, load_ckpt=None,i=0,root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray 
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    data_module = JetNetDataloader(config) #this loads the data
    model = LitNF(config,hyperopt) # the sets up the model,  config are hparams we want to optimize
    # Callbacks to use during the training, we  checkpoint our models
    
    callbacks = [ModelCheckpoint(monitor="val_logprob",save_top_k=2, filename='{epoch}-{val_logprob:.2f}-{val_w1m:.4f}', dirpath=root,every_n_epochs=10) ]
    
    # if True:#load_ckpt:
    #     model = model.load_from_checkpoint("/beegfs/desy/user/kaechben/t/2022_06_22-10_50-16/epoch=1749-val_logprob=0.94-val_w1m=0.0153.ckpt")
    model.load_datamodule(data_module)#adds datamodule to model
    model.config = config #config are our hyperparams, we make this a class property now
    logger = TensorBoardLogger(root)
    #log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    
    trainer = pl.Trainer(gpus=1, logger=logger,  log_every_n_steps=001,  # auto_scale_batch_size="binsearch",
                          max_steps=100000 if config["oversampling"]  else config["max_steps"], callbacks=callbacks, progress_bar_refresh_rate=int(not hyperopt)*10,
                          check_val_every_n_epoch=10,track_grad_norm=2 ,num_sanity_val_steps=1,#gradient_clip_val=.02, gradient_clip_algorithm="norm",
                         fast_dev_run=False,default_root_dir=root,max_epochs=-1)
    # This calls the fit function which trains the model
    trainer.fit(model, train_dataloaders=data_module )  


if __name__ == "__main__":
    
    hyperopt = True  # This sets to run a hyperparameter optimization with ray or just running the training once

    config = {
       "network_layers": 2,  # sets amount hidden layers in transformation networks -scannable
        "network_nodes": 256,  # amount nodes in hidden layers in transformation networks -scannable
        "batch_size": 2500,  # sets batch size -scannable
        "coupling_layers": 10,  # amount of invertible transformations to use -scannable
        "lr": 0.001,  # sets learning rate -scannable
        "batchnorm": False,  # use batchnorm or not -scannable
        "bins": 8,  # amount of bins to use in rational quadratic splines -scannable
        "tail_bound": 6,  # splines:max value that is transformed, over this value theree is id  -scannable
        "limit": -1,  # how many data points to use, test_set is 10% of this -scannable in a sense use 10 k for faster training
        "n_dim": 90,  # how many dimensions to use or equivalently /3 gives the amount of particles to use NEVER EVER CHANGE THIS
        "dropout": 0.4,  # use droput proportion, for 0 there is no dropout -scannable
        "lr_schedule": False,  # whether tos chedule the learning rate can be False or "smart","exp","onecycle" -semi-scannable
        "n_sched": 1000,  # how many steps between an annealing step -semi-scannable
        "canonical": False,  # transform data coordinates to px,py,pz -scannable
        "max_steps": 5,  # how many steps to use at max - lower for quicker training
        "lambda": 10,  # balance between massloss and nll -scannable
        "n_mse_turnoff": 10000000,  # when to turn off mass loss -scannable
        "n_mse_delay": 5,  # when to turn on mass loss -scannable
        "name": "q",  # name for logging folder
        "disc": False,  # whether to train gan style discriminator that decides whether point is simulated or generated-semi-scannable
        "calc_massloss": True, # whether to calculate mass loss, makes training slower, do not use with autoregressive! 
        "context_features":1, #amount of variables used for conditioning, for 0 no conditioning is used, for 1 o nly the mass is used, for 2 also the number part is used
        "variable":1, #use variable amount of particles otherwise only use 30, options are true or false 
        "spline":True,#whether to use splines or not, can also be set to "autoregressive" but they are unstable
        "parton":"t", #choose the dataset you want to train options: t for top,q for quark,g for gluon
        "oversampling":False
    }
    config["name"]=config["parton"]
    config["name"]=config["name"]+"working"
    root="/beegfs/desy/user/"+os.environ["USER"]+"/"+config["name"]+"/"+datetime.datetime.now().strftime("%Y_%m_%d-%H_%M-%S")

    if not hyperopt:
        hyperopt=True
        train(config,hyperopt=hyperopt,root=root)
    else:
        # if not os.path.isfile("/beegfs/desy/user/{}/ray_results/{}/summary.csv".format(os.environ["USER"],config["parton"])):
        #     pd.DataFrame().to_csv("/beegfs/desy/user/{}/ray_results/{}/summary.csv".format(os.environ["USER"],config["parton"]))
        num_samples = 500 # how many hparam settings to sample with ray
        resources = {"cpu": 10, "gpu": 0.5}
        # This sets the logging in ray
        # reporter = CLIReporter(max_progress_rows=40, max_report_frequency=300, sort_by_metric=True,
        #                        metric="logprob", parameter_columns=["network_nodes", "network_layers", "coupling_layers", "lr"])
        for i in range(num_samples):
            
            temproot=root+"/train_"+str(i)
            config["network_layers"]=np.random.randint(1, 4)
            config["network_nodes"]= np.random.randint(250, 500)
            config["coupling_layers"]= np.random.randint(5, 20)
            config["lr"]= stats.loguniform.rvs(0.0005, 0.001,size=1)[0]
            config["batchnorm"]= np.random.choice([True,False])
            config["n_mse_turnoff"]= np.random.choice([0,1500,3000])
            
            config["bins"]= np.random.randint(4, 10)
            config["tail_bound"]= np.random.randint(3, 10)
            config["dropout"]= np.random.rand()*0.5
            config["lambda"]= stats.loguniform.rvs(0.01, 200,size=1)[0]
            config["spline"]= np.random.choice([True,False])
            config["context_features"]= np.random.randint(0, 3) 
            print(config)
           
            try:
                train(config,hyperopt=hyperopt,i=i,root=temproot)
            except:
                print("error")
                traceback.print_exc()
        # reporter.add_metric_column("m_loss")
        # reporter.add_metric_column("logprob")
        # reporter.add_metric_column("val_w1p")
        # reporter.add_metric_column("val_w1efp")
        # reporter.add_metric_column("val_w1m")
        # ray.init("auto") # This connects to the main ray node
        
        #this starts the training 
        # result = tune.run(tune.with_parameters(
        #     train,  hyperopt=True),
        #     resources_per_trial=resources,
        #     config=config,
        #     num_samples=num_samples,
        #     progress_reporter=reporter,
        #     # checkpoint_freq=100,  # Checkpoint every 100 epoch
        #     local_dir="/beegfs/desy/user/"+os.environ["USER"]+"/ray_results/"+config["name"]+"/",
        #     verbose=2,
        # )
