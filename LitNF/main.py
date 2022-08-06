import datetime
import os
import time
import traceback

import pandas as pd
import pytorch_lightning as pl
import ray
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from scipy import stats
from torch.nn import functional as FF

from helpers import *
from jetnet_dataloader import JetNetDataloader
from lit_nf import TransGan
from plotting import plotting

# from comet_ml import Experiment

def train(config, hyperopt=False, load_ckpt=None,i=0,root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray 
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    #pl.seed_everything(42, workers=True)
    data_module = JetNetDataloader(config) #this loads the data
    model = TransGan(config,hyperopt) # the sets up the model,  config are hparams we want to optimize
    # Callbacks to use during the training, we  checkpoint our models
    
    callbacks = [ModelCheckpoint(monitor="val_w1m",save_top_k=2, filename='{epoch}-{val_logprob:.2f}-{val_w1m:.4f}', dirpath=root,every_n_epochs=10) ]
    
    # if True:#load_ckpt:
    #     model = model.load_from_checkpoint("/beegfs/desy/user/kaechben/t/2022_06_22-10_50-16/epoch=1749-val_logprob=0.94-val_w1m=0.0153.ckpt")
    model.load_datamodule(data_module)#adds datamodule to model
    model.config = config #config are our hyperparams, we make this a class property now
    logger = TensorBoardLogger(root)
    #log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    
    trainer = pl.Trainer(gpus=1, logger=logger,  log_every_n_steps=100,  # auto_scale_batch_size="binsearch",
                          max_epochs=1500*config["freq"], callbacks=callbacks, progress_bar_refresh_rate=int(not hyperopt)*10,
                          check_val_every_n_epoch=100 ,num_sanity_val_steps=0,#gradient_clip_val=.02, gradient_clip_algorithm="norm",
                         fast_dev_run=False,default_root_dir=root)
    # This calls the fit function which trains the model
    trainer.fit(model, train_dataloaders=data_module )  


if __name__ == "__main__":
    
    hyperopt = True  # This sets to run a hyperparameter optimization with ray or just running the training once

    config = {
       "network_layers": 3,  # sets amount hidden layers in transformation networks -scannable
        "network_nodes": 6,  # amount nodes in hidden layers in transformation networks -scannable
            "network_layers_nf": 2,  # sets amount hidden layers in transformation networks -scannable
        "network_nodes_nf": 256,  # amount nodes in hidden layers in transformation networks -scannable
        "batch_size": 4000,  # sets batch size -scannable
        "embedding_features":8,
        "coupling_layers": 15,  # amount of invertible transformations to use -scannable
        "lr": 0.001,  # sets learning rate -scannable
        "batchnorm": False,  # use batchnorm or not -scannable
        "bins":5,  # amount of bins to use in rational quadratic splines -scannable
        "tail_bound": 6,  # splines:max value that is transformed, over this value theree is id  -scannable
        "limit": 150000,  # how many data points to use, test_set is 10% of this -scannable in a sense use 10 k for faster training
        "n_dim": 3,  # how many dimensions to use or equivalently /3 gives the amount of particles to use NEVER EVER CHANGE THIS
        "dropout": 0.5,  # use droput proportion, for 0 there is no dropout -scannable
        "canonical": False,  # transform data coordinates to px,py,pz -scannable
        "max_steps": 100000,  # how many steps to use at max - lower for quicker training
        "lambda": 10,  # balance between massloss and nll -scannable
        "name": "Transflow",  # name for logging folder
        "disc": False,  # whether to train gan style discriminator that decides whether point is simulated or generated-semi-scannable
        "calc_massloss": False, # whether to calculate mass loss, makes training slower, do not use with autoregressive! 
        "context_features":0, #amount of variables used for conditioning, for 0 no conditioning is used, for 1 o nly the mass is used, for 2 also the number part is used
        "variable":1, #use variable amount of particles otherwise only use 30, options are true or false 
        "parton":"t", #choose the dataset you want to train options: t for top,q for quark,g for gluon
        "oversampling":False,
        "wgan":True,
        "corr":True,
        "num_layers":3,
        "autoreg":False,
        "freq":10,
        "n_part":30,
        "fc":False,
        "hidden":256,
        "heads":4,
        "l_dim":20,
        "lr_g":1e-5,
        "lr_d":1e-5,
        "lr_nf":1e-3,
        "sched":True,
        "pretrain":50,
        "opt":"Adam",
        "lambda":10
    
    }     
    print(config["name"])
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
            
            config["sched"]= np.random.choice([True,False])
            config["lambda"]=np.random.choice([1,10,100])

            config["lr_g"]=stats.loguniform.rvs(0.00001, 0.01,size=1)[0]
            config["lr_ratio"]=stats.loguniform.rvs(0.1, 10,size=1)[0]
            config["lr_d"]=config["lr_ratio"]*config["lr_g"]
            config["heads"]=np.random.randint(1, 6)
            config["l_dim"]=config["heads"]*np.random.randint(3,30)
            config["hidden"]=2*np.random.randint(6, 9)
            config["num_layers"]=np.random.randint(2, 6)
            config["corr"]= np.random.choice([True,False],p=[0.7,0.3])
            config["opt"]=np.random.choice(["Adam","AdamW","RMSprop"])
            print(config)
           
            try:
                train(config,hyperopt=hyperopt,i=i,root=temproot)
            except:
                print("error")
                traceback.print_exc()
