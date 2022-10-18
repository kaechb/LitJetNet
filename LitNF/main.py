import datetime
import os
import sys
import time
import traceback

import pandas as pd
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
from helpers import *
if True:
    from jetnet_dataloader import JetNetDataloader

else:
    from jetnet_dataloader import JetNetDataloader
from lit_nf import TransGan
from plotting import plotting
import yaml

# from comet_ml import Experiment


def train(config,  load_ckpt=False, i=0, root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    data_module = JetNetDataloader(config)  # this loads the data
    data_module.setup("training")
    model = TransGan(
        config,  data_module.num_batches
    )  # the sets up the model,  config are hparams we want to optimize
    model.data_module = data_module
    # Callbacks to use during the training, we  checkpoint our models

    callbacks = [
        ModelCheckpoint(
            monitor="val_fpnd",
            save_top_k=10,
            filename="{epoch}-{val_fpnd:.2f}-{val_w1m:.4f}--{val_w1efp:.6f}",
            dirpath=root,
            every_n_epochs=10,
        )
    ]

    if  load_ckpt:
        best_model={}
        best_model["q"]="/beegfs/desy/user/kaechben/bestever_q/epoch=5049-val_fpnd=0.09-val_w1m=0.0007--val_w1efp=0.000008.ckpt"
        best_model["t"]="/beegfs/desy/user/kaechben/bestever_t/epoch=8249-val_fpnd=0.09-val_w1m=0.0007--val_w1efp=0.000011.ckpt"
        best_model["g"]="/beegfs/desy/user/kaechben/bestever_g/epoch=3849-val_fpnd=0.07-val_w1m=0.0007--val_w1efp=0.000006.ckpt"
        best_model["t"]="/beegfs/desy/user/kaechben/bestever_t/epoch=5199-val_fpnd=0.06-val_w1m=0.0009--val_w1efp=0.000011.ckpt"
        best_model["w"]="/beegfs/desy/user/kaechben/part_30w/epoch=449-val_fpnd=1000.00-val_w1m=0.0007--val_w1efp=0.000013.ckpt"
        model = TransGan.load_from_checkpoint(
            best_model[config["parton"]]
        )

        model.data_module = data_module
    # model.config["lr_g"]*=np.random.choice([0.01,0.001])
    # model.config["lr_d"]*=np.random.choice([0.01,0.001])
    # model.config["freq"]=1
    # model.config["sched"]="cosine"
    config=model.config
    
    # config["wgan"]="gan"
    # pl.seed_everything(model.config["seed"], workers=True)
    # model.config["freq"]=20
    # model.config["lr_g"]=0.00001
    # model.config["lr_d"]=0.00001
    # model.config = config #config are our hyperparams, we make this a class property now
    print(root)
    logger = TensorBoardLogger(root)
    print("Version:",logger.version)
    # log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        log_every_n_steps=500,  # auto_scale_batch_size="binsearch",
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=config["val_check"],
        num_sanity_val_steps=1,  # gradient_clip_val=.02, 
        fast_dev_run=False,
        default_root_dir=root,
        precision=32
    )
    # This calls the fit function which trains the model

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":

 
    cols = [
        "name",
        "parton",
        "mass",
        "sched",
        "opt",
        "no_hidden",
        "last_clf",
        "warmup",
        "lr_d",
        "lr_g",
        "ratio"
    ]
    # parton=np.random.choice(["t","q"])
    best_hparam="/home/kaechben/JetNet_NF/LitJetNet/LitNF/bestever_{}/hparams.yaml".format("t")
    with open(best_hparam, 'r') as stream:
        config=yaml.load(stream,Loader=yaml.Loader)
        print(config)
        config=config["config"]
    hyperopt=True

    config["val_check"]=50
    if hyperopt:

        # config["no_hidden"]=np.random.choice([True,False,"more"])
        # config["no_hidden"]=config["no_hidden"]=="True" or config["no_hidden"]=="more"

        config["gen_mask"]=True
        config["bullshitbingo"]=np.random.choice([True,False])
        config["bullshitbingo2"]=np.random.choice([True,False])

        config["max_epochs"]=int(config["max_epochs"])#*np.random.choice([2]))
        config["warmup"]=np.random.choice([1500])
        config["sched"]=np.random.choice(["cosine2",None])

        config["freq"]=np.random.choice([5])    # config["opt"]="Adam"
        config["batch_size"]=int(np.random.choice([1024,2048,3096]))    # config["opt"]="Adam"
        config["dropout"]=np.random.choice([0.1])    
        config["opt"]=np.random.choice(["Adam","RMSprop"])#"AdamW",
        config["lr_g"]=np.random.choice([0.0003,0.0001]) 
        config["ratio"]=np.random.choice([0.9,1,1.1,])
        config["context_features"]=np.random.choice([0])
        config["l_dim"]=np.random.choice([25])
        config["heads"]=np.random.choice([3,4,5])
        config["hidden"]=np.random.choice([128,256,512])
        config["val_check"]=25
        config["lr_d"]=config["lr_g"]*config["ratio"]
        config["l_dim"] = config["l_dim"] * config["heads"]
        config["n_part"] = 30
        config["momentum"]=np.random.choice([True,False])
        config["parton"] =np.random.choice(["t","q","g"])
        config["name"] = config["name"]+config["parton"]
        config["no_hidden_gen"]=np.random.choice([True,False,"more"])
        config["no_hidden"]=np.random.choice([False,"more"])
        config["name"]="part_"+str(config["n_part"])+config["parton"]

        config["name"]="Ml4jets_"+str(config["n_part"])+config["parton"]

        # config["num_layers"]=np.random.choice([2,3,4])    
        config["last_clf"]=False
    else:
        # config["last_clf"]=True
        # config["gen_mask"]=True
        
        # config["name"]="bestever_"+parton#config["parton"]
        config["freq"]=6    # config["opt"]="Adam"
    print(config)
    if len(sys.argv) > 2:
        root = "/beegfs/desy/user/"+ os.environ["USER"]+"/"+config["name"]+"/"+config["parton"]+"_" +"run"+sys.argv[1]+"_"+str(sys.argv[2])
    else:
        root = "/beegfs/desy/user/" + os.environ["USER"] + "/"+ config["name"]

        for col in cols:
            print('"' + col + '":' + str(config[col]))

        train(config, root=root,load_ckpt=False)
   