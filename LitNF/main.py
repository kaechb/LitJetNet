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
from jetnet_dataloader import JetNetDataloader
from lit_nf import TransGan
from plotting import plotting

# from comet_ml import Experiment


def train(config, hyperopt=False, load_ckpt=None, i=0, root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    data_module = JetNetDataloader(config)  # this loads the data
    data_module.setup("training")
    model = TransGan(
        config, hyperopt, data_module.num_batches
    )  # the sets up the model,  config are hparams we want to optimize
    model.data_module = data_module
    # Callbacks to use during the training, we  checkpoint our models

    callbacks = [
        ModelCheckpoint(
            monitor="val_w1m",
            save_top_k=2,
            filename="{epoch}-{val_fpnd:.2f}-{val_w1m:.4f}",
            dirpath=root,
            every_n_epochs=10,
        )
    ]

    if False:  # load_ckpt:
        model = TransGan.load_from_checkpoint(
            "/beegfs/desy/user/kaechben/Transflow_reloaded2/2022_08_08-18_02-08/epoch=239-val_logprob=0.47-val_w1m=0.0014.ckpt"
        )
        model.data_module = data_module

    # pl.seed_everything(model.config["seed"], workers=True)
    # model.config["freq"]=20
    # model.config["lr_g"]=0.00001
    # model.config["lr_d"]=0.00001
    # model.config = config #config are our hyperparams, we make this a class property now
    print(root)
    logger = TensorBoardLogger(root)
    # log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        log_every_n_steps=10,  # auto_scale_batch_size="binsearch",
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        progress_bar_refresh_rate=int(not hyperopt) * 10,
        check_val_every_n_epoch=config["val_check"],
        num_sanity_val_steps=1,  # gradient_clip_val=.02, 
        fast_dev_run=False,
        default_root_dir=root,
    )
    # This calls the fit function which trains the model

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":

    hyperopt = False  # This sets to run a hyperparameter optimization with ray or just running the training once

    # config = {
    #     "autoreg": False,
    #     "context_features": 0,
    #     "network_layers": 3,  # sets amount hidden layers in transformation networks -scannable
    #     "network_layers_nf": 2,  # sets amount hidden layers in transformation networks -scannable
    #     "network_nodes_nf": 256,  # amount nodes in hidden layers in transformation networks -scannable
    #     "batch_size": 2000,  # sets batch size -scannable #best one 4000
    #     "coupling_layers": 15,  # amount of invertible transformations to use -scannable
    #     "lr": 0.001,  # sets learning rate -scannable
    #     "batchnorm": False,  # use batchnorm or not -scannable
    #     "bins": 5,  # amount of bins to use in rational quadratic splines -scannable
    #     "tail_bound": 6,  # splines:max value that is transformed, over this value theree is id  -scannable
    #     "limit": 150000,  # how many data points to use, test_set is 10% of this -scannable in a sense use 10 k for faster training
    #     "n_dim": 3,  # how many dimensions to use or equivalently /3 gives the amount of particles to use NEVER EVER CHANGE THIS
    #     "dropout": 0.2,  # use droput proportion, for 0 there is no dropout -scannable
    #     "canonical": False,  # transform data coordinates to px,py,pz -scannable
    #     "max_steps": 100000,  # how many steps to use at max - lower for quicker training
    #     "lambda": 100,  # balance between massloss and nll -scannable
    #     "name": "Transflow_best",  # name for logging folder
    #     "disc": False,  # whether to train gan style discriminator that decides whether point is simulated or generated-semi-scannable
    #     "variable": 1,  # use variable amount of particles otherwise only use 30, options are true or False
    #     "parton": "t",  # choose the dataset you want to train options: t for top,q for quark,g for gluon
    #     "wgan": False,
    #     "corr": True,
    #     "num_layers": 5,
    #     "freq": 10,
    #     "n_part": 30,
    #     "fc": False,
    #     "hidden": 80,
    #     "heads": 3,
    #     "l_dim": 63,
    #     "lr_g": 1e-4,
    #     "lr_d": 1e-4,
    #     "lr_nf": 0.000722,
    #     "sched": "cosine2",
    #     "opt": "SGD",
    #     "lambda": 1,
    #     "max_epochs": 1600,
    #     "mass": True,
    #     "no_hidden": True,
    #     "clf": True,
    #     "val_check": 50,
    #     "frac_pretrain": 80,
    # }
    cols = [
        "name",
        "parton",
        "mass",
        "sched",
        "opt",
        "no_hidden",
        "clf",
        "batch_size",
        "freq",
        "seed",
        "lr_g",
        "heads",
        "hidden",
        "l_dim",
        "num_layers",
        "val_check",
    ]
    config = {
        "autoreg": False,
        "context_features": 0,
        "network_layers": 3,
        "network_layers_nf": 2,
        "network_nodes_nf": 256,
        "batch_size": 1024,
        "coupling_layers": 15,
        "lr": 0.001,
        "batchnorm": False,
        "bins": 5,
        "tail_bound": 6,
        "limit": 200000,
        "n_dim": 3,
        "dropout": 0.2,
        "canonical": False,
        "max_steps": 100000,
        "lambda": 100,
        "name": "Transflow_best",
        "disc": False,
        "variable": 1,
        "parton": "t",
        "wgan": True,
        "corr": True,
        "num_layers": 4,
        "freq": 6,
        "n_part": 30,
        "fc": False,
        "hidden": 500,
        "heads": 6,
        "l_dim": 25,
        "lr_g": 0.0004327405312571664,
        "lr_d": 0.0004327405312571664,
        "lr_nf": 0.000722,
        "sched": "cosine",
        "opt": "Adam",
        "max_epochs": 1600,
        "mass": True,
        "no_hidden": False,
        "clf": True,
        "val_check": 50,
        "frac_pretrain": 80,
        "seed": 69,
        "quantile": False,
    }  #'seed': 744,sched:"None","wgan":False,"freq":8,"sched":None,"heads":4
    config={'autoreg': False, 'context_features': 0, 'network_layers': 3, 'network_layers_nf': 2, 'network_nodes_nf': 256, 'batch_size': 1024, 'coupling_layers': 15, 'lr': 0.001, 'batchnorm': False, 'bins': 5, 'tail_bound': 6, 'limit': 150000, 'n_dim': 3, 'dropout': 0.2, 'canonical': False, 'max_steps': 100000, 'lambda': 1, 'name': 'Transflow_best', 'disc': False, 'variable': 1, 'parton': 't', 'wgan': False, 'corr': True, 'num_layers': 4, 'freq': 6, 'n_part': 30, 'fc': False, 'hidden': 500, 'heads': 4, 'l_dim': 25, 'lr_g': 0.0004327405312571664, 'lr_d': 0.0004327405312571664, 'lr_nf': 0.000722, 'sched': "cosine2", 'opt': 'RMSprop', 'max_epochs': 3200, 'mass': True, 'no_hidden': False, 'clf': True, 'val_check': 50, 'frac_pretrain': 80,"seed":69,"quantile": False, }#'seed':
    config["l_dim"] = config["l_dim"] * config["heads"]

    print(config["name"])

    if len(sys.argv) > 2:
        root = "/beegfs/desy/user/"+ os.environ["USER"]+"/"+config["name"]+"/"+config["parton"]+"_" +"run"+sys.argv[1]+"_"+str(sys.argv[2])
    else:
        root = "/beegfs/desy/user/" + os.environ["USER"] + "/"+ config["name"]
    if not hyperopt:
        hyperopt = True
        for col in cols:
            print('"' + col + '":' + str(config[col]))

        train(config, hyperopt=hyperopt, root=root)
    else:
        # if not os.path.isfile("/beegfs/desy/user/{}/ray_results/{}/summary.csv".format(os.environ["USER"],config["parton"])):
        #     pd.DataFrame().to_csv("/beegfs/desy/user/{}/ray_results/{}/summary.csv".format(os.environ["USER"],config["parton"]))
        num_samples = 1  # how many hparam settings to sample with ray
        resources = {"cpu": 10, "gpu": 0.5}
        # This sets the logging in ray
        # reporter = CLIReporter(max_progress_rows=40, max_report_frequency=300, sort_by_metric=True,
        #                        metric="logprob", parameter_columns=["network_nodes", "network_layers", "coupling_layers", "lr"])
        for i in range(num_samples):

            temproot = root

            config["sched"] = np.random.choice(["cosine", "cosine2", None])
            #config["opt"] = np.random.choice(["Adam", "RMSprop"])
            #config["mass"] = np.random.choice([True, False])
            config["quantile"] = np.random.choice([True, False])
            config["no_hidden"] = np.random.choice([True, False,"more"])
            config["clf"] = np.random.choice([True, False])
            # config["batch_size"] = 2 ** np.random.randint(8, 12)
            config["freq"] = np.random.randint(5, 10)
            config["seed"] = int(np.random.randint(1, 1000))
            
            # config["lr_g"] = stats.loguniform.rvs(0.00001, 0.001, size=1)[0]
            
            
            config["heads"] = np.random.randint(3, 6)
            config["l_dim"] = config["heads"] * 25
            # config["hidden"] = 100 * np.random.randint(2, 7)
            config["num_layers"] = np.random.randint(3, 6)

            for col in cols:
                print('"' + col + '":' + str(config[col]))

            try:
                train(config, hyperopt=hyperopt, i=i, root=temproot)
            except:
                print("error")
                traceback.print_exc()
