import datetime
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF

from helpers import *
from jetnet_dataloader import JetNetDataloader
from training import ParGan

# from comet_ml import Experiment


def train(config, load_ckpt=False, i=0, root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    data_module = JetNetDataloader(config)  # this loads the data
    data_module.setup("training")
    model = ParGan(
        config, data_module.num_batches
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
    if load_ckpt:
        model = TransGan.load_from_checkpoint(load_ckpt)
        model.data_module = data_module
        model.config["ckpt"] = True
    if "pretrain" in config.keys() and config["pretrain"]:
        model.config["lr_g"] = config["lr_g"]
        model.config["lr_d"] = config["lr_d"]
        model.config["ratio"] = config["ratio"]
        model.config["freq"] = config["freq"]
        model.config["sched"] = config["sched"]
        model.config["batch_size"] = config["batch_size"]
        model.config["opt"] = config["opt"]
        model.config["name"] = config["name"]
    # pl.seed_everything(model.config["seed"], workers=True)
    # model.config["freq"]=20
    # model.config["lr_g"]=0.00001
    # model.config["lr_d"]=0.00001
    # model.config = config #config are our hyperparams, we make this a class property now
    print(root)
    logger = TensorBoardLogger(root)  # ,version=version_name
    # log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        log_every_n_steps=50,  # auto_scale_batch_size="binsearch",
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=config["val_check"],
        num_sanity_val_steps=1,  # gradient_clip_val=.02,
        fast_dev_run=False,
        default_root_dir=root,
        track_grad_norm=2,
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
        "ratio",
    ]
    parton = np.random.choice(["q"])  # "q","g",

    best_hparam = (
        "/home/kaechben/JetNet_NF/LitJetNet/LitNF/bestever_{}/hparams.yaml".format(
            parton
        )
    )
    with open(best_hparam, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
        config = config["config"]
    delete = [
        "autoreg",
        "disc",
        "fc",
        "quantile",
        "bullshitbingo",
        "bullshitbingo2",
        "coupling_layers",
        "batchnorm",
        "lr_nf",
    ]
    for key in delete:
        config.pop(key, None)
    hyperopt = True
    config["val_check"] = 50
    config["parton"] = parton
    config["pretrain"] = np.random.choice([True, False])
    ckpt = False
    config = {
        "val_check": 50,
        "parton": parton,
        "affine_add": True,
        "warmup": 1200,
        "sched": "linear",
        "mass": True,
        "momentum": False,
        "freq": 3,
        "batch_size": 1024,
        "dropout": 0.1,
        "opt": "Adam",
        "lr_g": 0.0001,
        "ratio": 1,
        "l_dim": 25,
        "no_hidden_gen": False,
        "no_hidden": False,
        "hidden": 1024,
        "max_epochs": 3600,
        "last_clf": False,
        "name": "plsdontbebetter",
        "n_part": 30,
        "n_dim": 3,
        "heads": 5,
        "clf": True,
        "wgan_gen": False,
        "flow_prior": True,
        "load_ckpt": "/beegfs/desy/user/kaechben/pointflow_q/epoch=49-val_fpnd=182.38-val_w1m=0.0148-val_w1efp=0.000054-val_w1p=0.00501.ckpt",
    }
    if hyperopt:

        config["flow_prior"] = np.random.choice([True, False])

        config["sched"] = np.random.choice(["linear", "cosine2", None])
        config["freq"] = np.random.choice([1, 3, 5, 7])  # config["opt"]="Adam"
        config["batch_size"] = int(np.random.choice([128, 256]))  # config["opt"]="Adam"
        # config["dropout"]=np.random.choice([0.1,0.2,0.05])
        # config["class_dropout"]=2*np.random.choice([0.1,0.2,0.05])

        config["opt"] = np.random.choice(["mixed", "Adam"])  #
        config["lr_g"] = np.random.choice([0.0005, 0.0001])
        config["ratio"] = np.random.choice(
            [
                0.9,
                1,
                1.1,
            ]
        )
        config["num_layers"] = np.random.choice([4, 8])  # 5,6,7
        config["heads"] = np.random.choice([4, 6])
        config["val_check"] = 10
        config["lr_d"] = config["lr_g"] * config["ratio"]
        config["max_epochs"] = np.random.choice([1200, 2400])
        config["warmup"] = np.random.choice([0.7, 0.3]) * config["max_epochs"]
        config["name"] = "newage_" + parton

        config["coupling_layers"] = np.random.randint(low=2, high=5)
        config["network_layers_nf"] = np.random.randint(low=2, high=5)
        config["coupling_layers"] = np.random.randint(low=2, high=5)
        config["tail_bound"] = np.random.randint(low=2, high=5)

        config["lr_nf"] = 10.0 ** np.random.choice([-3, -3.5, -4, -4.5, -5])
        config["bins"] = np.random.randint(low=3, high=8)
        config["batch_size"] = int(
            np.random.choice([128, 1024, 2048])
        )  # config["opt"]="Adam"
        config["dropout"] = np.random.choice([0.15, 0.05, 0.5])
        config["network_nodes_nf"] = np.random.choice([32, 64, 128])

        config["val_check"] = 10
        print(config)
    else:
        # config["last_clf"]=True
        # config["gen_mask"]=True
        print("hyperopt off" * 100)
        config["name"] = "bestever_" + parton  # config["parton"]
        # config["freq"]=6    # config["opt"]="Adam"

    if len(sys.argv) > 2:
        root = ( "/beegfs/desy/user/"+ os.environ["USER"]+ "/"+ config["name"]+ "/"+ config["parton"]+ "_"+ "run"+ sys.argv[1]+ "_"+ str(sys.argv[2])
        )
    else:
        root = "/beegfs/desy/user/" + os.environ["USER"] + "/" + config["name"]

        # for col in cols:
        #     print('"' + col + '":' + str(config[col]))

        train(config, root=root, load_ckpt=ckpt)
