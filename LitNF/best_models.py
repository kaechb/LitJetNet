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
import sys


def train(config, hyperopt=False, load_ckpt=None, i=0, root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    pl.seed_everything(42, workers=True)
    data_module = JetNetDataloader(config)  # this loads the data
    model = LitNF(
        config, hyperopt
    )  # the sets up the model,  config are hparams we want to optimize
    # Callbacks to use during the training, we  checkpoint our models
    print(model.config)
    callbacks = [
        ModelCheckpoint(
            monitor="val_logprob",
            save_top_k=50,
            filename="{epoch}-{val_logprob:.2f}-{val_w1m:.4f}-{val_w1efp:.6f}-{val_fpnd:.2f}",
            every_n_epochs=100,
        )
    ]

    # if True:#load_ckpt:
    #     model = model.load_from_checkpoint("/beegfs/desy/user/kaechben/t/2022_06_22-10_50-16/epoch=1749-val_logprob=0.94-val_w1m=0.0153.ckpt")
    model.load_datamodule(data_module)  # adds datamodule to model
    model.config = (
        config  # config are our hyperparams, we make this a class property now
    )
    logger = TensorBoardLogger(root)
    # log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    print(model.config)
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        log_every_n_steps=1000,  # auto_scale_batch_size="binsearch",
        max_steps=100000 if config["oversampling"] else config["max_steps"],
        callbacks=callbacks,
        check_val_every_n_epoch=100,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0,  # gradient_clip_val=.02, gradient_clip_algorithm="norm",
        fast_dev_run=False,
        default_root_dir=root,
        max_epochs=-1,
    )
    # This calls the fit function which trains the model
    trainer.fit(model, train_dataloaders=data_module)


if __name__ == "__main__":
    p = sys.argv[1]
    typ = sys.argv[2]
    c = sys.argv[3]
    hyperopt = False

    df = pd.read_csv(
        "/beegfs/desy/user/kaechben/bestmodels_final/top_{}{}_{}.csv".format(typ, c, p)
    ).set_index("path_index")
    print(p, typ, c)
    for index, row in df.iterrows():
        config = {
            "network_layers": int(
                row["network_layers"]
            ),  # sets amount hidden layers int(in transformation networks -scannabl)e
            "network_nodes": int(row["network_nodes"]),
            "batch_size": int(row["batch_size"]),
            "coupling_layers": int(row["coupling_layers"]),
            "lr": row["lr"],
            "batchnorm": row["batchnorm"],
            "bins": int(row["bins"]),
            "tail_bound": int(row["tail_bound"]),
            "limit": int(
                row["limit"]
            ),  # how many data points to use, test_set is 10% of this -scannable in a sense use int(10 k for faster trainin)g
            "n_dim": int(row["n_dim"]),
            "dropout": row["dropout"],
            "lr_schedule": row["lr_schedule"],
            "n_sched": row["n_sched"],
            "canonical": row["canonical"],
            "max_steps": 50000,
            "lambda": row["lambda"],
            "n_mse_turnoff": 1000000,  # int(row["n_mse_turnoff"]  ),
            "n_mse_delay": int(row["n_mse_delay"]),
            "name": row["name"],
            "disc": row["disc"],
            "calc_massloss": row["calc_massloss"],
            "context_features": int(row["context_features"]),
            "variable": row["variable"],
            "spline": row["spline"],
            "parton": row["parton"],
            "oversampling": row["oversampling"],
            "batch_size": int(row["batch_size"]),
        }
        root = "/beegfs/desy/user/kaechben/bestmodels_final/top_{}{}_{}".format(
            typ, c, p
        )
        if not hyperopt:

            train(config, hyperopt=hyperopt, root=root)
        break
