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
def train(config, hyperopt=False, load_ckpt=None):
    # This function is a wrapper for the hyperparameter optimization module called ray 
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    data_module = JetNetDataloader(config) #this loads the data
    model = LitNF(config) # the sets up the model,  config are hparams we want to optimize
    # Callbacks to use during the training, we  checkpoint our models
    callbacks = [ModelCheckpoint(monitor="val_logprob", filename='{epoch}-{val_logprob:.2f}-{val_w1m:.4f}', dirpath="/beegfs/desy/user/"+os.environ["USER"]+"/"+config["name"]) ]
    if load_ckpt:
        model = model.load_from_checkpoint(load_ckpt)
    model.load_datamodule(data_module)#adds datamodule to model
    model.config = config #config are our hyperparams, we make this a class property now

    if hyperopt: #Tune gives us an automatic logger for tensorboard
        metrics = {"val_w1m": "val_w1m", "val_w1p": "val_w1p", "val_logprob":"val_logprob"}
        callbacks = [TuneReportCheckpointCallback(
            metrics, on="validation_end")]#tune is the hyperparameter optimization library we use
        logger = TensorBoardLogger(tune.get_trial_dir())
        callbacks.append(TuneReportCallback(
        {
            "logprob": "logprob",
            "w1m": "w1m",
            "w1m": "w1p",
            "w1m": "w1efp"
        },
        on="validation_end"))
    else:

        logger = TensorBoardLogger(
            "/beegfs/desy/user/"+os.environ["USER"]+"/"+config["name"])
    
    #log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    
    trainer = pl.Trainer(gpus=1, logger=logger,  log_every_n_steps=10,  # auto_scale_batch_size="binsearch",
                          max_steps=config["max_steps"], callbacks=callbacks, #progress_bar_refresh_rate=0,
                          check_val_every_n_epoch=10,track_grad_norm=2,num_sanity_val_steps=50,#gradient_clip_val=.02, gradient_clip_algorithm="norm",
                         fast_dev_run=False)
    # This calls the fit function which trains the model
    trainer.fit(model, train_dataloaders=data_module  )  


if __name__ == "__main__":

    hyperopt = False  # This sets to run a hyperparameter optimization with ray or just running the training once
    if not hyperopt:
        config = {
            "network_layers": 2,  # sets amount hidden layers in transformation networks
            "network_nodes": 256,  # amount nodes in hidden layers in transformation networks
            "batch_size": 7500,  # sets batch size
            "coupling_layers": 10,  # amount of invertible transformations to use
            "lr": 0.001,  # sets learning rate
            "batchnorm": False,  # use batchnorm or not
            "bins": 8,  # amount of bins to use in rational quadratic splines
            "UMNN": False,  # whether to use monotonic network instead of splines in trafo
            "tail_bound": 6,  # splines:max value that is transformed, over this value theree is id
            "limit": 100000,  # how many data points to use, test_set is 10% of this
            "n_dim": 90,  # how many dimensions to use or equivalently /3 gives the amount of particles to use
            "dropout": 0.4,  # use droput proportion, for 0 there is no dropout
            "lr_schedule": False,  # whether tos chedule the learning rate
            "gamma": 0.75,  # gamma that is annealing the learning rate
            "n_sched": 1000,  # how many steps between an annealing step
            "canonical": False,  # transform data coordinates to px,py,pz
            "max_steps": 10000,  # how many steps to use at max
            "lambda": 10,  # balance between massloss and nll
            "n_mse_turnoff": 1000,  # when to turn off mass loss
            "n_mse_delay": 5,  # when to turn on mass loss
            "name": "pointflow",  # name for logging folder
            "disc": False,  # whether to train gan style discriminator that decides whether point is simulated or generated
            "calc_massloss": False, # whether to calculate mass loss, makes training slower, do not use with autoregressive!
            "context_features":2, #amount of variables used for conditioning, for 0 no conditioning is used, for 1 only the mass is used, for 2 also the number part is used
            "variable":1, #use variable amount of particles otherwise only use 30, options are true or false 
            "spline":True,#whether to use splines or not, can also be set to "autoregressive" but they are unstable
            "parton":"q" #choose the dataset you want to train options: t for top,q for quark,g for gluon
        }
        if config["variable"] and config["context_features"]<2:
            raise
        data_module = JetNetDataloader(config)
        train(config,hyperopt=hyperopt)
    else:
        num_samples = 100 # how many hparam settings to sample with ray
        resources = {"cpu": 10, "gpu": 0.5}
        # This sets the logging in ray
        reporter = CLIReporter(max_progress_rows=40, max_report_frequency=300, sort_by_metric=True,
                               metric="logprob", parameter_columns=["network_nodes", "network_layers", "coupling_layers", "lr"])
        reporter.add_metric_column("m_loss")
        reporter.add_metric_column("logprob")
        reporter.add_metric_column("val_w1p")
        reporter.add_metric_column("val_w1efp")
        reporter.add_metric_column("val_w1m")

        config["network_layers"]=tune.randint(2, 6),
        config["network_nodes"]= tune.randint(250, 500),
        config["coupling_layers"]= tune.randint(5, 20),
        config["lr"]= tune.loguniform(0.001, 0.00005),
        config["batchnorm"]= tune.choice([True,False]),
        config["bins"]= tune.randint(4, 10),
        config["tail_bound"]= tune.randint(3, 10),
        config["dropout"]= tune.uniform(0.0,0.5),
        config["lambda"]= tune.loguniform(1, 500),
        
        data_module = JetNetDataloader(config)
        ray.init("auto") # This connects to the main ray node
        #this starts the training 
        result = tune.run(tune.with_parameters(
            train,  hyperopt=True),
            resources_per_trial=resources,
            config=config,
            num_samples=num_samples,
            progress_reporter=reporter,
            # checkpoint_freq=100,  # Checkpoint every 100 epoch
            local_dir="/beegfs/desy/user/"+os.environ["USER"]+"/ray_results/" + \
            config["name"]+"/",
            verbose=2,
        )
