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
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
def train(config, hyperopt=False, load_ckpt=None):
    # This function is a wrapper for the hyperparameter optimization module called ray 
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    data_module = JetNetDataloader(config) #this loads the data
    model = LitNF(config) # the sets up the model,  config are hparams we want to optimize
    # Callbacks to use during the training, we automatically checkpoint our models
    callbacks = [ModelCheckpoint(monitor="val_logprob", filename="best_loss"), 
                ModelCheckpoint(monitor="val_w1m", filename="best_mass"),
                EarlyStopping(monitor="w1m", min_delta=0.0002, patience=30, verbose=False, mode="min")]
    if hyperopt:
        metrics = {"val_w1m": "val_w1m", "val_w1p": "val_w1p", "val_logprob":"val_logprob"}
        callbacks = [TuneReportCheckpointCallback(
            metrics, on="validation_end")]#tune is the hyperparameter optimization library we use
    if load_ckpt:
        model = model.load_from_checkpoint(load_ckpt)
    model.load_datamodule(data_module)#adds datamodule to model
    model.config = config #config are our hyperparams, we make this a class property now

    if hyperopt: #Tune gives us an automatic logger for tensorboard
        logger = TensorBoardLogger(tune.get_trial_dir())

    else:
        logger = TensorBoardLogger(
            "/beegfs/desy/user/"+os.environ["USER"]+"/"+config["name"])
    

    callbacks.append(TuneReportCallback(

        {
            "logprob": "logprob",
            "w1m": "w1m",
            "w1m": "w1p",
            "w1m": "w1efp"
        },
        on="validation_end"))
    #log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    trainer = pl.Trainer(gpus=1, logger=logger, enable_progress_bar=not hyperopt, log_every_n_steps=10,  # auto_scale_batch_size="binsearch",
                          max_steps=config["max_steps"], callbacks=callbacks,
                         gradient_clip_val=0.5, gradient_clip_algorithm="value", check_val_every_n_epoch=100
                         )
    # This calls the fit function which trains the model
    trainer.fit(model, train_dataloaders=data_module,
                )  


if __name__ == "__main__":

    hyperopt = True  # This sets to run a hyperparameter optimization with ray or just running the training once
    if not hyperopt:
        config = {
            "network_layers": 2,  # sets amount hidden layers in transformation networks
            "network_nodes": 256,  # amount nodes in hidden layers in transformation networks
            "batch_size": 2000,  # sets batch size
            "coupling_layers": 15,  # amount of invertible transformations to use
            "conditional": True,  # whether to condition on mass or not
            "lr": 0.0001,  # sets learning rate
            "batchnorm": False,  # use batchnorm or not
            "autoreg": False,  # whether to use autoregressive transform, alot slower together with calc_massloss or massloss
            "bins": 8,  # amount of bins to use in rational quadratic splines
            "UMNN": False,  # whether to use monotonic network instead of splines in trafo
            "tail_bound": 10,  # splines:max value that is transformed, over this value theree is id
            "limit": 100000,  # how many data points to use, test_set is 10% of this
            "n_dim": 90,  # how many dimensions to use or equivalently /3 gives the amount of particles to use
            "dropout": 0.0,  # use droput proportion, for 0 there is no dropout
            "lr_schedule": False,  # whether tos chedule the learning rate
            "gamma": 0.75,  # gamma that is annealing the learning rate
            "n_sched": 1000,  # how many steps between an annealing step
            "canonical": False,  # transform data coordinates to px,py,pz
            "max_steps": 10000,  # how many steps to use at max
            "lambda": 500,  # balance between massloss and nll
            "n_mse_turnoff": 1000,  # when to turn off mass loss
            "n_mse_delay": 50,  # when to turn on mass loss
            "name": "debug",  # name for logging folder
            "disc": False,  # whether to train gan style discriminator that decides whether point is simulated or generated
            # whether to calculate mass loss, makes training slower, do not use with autoregressive!
            "calc_massloss": True, 
            "context_features":2, #amount of variables used for conditioning
            "particle_net_cond":False,#use particle net for conditioning (not implemented)
            "particle_net":False, #use particle net (not implemented)
             "variable":True, #use variable amount of particles
        }
        data_module = JetNetDataloader(config)
        train(config, data_module)
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

        config = {
            "network_layers": tune.randint(2, 6),
            "network_nodes": tune.randint(250, 500),
            "batch_size": 2000,
            # 40,#tune.uniform(3,20),#tune.randint(6,300),
            "coupling_layers": tune.randint(5, 20),
            "conditional": True,  # tune.choice([True,False]),
            "lr": tune.loguniform(0.001, 0.00005),
            "batchnorm": tune.choice([True,False]),
            "bins": tune.randint(4, 10),
            "UMNN": False,
            "tail_bound": tune.randint(3, 10),
            "limit": 100000,
            "n_dim": 90,
            "dropout": tune.uniform(0.0,0.5),
            "lr_schedule": False,
            "gamma": 0.75,
            "n_sched": 1000,
            "canonical": False,
            "max_steps": 40000,
            "lambda": tune.loguniform(1, 500),
            "n_mse_delay": 500,
            "n_mse_turnoff": 10000,
            "name": "testing_n_particles_q",
            "calc_massloss": True,
            "context_features":2,
            "variable":True,
        }
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
