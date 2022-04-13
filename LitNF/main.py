from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os
from plotting import plotting
from lit_nf import LitNF
from jetnet_dataloader import JetNetDataloader
from helpers import *
from ray import tune
import ray
from ray.tune import CLIReporter
from pytorch_lightning.loggers import TensorBoardLogger


hyperopt = True


def train(config, data_module=None,progbar=True):
    data_module = JetNetDataloader(config)
    model = LitNF(config)

    checkpoint_callbacks = [ModelCheckpoint(monitor="logpob", filename="best_loss-{}".format(
        config["n_dim"])), ModelCheckpoint(monitor="val_w1m", filename="best_mass-{}".format(config["n_dim"]))]
    path = "/home/kaechben/JetNet_NF/LitJetNet/best_model/checkpoints/"
    ckpt = os.listdir(path)[0]
    model = model.load_from_checkpoint(path+ckpt)
    model.load_datamodule(data_module)
    model.config = config
    model.build_discriminator()
    logger = TensorBoardLogger(
        "/beegfs/desy/user/kaechben/lightninglogs"+config["name"], name="my_model")

    trainer = pl.Trainer(gpus=0, logger=logger, log_every_n_steps=10,  # auto_scale_batch_size="binsearch",
                         auto_lr_find=False, max_steps=config["max_steps"], callbacks=checkpoint_callbacks,
                         gradient_clip_val=0.5, gradient_clip_algorithm="value", check_val_every_n_epoch=10
                         )
    # # trainer.tune(model,data_module)
    # # lr_finder = trainer.tuner.lr_find(model,data_module, early_stop_threshold=100, min_lr=1e-5)
    trainer.fit(model, train_dataloaders=data_module,
                ckpt_path=checkpoint_callbacks[0].best_model_path)


if __name__ == "__main__":
        
        hyperopt = True #This sets to run a hyperparameter optimization with ray or just running the training once
        if not hyperopt:
                config = {
                "network_layers": 2,
                "network_nodes": 128,
                        "batch_size": 2000,
                "coupling_layers": 15,  # tune.uniform(3,20),#tune.randint(6,300),
                "conditional": True,
                "lr": 0.0001,
                "batchnorm": False,
                "autoreg": False,
                "bins": 8,
                "UMNN": False,
                "tail_bound": 10,
                "limit": 100000,
                "n_dim": 90,
                "dropout": 0.0,
                "lr_schedule": False,
                "gamma": 0.75,
                "n_sched": 1000,
                "canonical": False,
                "max_steps": 10,
                "lambda": 500,
                "n_turnoff": 1000,
                "n_mse_delay":50,
                "name": "debug",
                "disc": False,
                "calc_massloss":True
                }
                data_module = JetNetDataloader(config)
                train(config,data_module)
        else:
                #This sets the logging in ray
                reporter = CLIReporter(max_progress_rows=40,max_report_frequency=30, sort_by_metric=True,
                metric="logprob",parameter_columns=["network_nodes","network_layers","coupling_layers","lr"])
                reporter.add_metric_column("loss")
                reporter.add_metric_column("w1p")
                reporter.add_metric_column("w1efp")
                reporter.add_metric_column("w1m")
                config={
                "network_layers":tune.randint(2,6),
                "network_nodes":tune.randint(250,500),
                "batch_size":500,
                "coupling_layers":tune.randint(20,30),#40,#tune.uniform(3,20),#tune.randint(6,300),
                "conditional":True,#tune.choice([True,False]),
                "lr":tune.loguniform(0.005,0.00005),
                "batchnorm":False,
                "autoreg":False,
                "bins":tune.randint(4,10),
                "UMNN":False,
                "tail_bound":tune.randint(3,10), 
                "n_mse":10,
                "limit":100000,
                "n_dim":90,
                "dropout":0.0,
                "lr_schedule":False,
                "gamma":0.75,
                "n_sched":1000,
                "canonical":False,
                "max_steps":40000,
                "lambda":tune.loguniform(1,500),
                "n_mse_delay":50,
                "n_mse_turnoff":10000,
                "name":"debug",
                "disc": False,
                "calc_massloss":True
                }
                data_module = JetNetDataloader(config)
                ray.init("auto")
                result = tune.run(tune.with_parameters(
                        train,data_module=data_module),   
                        resources_per_trial=resources,
                        config=config,
                        num_samples=num_samples,
                        progress_reporter=reporter,
                        #checkpoint_freq=100,  # Checkpoint every 100 epoch
                        local_dir="/beegfs/desy/user/kaechben/ray_results/"+config["name"],
                        verbose=2,

                        )

