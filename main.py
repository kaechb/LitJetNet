from pytorch_lightning.callbacks import ModelCheckpoint
# from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import os
from plotting import *
from lit_nf import *
from jetnet_dataloader import *
from helpers import *
from ray import tune
import ray
from ray.tune import CLIReporter
from pytorch_lightning.loggers import TensorBoardLogger


hyperopt=True

def train(config,data_module=None):
        print("model initial")
        model=LitNF(config)
        checkpoint_callbacks = [ModelCheckpoint(monitor="train_loss",filename="best_loss-{}".format(config["n_dim"]))
                                ,ModelCheckpoint(monitor="val_w1m",filename="best_mass-{}".format(config["n_dim"]))]
        print("ok")
        checkpoint=False
        if checkpoint:
                path="lightning_logs/version_200/checkpoints/"
                ckpt=os.listdir(path)[0]
                model=model.load_from_checkpoint(path+ckpt)
                
                model.config=config
        print("ok until trainer")
        logger = TensorBoardLogger("/beegfs/desy/user/kaechben/lightninglogs"+config["name"], name="my_model")

        trainer = pl.Trainer(gpus=1,#auto_scale_batch_size="binsearch",logger
                        logger=logger,
                        auto_lr_find=False,max_steps=config["max_steps"],callbacks=checkpoint_callbacks,
                        gradient_clip_val=0.5, gradient_clip_algorithm="value",check_val_every_n_epoch=1,  
                        progress_bar_refresh_rate=0,log_every_n_steps=10)
        model.load_datamodule(data_module)
        # trainer.tune(model,data_module)
        # lr_finder = trainer.tuner.lr_find(model,data_module, early_stop_threshold=100, min_lr=1e-5)
        trainer.fit(model,train_dataloaders=data_module, ckpt_path=checkpoint_callbacks[0].best_model_path)
        
if __name__=="__main__":
        num_samples=500
        resources={"cpu":10 , "gpu": 0.3}
        if not hyperopt:
                config={
                "network_layers":2,
                "network_nodes":128,
                "batch_size":2500,
                "coupling_layers":30,#tune.uniform(3,20),#tune.randint(6,300),
                "conditional":True,
                "lr":0.001,
                "batchnorm":False,
                "autoreg":False,
                "bins":8,
                "UMNN":False,
                "tail_bound":10,
                "n_mse":1000,
                "limit":100000,
                "n_dim":90,
                "dropout":0.0,
                "lr_schedule":False,
                "gamma":0.75,
                "n_sched":1000,
                "canonical":False,
                "max_steps":40000,
                "lambda":500,
                "n_turnoff":10000
                }

        else:
                reporter = CLIReporter(max_progress_rows=40,max_report_frequency=30, sort_by_metric=True,
                metric="logprob",parameter_columns=["network_nodes","network_layers","coupling_layers","lr"])
                        # Add a custom metric column, in addition to the default metrics.
                # Note that this must be a metric that is returned in your training results.
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
                "n_turnoff":10000,
                "name":"debug"
                }
        data_module = JetNetDataloader(config)
        print("should start")
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