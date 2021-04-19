import torch
import wandb
import pytorch_lightning as pl

from networks.model import clf
from paths import Path_Handler
from utilities import load_config
from dataloading import MB_nohybridsDataModule


paths = Path_Handler()
path_dict = paths._dict()
config = load_config()

# Normalise epoch number to account for data splitting
n_epochs = int(
    config["train"]["n_epochs"] / (config["data"]["split"] * config["data"]["fraction"])
)

for _ in range(9):
    # Initialise wandb logger and save hyperparameters
    wandb_logger = pl.loggers.WandbLogger(
        project="Mirabest Classification",
        save_dir=path_dict["files"],
        reinit=True,
        # mode="disabled",
    )
    wandb_logger.log_hyperparams(config)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=n_epochs,
        logger=wandb_logger,
    )

    model = clf()

    # Load data and store hyperparams
    mb_nohybrids = MB_nohybridsDataModule()

    trainer.fit(model, mb_nohybrids)
