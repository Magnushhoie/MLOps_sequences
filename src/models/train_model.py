from pathlib import Path
import os

import hydra
import wandb
from dotenv import find_dotenv
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger

from src.path import CONF_PATH, WEIGHTS_PATH

# Automagically find path to config files
#CONF_PATH = Path(find_dotenv(), "../..", "conf").as_posix()
# CONF_PATH = Path(os.getcwd(),"conf")

@hydra.main(config_path=CONF_PATH, config_name="main")
def train(config: DictConfig) -> float:
    """Trains a model according to the provided configuration file.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Experiment configuration

    Examples
    --------
    From command line:

    $ python src/models/train_model.py experiment=test

    If you want to run a hyperparameter sweep:

    $ python src/models/train_model.py experiment=test search=optuna --multirun
    """
    # Set seed to ensure reproducibility
    deterministic = config.seed is not None
    if deterministic:
        seed_everything(config.seed, workers=True)

    # Initialize dataset (with dummy inputs/targets)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    sequence_len = 30
    embedding_size = 60

    def get_dummy_dataloader(num_samples):
        inputs = torch.randn(num_samples, sequence_len, embedding_size)
        targets = (torch.randn(num_samples, 1) > 0.5).int()
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=config.batch_size)
    
    train_dataloader = get_dummy_dataloader(400)
    val_dataloader = get_dummy_dataloader(100)
    if config.test_after_train:
        test_dataloader = get_dummy_dataloader(50)

    # Initialize logger
    logger = WandbLogger(name="optuna", project="MLOps_Sequences", log_model=True)

    # Initialize model
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Initialize trainer
    trainer: Trainer = hydra.utils.instantiate(config.training, deterministic=deterministic, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Retrieve score (required if sweeping)
    score = trainer.callback_metrics.get(config.get("objective"))

    if config.test_after_train:
        trainer.test(model, test_dataloader)

    # Finish
    wandb.finish()

    if score is not None:
        return score.item()
    return None

if __name__ == "__main__":
    train()
