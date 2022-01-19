from pathlib import Path
import os

import hydra
import wandb
from dotenv import find_dotenv
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.path import ROOT_PATH, CONF_PATH, WEIGHTS_PATH

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
    
    # train_dataloader = get_dummy_dataloader(400)
    # val_dataloader = get_dummy_dataloader(100)
    train_dataset = torch.load(Path(ROOT_PATH, "data/processed/trainset.pt"))
    valid_dataset = torch.load(Path(ROOT_PATH, "data/processed/validset.pt"))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=8)

    if config.test_after_train:
        test_dataset = torch.load(Path(ROOT_PATH, "data/processed/testset.pt"))
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=8)
        # test_dataloader = get_dummy_dataloader(50)


    # Initialize logger
    logger = WandbLogger(name="optuna", project="MLOps_Sequences", log_model=True)

    # Initialize model
    model: LightningModule = hydra.utils.instantiate(config.model)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss_epoch', 
        save_on_train_epoch_end=True,
        dirpath=WEIGHTS_PATH,
        filename='cnn_model'
    )

    # Initialize trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.training, 
        deterministic=deterministic, 
        logger=logger, 
        weights_save_path=WEIGHTS_PATH
        # callbacks=[checkpoint_callback]
        )

    trainer.fit(model, train_dataloader, valid_dataloader)

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
