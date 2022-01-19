from pathlib import Path
import os

import hydra
import wandb
from dotenv import find_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.path import CONF_PATH, WEIGHTS_PATH

# Automagically find path to config files
#CONF_PATH = Path(find_dotenv(), "../..", "conf").as_posix()
# CONF_PATH = Path(os.getcwd(),"conf")

@hydra.main(config_path=CONF_PATH, config_name="main")
def train(config: DictConfig):
    """Trains a model according to the provided configuration file.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Experiment configuration

    Examples
    --------
    From command line:

    $ python src/models/train_model.py experiment=test
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
        return DataLoader(dataset, batch_size=32, num_workers=5)
    
    train_dataloader = get_dummy_dataloader(400)
    val_dataloader = get_dummy_dataloader(100)
    if config.test_after_train:
        test_dataloader = get_dummy_dataloader(50)

    # Initialize logger
    logger = WandbLogger(name=config.name, project="MLOps_Sequences", id=config.name, log_model=True)

    # Initialize model
    model: LightningModule = hydra.utils.instantiate(config.model)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss', 
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

    trainer.fit(model, train_dataloader, val_dataloader)

    # Make scripted version of the model
    script_model = torch.jit.script(model)
    script_model.save(Path(WEIGHTS_PATH,'deployable_cnn.pt'))

    # Test the model on an external test set
    if config.test_after_train:
        trainer.test(model, test_dataloader)

    # Finish
    wandb.finish()

if __name__ == "__main__":
    train()
