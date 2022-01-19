from pathlib import Path
import os

import hydra
import wandb
from dotenv import find_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.path import checkpoint_path

# Automagically find path to config files
# CONF_PATH = Path(find_dotenv(), "../..", "conf").as_posix()
CONF_PATH = Path(os.getcwd(), "conf")

@hydra.main(config_path=CONF_PATH, config_name="main")
def predict(config: DictConfig):
    """Predicts using a trained model provided configuration file.

    Parameters
    ----------
    config : omegaconf.DictConfig
        Experiment configuration
        The config file should contain an input and an output 
        file paths

    Examples
    --------
    From command line:

    $ python src/models/predict_model.py experiment=test
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
    
    predict_dataloader= get_dummy_dataloader(100)
    
    # Initialize logger
    logger = WandbLogger(name=config.name, project="MLOps_Sequences", log_model=False)

    # Initialize model
    model = hydra.utils.instantiate(config.model)

    # # Download model checkpoint
    # run = wandb.init()
    # # CKPT_PATH = Path(os.get.cwd)
    # artifact = run.use_artifact(config.predict.artifact_dir) 
    # artifact_dir = artifact.download()

    # # Load model checkpoint
    # model = model.load_from_checkpoint(Path(artifact_dir, "model.ckpt"))
    model = model.load_from_checkpoint(checkpoint_path(project="MLOps_Sequences", experiment="mlops_peps"))


    trainer = hydra.utils.instantiate(config.training, deterministic=deterministic, logger=logger)
    trainer.test(model, predict_dataloader)

    # Finish
    wandb.finish()

if __name__ == "__main__":
    predict()
