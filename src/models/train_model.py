from pathlib import Path

import hydra
from dotenv import find_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

# Automagically find path to config files
CONF_PATH = Path(find_dotenv(), "..", "conf").as_posix()


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

    $ python src/models/train_model.py experiment=test.yaml
    """
    # Set seed to ensure reproducibility
    deterministic = config.seed is not None
    if deterministic:
        seed_everything(config.seed, workers=True)

    # Initialize model
    model = hydra.utils.instantiate(config.model)

    # Initialize trainer
    trainer = hydra.utils.instantiate(config.training, deterministic=deterministic)

    # Check if model parameters were loaded correctly
    print(
        f"Model has:\n\tlr={model.hparams.lr}\n\tweight_decay={model.hparams.weight_decay}"
    )


if __name__ == "__main__":
    train()
