import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from src.features.build_features import SequenceEmbedder
from src.path import ROOT_PATH, checkpoint_path

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

    $ python src/models/predict_model.py experiment=test predict.input_file=test.fasta
    """
    # CURRENTLY ONLY TAKES FILES SOMEWHERE IN THE ROOT FOLDER, HAS TO CHANGE
    input_file = ROOT_PATH / Path(config.predict.input_file)

    # Build features
    embedder = SequenceEmbedder()
    sequences, batch = embedder.embed(input_file)

    # Initialize model
    model = hydra.utils.instantiate(config.model)

    # # Download model checkpoint
    # run = wandb.init()
    # # CKPT_PATH = Path(os.get.cwd)
    # artifact = run.use_artifact(config.predict.artifact_dir)
    # artifact_dir = artifact.download()

    # # Load model checkpoint
    # model = model.load_from_checkpoint(Path(artifact_dir, "model.ckpt"))
    model = model.load_from_checkpoint(
        checkpoint_path(project="MLOps_Sequences", experiment=config.name)
    )
    # Make sure weights do not get updated
    model.eval()
    model.freeze()

    # Predict
    with torch.no_grad():
        logits = model(batch)

    probs = logits.exp()
    preds = (probs > 0.5).flatten().tolist()

    # Print results
    results = pd.DataFrame(dict(
        sequences=sequences,
        is_antimicrobial=preds
    ))
    output_file = input_file.parent / f"predictions_{input_file.name}"
    results.to_csv(output_file, index=False)
    print(f"Predictions saved at: {input_file}")


if __name__ == "__main__":
    predict()
