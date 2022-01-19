__all__ = ["checkpoint_path", "CONF_PATH", "WEIGHTS_PATH"]

import os
from pathlib import Path

from dotenv import find_dotenv

#ROOT_PATH = Path(find_dotenv()).parent
ROOT_PATH = os.getcwd()

CONF_PATH = Path(ROOT_PATH, "conf")
WEIGHTS_PATH = Path(ROOT_PATH, "models")


def checkpoint_path(project: str, experiment: str) -> Path:
    try:
        return next(WEIGHTS_PATH.joinpath(project, experiment).glob("**/*.ckpt"))
    except StopIteration:
        raise FileNotFoundError(f"Checkpoint for experiment ´{experiment}´ not found")