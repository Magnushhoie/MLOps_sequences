from src.models.cnn import PerResidueModel
from src.path import checkpoint_path
import torch

CKPT_PATH = checkpoint_path(project="MLOps_Sequences", experiment="mlops_antimb")

model = PerResidueModel.load_from_checkpoint(CKPT_PATH)

script_model = torch.jit.script(model)
script_model.save(CKPT_PATH.parents[1]/"pretrained_cnn.pt")

