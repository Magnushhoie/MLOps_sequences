__all__ = ["FixedLengthModel"]

import torch
from torch import nn

from src.models.base.model import PredictionModel


class FixedLengthModel(PredictionModel):
    def __init__(self, num_hidden: int, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.Sequential(
            nn.LazyLinear(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average the embedding over the full sequence length
        mean = x.mean(dim=-1)
        return self.layers(mean)
