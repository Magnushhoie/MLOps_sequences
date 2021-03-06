__all__ = ["PerResidueModel"]

import torch
from torch import nn

from src.models.base.model import PredictionModel
from src.models.cnn.layers import ConvBlock, Transpose


class PerResidueModel(PredictionModel):
    def __init__(self, lr: float, weight_decay: float):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.Sequential(
            Transpose(),
            nn.BatchNorm1d(60),
            *[ConvBlock() for _ in range(5)],
            Transpose(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(1),
            nn.Flatten(),
            nn.Linear(30, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
