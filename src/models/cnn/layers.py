__all__ = ["ConvBlock", "Transpose"]

import torch
from torch import nn


class Transpose(nn.Module):
    """Transposes the incoming data without affecting batch size."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, -1)


class ConvBlock(nn.Module):
    """Applies a 1D convolution, batch normalization, leaky ReLU activation, max
    pooling, and dropout operations to the incoming data."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=2, padding=1),
            nn.BatchNorm1d(num_features=60),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.55),
        )
        nn.init.kaiming_uniform_(self.layers[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
