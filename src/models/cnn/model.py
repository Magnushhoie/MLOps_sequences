__all__ = ["PerResidueModel"]

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import ConfusionMatrix

from src.models.cnn.layers import ConvBlock, Transpose


class PerResidueModel(LightningModule):
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
        self.training_confmat = ConfusionMatrix(num_classes=2)
        self.validation_confmat = ConfusionMatrix(num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), weight_decay=self.weight_decay, lr=self.lr
        )

    def step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        """Computes loss and updates the confusion matrix at every step.

        Parameters
        ----------
        batch : torch.Tensor
        mode : {'training', 'validation'}

        Returns
        -------
        loss : torch.Tensor
        """
        # Do forward pass
        x, y = batch
        logits = self(x)

        # Calculate loss and predictions
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, y.float())  # Must be float
        preds = (logits > 0.5).int()  # Must be int

        # Update training/validation confusion matrix with step's values
        getattr(self, f"{mode}_confmat")(preds, y)

        # Log loss at every step
        self.log(f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def epoch_end(self, mode: str) -> None:
        """Computes and logs confusion matrix at end of every epoch.

        Parameters
        ----------
        mode : {'training', 'validation'}
        """
        # Compute confusion matrix from epoch's outputs
        confmat = getattr(self, f"{mode}_confmat")
        tn, fp, fn, tp = confmat.compute().view(-1)

        # Compute accuracy, recall, precision and F1 score
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)

        # Log metrics
        self.log(f"{mode}_accuracy", accuracy)
        self.log(f"{mode}_recall", recall)
        self.log(f"{mode}_precision", precision)
        self.log(f"{mode}_f1", f1)

        # Reset confusion matrix so it is recalculated for next epoch
        confmat.reset()

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.step(batch, "training")

    def training_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("training")

    def validation_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.step(batch, "validation")

    def validation_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("validation")
