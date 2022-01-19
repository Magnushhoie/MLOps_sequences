__all__ = ["PredictionModel"]

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import ConfusionMatrix


class PredictionModel(LightningModule):
    """Base class for prediction models, defines logic at each training step and
    epoch."""

    def __init__(self):
        super().__init__()
        self.training_confmat = ConfusionMatrix(num_classes=2)
        self.validation_confmat = ConfusionMatrix(num_classes=2)
        self.test_confmat = ConfusionMatrix(num_classes=2)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr,
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
        mode : {'training', 'validation''}
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

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "training")

    def training_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("training")

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "validation")

    def validation_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("validation")

    def test_step(self, batch: torch.Tensor, batch_idx):
        x, y = batch
        logits = self(x)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, y.float())
        preds = (logits > 0.5).int()
        self.log("test_loss", loss)

        mode = "test"
        confmat = getattr(self, f"{mode}_confmat")(preds, y)
        tn, fp, fn, tp = confmat.view(-1)

        # Compute accuracy, recall, precision and F1 score
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)

        # Log metrics
        self.log(f"{mode}_accuracy", accuracy, on_step=True, on_epoch=False)
        self.log(f"{mode}_recall", recall, on_step=True, on_epoch=False)
        self.log(f"{mode}_precision", precision, on_step=True, on_epoch=False)
        self.log(f"{mode}_f1", f1, on_step=True, on_epoch=False)
