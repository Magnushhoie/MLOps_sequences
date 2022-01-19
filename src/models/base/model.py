__all__ = ["PredictionModel"]

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import ConfusionMatrix, AUROC, MatthewsCorrCoef


class PredictionModel(LightningModule):
    """Base class for prediction models, defines logic at each training step and
    epoch."""
    def __init__(self):
        super().__init__()
        self.training_confmat = ConfusionMatrix(num_classes=2)
        self.validation_confmat = ConfusionMatrix(num_classes=2)
        self.test_confmat = ConfusionMatrix(num_classes=2)

        self.training_auroc = AUROC(num_classes=2)
        self.validation_auroc = AUROC(num_classes=2)
        self.test_auroc = AUROC(num_classes=2)

        self.training_mcc = MatthewsCorrCoef(num_classes=2)
        self.validation_mcc = MatthewsCorrCoef(num_classes=2)
        self.test_mcc = MatthewsCorrCoef(num_classes=2)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr
        )

    def step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        """Computes loss and updates the confusion matrix at every step.

        Parameters
        ----------
        batch : torch.Tensor
        mode : {'training', 'validation', 'test'}

        Returns
        -------
        loss : torch.Tensor
        """
        # Do forward pass
        x, y = batch
        # print(x.shape)
        logits = self(x)

        # Calculate loss and predictions
        criterion = nn.BCEWithLogitsLoss()
        y = y.unsqueeze(1)
        loss = criterion(logits, y.float())  # Must be float
        preds = (logits > 0.5).int()  # Must be int

        # Update confusion matrix with step's values
        getattr(self, f"{mode}_confmat")(preds, y)

        # Update AUROC
        getattr(self, f"{mode}_auroc")(logits, y)

        # Update MCC
        getattr(self, f"{mode}_mcc")(preds, y)

        # Log loss at every step
        self.log(
            f"{mode}_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def epoch_end(self, mode: str) -> None:
        """Computes and logs confusion matrix, AUC, MCC at end of every epoch.

        Parameters
        ----------
        mode : {'training', 'validation', 'test'}
        """
        # Compute confusion matrix from epoch's outputs
        confmat = getattr(self, f"{mode}_confmat")
        tn, fp, fn, tp = confmat.compute().view(-1)

        auroc = getattr(self, f"{mode}_auroc")
        auroc_value = auroc.compute()

        mcc = getattr(self, f"{mode}_mcc")
        mcc_value = mcc.compute()

        # Compute accuracy, recall, precision and F1 score
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        # Log metrics
        self.log(f"{mode}_accuracy", accuracy)
        self.log(f"{mode}_recall", recall)
        self.log(f"{mode}_precision", precision)
        self.log(f"{mode}_auroc", auroc_value)
        self.log(f"{mode}_mcc", mcc_value)

        # Reset confusion matrix so it is recalculated for next epoch
        confmat.reset()
        auroc.reset()
        mcc.reset()

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "training")

    def training_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("training")

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "validation")

    def validation_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("validation")

    def test_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        return self.step(batch, "test")

    def test_epoch_end(self, outputs: torch.Tensor) -> None:
        self.epoch_end("test")

    


    # def test_step(self, batch: torch.Tensor, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     criterion = nn.BCEWithLogitsLoss()
    #     y = y.unsqueeze(1)
    #     loss = criterion(logits, y.float())
    #     preds = (logits > 0.5).int()
    #     self.log("test_loss", loss)
        
    #     mode='test'
    #     confmat = getattr(self, f"{mode}_confmat")(preds, y)
    #     tn, fp, fn, tp = confmat.view(-1)

    #     # Compute accuracy, recall, precision and F1 score
    #     accuracy = (tn + tp) / (tn + fp + fn + tp)
    #     recall = tp / (tp + fn)
    #     precision = tp / (tp + fp)

    #     # Log metrics
    #     self.log(f"{mode}_accuracy", accuracy, on_step=True, on_epoch=False)
    #     self.log(f"{mode}_recall", recall, on_step=True, on_epoch=False)
    #     self.log(f"{mode}_precision", precision, on_step=True, on_epoch=False)




