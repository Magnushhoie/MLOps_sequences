from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.path import ROOT_PATH


class PepDataModule(LightningDataModule):
    def __init__(
        self, data_dir: str = "path/to/dir", batch_size: int = 32, n_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_workers = n_workers

    def setup(self, stage: Optional[str] = None):
        self.train = torch.load(Path(self.data_dir, "data/processed/trainset.pt"))
        self.valid = torch.load(Path(self.data_dir, "data/processed/validset.pt"))
        self.test = torch.load(Path(self.data_dir, "data/processed/testset.pt"))

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid, batch_size=self.batch_size, num_workers=self.n_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.n_workers
        )
