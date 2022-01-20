from os import PathLike

import torch
from torch.nn import functional as F
from ts.torch_handler.base_handler import BaseHandler

from src.features.build_features import SequenceEmbedder


class MyHandler(BaseHandler):
    def preprocess(self, data: PathLike) -> torch.Tensor:
        embedder = SequenceEmbedder()
        sequences, batch = embedder.embed(data)
        self.sequences = sequences
        return batch

    def inference(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        logits = self.model(data)
        probs = F.sigmoid(logits)
        preds = (probs > 0.5).int()
        return preds

    def postprocess(self, preds: torch.Tensor) -> list:
        return preds.tolist()
