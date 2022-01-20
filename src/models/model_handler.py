import torch
from torch.nn import functional as F
from ts.torch_handler.base_handler import BaseHandler

from src.features.build_features import SequenceEmbedder


class MyHandler(BaseHandler):
    def preprocess(self, data: list) -> torch.Tensor:
        self.sequences = []

        embedder = SequenceEmbedder()
        batches = []

        for fd in data:
            fasta = fd.get("data")

            if fasta is None:
                fasta = fd.get("body")

            if isinstance(fasta, (bytes, bytearray)):
                fasta = fasta.decode("utf-8")

            with open("tmp.fasta", "w") as tmp:
                tmp.write(fasta)
            seqs, batch = embedder.embed("tmp.fasta")
            batches.append(batch)
            self.sequences.extend(seqs)

        return torch.stack(batches).view(-1, 30, 60)

    def inference(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(data)
        probs = F.sigmoid(logits)
        preds = (probs > 0.5).int()
        return probs

    def postprocess(self, probs: torch.Tensor) -> list:
        return [dict(zip(self.sequences, probs.flatten().tolist()))]
