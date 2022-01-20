__all__ = ["SequenceEmbedder"]

from os import PathLike
from typing import List, Tuple

import esm
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.path import ROOT_PATH

PCA_PATH = ROOT_PATH / "data" / "processed" / "pca_train.pt"


class SequenceEmbedder:
    def __init__(self, max_sequence_len: int = 30, embedding_size: int = 60) -> None:
        if not PCA_PATH.exists():
            raise FileNotFoundError("Trained PCA does not exist!")
        self.trained_pca: PCA = torch.load(PCA_PATH)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            "esm1_t6_43M_UR50S"
        )
        self.model.eval()
        self.repr_layer = self.model.num_layers % (self.model.num_layers + 1)
        self.max_sequence_len = max_sequence_len
        self.embedding_size = embedding_size

    @torch.no_grad()
    def embed(self, fasta_file: PathLike) -> Tuple[List[str], torch.Tensor]:
        """Embeds the sequences listed in the input file, returning a list of strings and a batch of N×L×E dimensions,
        where N is the number of sequences in FASTA, L is the maximum sequence length, and E is the embedding size.

        Parameters
        ----------
        fasta_file : os.PathLike

        Returns
        -------
        Tuple[List[str], torch.Tensor]
        """
        # Generate dataset from FASTA file
        dataset = esm.FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(toks_per_batch=4096, extra_toks_per_seq=1)
        dataloader = DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(),
            batch_sampler=batches,
        )
        x = {}
        for labels, sequences, toks in dataloader:
            # Truncate
            toks = toks[:, :1022]
            # Generate representations
            out = self.model(toks, repr_layers=[self.repr_layer], return_contacts=False)
            representations = out["representations"][self.repr_layer]
            for i, sequence in enumerate(sequences):
                sequence_len = len(sequence)
                representation = representations[i, 1 : sequence_len + 1]
                embedding = torch.zeros((self.max_sequence_len, self.embedding_size))
                embedding[:sequence_len, :] = torch.Tensor(
                    self.trained_pca.transform(representation)
                )
                x[sequence] = embedding
        batch = torch.stack(tuple(x.values()))
        return list(x.keys()), batch
