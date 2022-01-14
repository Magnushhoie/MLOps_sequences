# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
from pathlib import Path

import click
import subprocess
from dotenv import find_dotenv, load_dotenv

import argparse, os, glob
from collections import OrderedDict

import numpy as np
from sklearn.decomposition import PCA

from Bio import SeqIO

import torch

# Params
interimDir = "data/interim/"
procDir = "data/processed/"
dirType_list = ["test_pos/", "test_neg/", "train_pos/", "train_neg/"]
fasta_list = ["test_pos.fasta", "test_neg.fasta", "train_pos.fasta", "train_neg.fasta"]
pca_dim=60

def stack_res_tensors(tensors):
    embedding_dim = tensors[0].shape[-1]

    # Prepare to stack tensors
    n_seqs = len(tensors)
    seq_max_length = max([t.shape[0] for t in tensors])

    # Initialize empty padded vector, with 0-padding for sequences with less than max length residues
    fill_tensor = torch.zeros(size=(n_seqs, seq_max_length, embedding_dim))
    
    # Load torch tensors from ESM embeddings matching sequence, fill padded tensor
    for i, tensor in enumerate(tensors):
        fill_tensor[i, 0:tensor.shape[0]] = tensor

    return(fill_tensor)

def get_PCA_fit_train():
    """ Returns fitted PCA object on training dataset """

    def PCA_fit(stacked_tensor, pca_dim=60):
        """ Returns fitted PCA on flattened positions (N x L x D) -> PCA(N x LD) """

        # Flatten training dataset to shape n_positions/residues x embedding_dim
        embedding_dim = stacked_tensor.shape[-1]
        positions = stacked_tensor.view( stacked_tensor.shape[0]*stacked_tensor.shape[1], embedding_dim )

        pca = PCA(pca_dim)
        
        return(pca.fit(positions))
        
    id_tensor_odict = OrderedDict()
    tensor_files = glob.glob(interimDir + "train_pos/*.pt") + glob.glob(interimDir + "train_neg/*.pt")

    for file in tensor_files:
        tensor_dict = torch.load(file)
        id_tensor_odict[tensor_dict["label"]] = tensor_dict["representations"][6]

    # PCA fit
    tensors = [tensor for tensor in id_tensor_odict.values()]
    stacked_tensor = stack_res_tensors(tensors)

    embed_dim = stacked_tensor.shape[-1]
    max_len = len(stacked_tensor[0])

    print(f"Fitting PCA on train dataset {stacked_tensor.shape}")
    print(f"{embed_dim} -> {pca_dim}")
    pca = PCA_fit(stacked_tensor, pca_dim)

    return pca, embed_dim, max_len

def PCA_transformed_save_embed_files(pca, embed_dim, max_len, dirType_list, fasta_list, interimDir, procDir):
    print(f"PCA transforming and saving embedding pt files in {dirType_list}")
    # PCA transform
    id_tensor_odict = OrderedDict()

    for dirType in dirType_list:
        os.makedirs(procDir + dirType, exist_ok=True)

    for fasta, dirType in zip(fasta_list, dirType_list):

        # Load FASTA file for ids
        record_dict = SeqIO.to_dict(SeqIO.parse(interimDir + fasta, "fasta"))

        # Load, transform and tensor files
        tensor_files = glob.glob(interimDir + dirType + "*.pt")

        for file in tensor_files:
            name = os.path.basename(file)

            t_dict = torch.load(file)
            id = t_dict["label"]

            # PCA transform tensor
            tensor = t_dict["representations"][6]
            
            arr = pca.transform(tensor)
            padded_arr = np.zeros( (max_len, pca_dim) )
            padded_arr[ :len(arr) ] = arr
            pca_tensor = torch.from_numpy(padded_arr).type(torch.float32)

            value_dict = {
                        "tensor": pca_tensor,
                        "sequence": str(record_dict[id].seq),
                        "length": len(arr)
                        }

            outFile = procDir + dirType + name
            torch.save(value_dict, outFile)


@click.command()
@click.argument('interim_filepath', default='data/interim/', type=click.Path())
@click.argument('output_filepath', default='data/processed/', type=click.Path())
def main(interim_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Get fitted PCA object
    pca_train, embed_dim, max_len = get_PCA_fit_train()
    torch.save(pca_train, procDir + "pca_train.pt")

    # PCA transform and save embedding files
    PCA_transformed_save_embed_files(pca_train, embed_dim, max_len, dirType_list, fasta_list, interimDir, procDir)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
