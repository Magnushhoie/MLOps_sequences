# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
from pathlib import Path
import logging

import pandas as pd

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
val_frac = 0.10
verbose = True

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

def get_PCA_fit_train(pca_dim=60):
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

    log.info(f"Fitting PCA on train dataset {stacked_tensor.shape}")
    log.info(f"{embed_dim} -> {pca_dim}")
    pca = PCA_fit(stacked_tensor, pca_dim)

    return pca, embed_dim, max_len

def PCA_transformed_save_embed_files(pca, embed_dim, max_len, dirType_list, fasta_list, interimDir, procDir):
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

def load_tensor_folder(ds_str, label_int, dataDir):
    """ 
    Loads dict .pt files with embedded sequences
    Returns X and y tensor and dataframe with sequence info
    """

    file_list = glob.glob(dataDir + ds_str + "/*.pt")

    tensor_list = []
    df = pd.DataFrame(columns=["id", "dataset", "label", "length", "sequence"])

    for i, file in enumerate(file_list):
        td = torch.load(file)
        tensor_list.append(td["tensor"])

        id_str = Path(os.path.basename(file)).stem
        df.loc[i] = (id_str, ds_str, label_int, td["length"], td["sequence"])
    
    X_tensor = torch.stack(tensor_list)
    y_tensor = torch.zeros(len(tensor_list))
    y_tensor[:] = label_int

    return X_tensor, y_tensor, df

def prepare_dataset(ds_pos_folder, ds_neg_folder, dataDir):
    """
    Returns X and y tensor and DataFrame with sequence id, length and dataset type
    """
    X_tensor_pos, y_tensor_pos, df_pos = load_tensor_folder(ds_pos_folder, label_int=1, dataDir=dataDir)
    X_tensor_neg, y_tensor_neg, df_neg = load_tensor_folder(ds_neg_folder, label_int=0, dataDir=dataDir)

    X_tensor_all = torch.concat([X_tensor_pos, X_tensor_neg]).type(torch.float32)
    y_tensor_all = torch.concat([y_tensor_pos, y_tensor_neg]).type(torch.int64)

    X_df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    
    return X_tensor_all, y_tensor_all, X_df_all

def split_train_val(X_trainval, y_trainval, df_trainval, val_frac=0.10):
    # Testset dataset
    trainvalset = torch.utils.data.TensorDataset(X_trainval, y_trainval)

    # Split X_train and X_valid
    val_frac = 0.10
    valid_n = int(len(X_trainval)*0.10)
    train_n = len(X_trainval) - valid_n
    train, valid = torch.utils.data.random_split(X_trainval, [train_n, valid_n])

    df_train = df_trainval.iloc[train.indices]
    df_valid = df_trainval.iloc[valid.indices]

    X_train, y_train = X_trainval[train.indices], y_trainval[train.indices]
    X_valid, y_valid = X_trainval[valid.indices], y_trainval[valid.indices]

    return X_train, y_train, df_train, X_valid, y_valid, df_valid

@click.command()
@click.argument('interim_filepath', default='data/interim/', type=click.Path())
@click.argument('output_filepath', default='data/processed/', type=click.Path())
def main(interim_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # Get fitted PCA object
    pca_train, embed_dim, max_len = get_PCA_fit_train()
    logger.info(f'Saving fitted PCA to {procDir}pca_train.pt')
    torch.save(pca_train, procDir + "pca_train.pt")

    # PCA transform and save embedding files
    # NB: Saving of intermediate files could be skipped
    log.info(f"PCA transforming and saving embedding pt files in {dirType_list}")
    PCA_transformed_save_embed_files(pca_train, embed_dim, max_len, dirType_list, fasta_list, interimDir, procDir)

    # Prepare dataset
    # To-do: Could work directly with data from above function
    logger.info(f'Preparing train, validation and test TensorDataset')
    X_trainval, y_trainval, df_trainval = prepare_dataset(ds_pos_folder="train_pos", ds_neg_folder="train_neg", dataDir=procDir)
    X_test, y_test, df_test = prepare_dataset(ds_pos_folder="test_pos", ds_neg_folder="test_neg", dataDir=procDir)

    logger.info(f'Splitting of {val_frac} of training dataset into validation')
    X_train, y_train, df_train, X_valid, y_valid, df_valid = split_train_val(X_trainval, y_trainval, df_trainval, val_frac)

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    validset = torch.utils.data.TensorDataset(X_valid, y_valid)
    testset = torch.utils.data.TensorDataset(X_test, y_test)

    # Save
    log.info(f"Saving trainset.pt ({trainset.tensors[0].shape}), trainset.csv to {procDir}")
    df_train.to_csv(procDir + "trainset.csv", index=False)
    torch.save(trainset, procDir + "trainset.pt")

    log.info(f"Saving validset.pt ({validset.tensors[0].shape}), validset.csv to {procDir}")
    df_valid.to_csv(procDir + "validset.csv", index=False)
    torch.save(validset, procDir + "validset.pt")

    log.info(f"Saving testset.pt ({testset.tensors[0].shape}), testset.csv to {procDir}")
    df_test.to_csv(procDir + "testset.csv", index=False)
    torch.save(testset, procDir + "testset.pt")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
    log = logging.getLogger(__name__)
    log.info("making dataset")

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
