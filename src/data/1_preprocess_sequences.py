# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
from pathlib import Path

import click
from Bio import SeqIO

import subprocess
from dotenv import find_dotenv, load_dotenv

def find_train_test_files(inDir="data/raw/"):
    pass


def de_duplicate_FASTA_files(fastaList, outName, interimDir="data/interim/", v=1):
    """ Reads and checks list of FASTA files, and saves individual ID+sequence FASTA to fastaOutdir in interimDir """
    if "count_entries" not in globals():
        global dict_seqs, dict_ids, count_entries, no_duplicates
        dict_seqs, dict_ids, count_entries, no_duplicates = {}, {}, 0, 0

    outFasta = interimDir + outName
    
    # Find train and test fasta files in raw
    print("Checking for duplicate sequences in:\n", fastaList, end="\n")
    print(f"Saving to {outFasta}")

    # Load individual FASTA files and check whether sequences are duplicates before saving
    # IDs are only stored for counting
    for inFasta in fastaList:
        with open(outFasta, 'w') as outHandle:
            for record in SeqIO.parse(inFasta, "fasta"):
                count_entries += 1

                # Store unique sequences for later checking
                # Write ID+sequence if sequence is unique
                if str(record.seq) not in dict_seqs:
                    dict_seqs[str(record.seq)] = True
                    
                    # If ID is not a duplicate, write FASTA
                    if str(record.id) not in dict_ids:
                        SeqIO.write(record, outHandle, 'fasta')
                        dict_ids[str(record.id)] = True
                    else:
                        print(f"NB: Duplicate ID {record.id} for sequence {record.seq}")

                else:
                    no_duplicates += 1

            if no_duplicates >= 1:
                print(f"NB: Found {no_duplicates} duplicate sequences in {inFasta}")



@click.command()
@click.argument('input_filepath', default='data/raw/', type=click.Path(exists=True))
@click.argument('interim_filepath', default='data/interim/', type=click.Path())
def main(input_filepath, interim_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Get training and testing FASTAs
    test_pos_list = glob.glob(input_filepath + "test/pos/*.fasta")
    test_neg_list = glob.glob(input_filepath + "test/neg/*.fasta")
    train_pos_list = glob.glob(input_filepath + "train/pos/*.fasta")
    train_neg_list = glob.glob(input_filepath + "train/neg/*.fasta")

    fasta_list_list = [test_pos_list, test_neg_list, train_pos_list, train_neg_list]
    outName_list = ["test_pos.fasta", "test_neg.fasta", "train_pos.fasta", "train_neg.fasta"]

    # Preprocess sequences
    for fastaList, outName in zip(fasta_list_list, outName_list):
        de_duplicate_FASTA_files(fastaList, outName, interim_filepath)

    # Print statistics
    print(f"\nProcessed {count_entries} sequences from {len(fasta_list_list)} FASTA files in {input_filepath}:")
    print(f"Unique IDs / Sequences: {len(dict_ids.keys())} / {len(dict_seqs.keys())}")

    # Split into separate unique 

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
