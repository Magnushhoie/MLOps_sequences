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


def de_duplicate_FASTA_files(inDir="data/raw/", outDir="data/interim/", v=1):
    # Find train and test fasta files in raw
    fasta_files = glob.glob(inDir + "test*.fasta") + glob.glob(inDir + "train*.fasta")
    print("Checking for duplicate sequences in:\n", fasta_files, end="\n")

    # Count duplicates and entries across all fasta files
    dict_seqs = {}
    dict_ids = {}
    count_entries = 0

    # Load individual FASTA files and check whether sequences are duplicates before saving
    # IDs are only stored for counting
    for inFasta in fasta_files:
        outFasta = outDir + "filtered_" + os.path.basename(inFasta)

        with open(outFasta, 'w') as outHandle:
            no_duplicates = 0
            for record in SeqIO.parse(inFasta, "fasta"):
                count_entries += 1

                # Store unique sequences for later checking
                # Write ID+sequence if sequence is unique
                if str(record.seq) not in dict_seqs:
                    dict_seqs[str(record.seq)] = True
                    SeqIO.write(record, outHandle, 'fasta')

                    # Store unique IDs
                    if str(record.id) not in dict_ids:
                        dict_ids[str(record.id)] = True

                else:
                    no_duplicates += 1

            if v:
                print(f"Found {no_duplicates} duplicate sequences in {inFasta}")
                print(f"Saving to {outFasta}")
                
    # Print statistics
    print(f"\nProcessed {count_entries} sequences from {len(fasta_files)} train/test FASTA files in {inDir} to {outDir}:")
    print(f"Unique IDs / Sequences: {len(dict_ids.keys())} / {len(dict_seqs.keys())}")



@click.command()
@click.argument('input_filepath', default='data/raw/', type=click.Path(exists=True))
@click.argument('interim_filepath', default='data/interim/', type=click.Path())
@click.argument('output_filepath', default='data/processed/', type=click.Path())
def main(input_filepath, interim_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Preprocess sequences
    de_duplicate_FASTA_files(inDir=input_filepath, outDir=interim_filepath)

    # Run ESM pipeline
    #call("${PYTHON_INTERPRETER} -u " + "src/data/extract.py " + "", shell=True)
    script="src/data/run_esm.sh"
    # os.chmod(script, 0o755)
    subprocess.call(script)

    # Open ESM .pt files, put sequences and labels inside
    # ....

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
