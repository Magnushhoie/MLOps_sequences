#!/bin/bash

# Params
rawDir=${1:-"data/raw"}
interimDir=${2:-"data/interim"}
procDir=${3:-"data/processed"}
esm_model=${4:-"esm1_t6_43M_UR50S"} # From https://github.com/facebookresearch/esm

# Pipeline
echo -e "\n1. Pre-processing and checking raw dataset sequences from $rawDir to $interimDir"
python src/data/1_preprocess_sequences.py "$rawDir/" "$interimDir/"

echo -e "\n2. Embedding sequences to N x L x 768 using wrapper with ESM model $esm_model using extract.py on FASTA files in $interimDir"
chmod +x src/data/2_embed_sequences_esm.sh
./src/data/2_embed_sequences_esm.sh "$interimDir/" $esm_model

echo -e "\n3. PCA-reduce and prepare final torch .pt files for each sequence+embedding+label"
python src/data/3_process_embedded_sequences.py