#!/bin/bash
# Generates ESM embeddings per sequence and stores to interim/FASTA_NAME/ID.pt files

inDir=${1:-"data/interim"}
esm_model=${2:-"esm1_t6_43M_UR50S"}

for fastaFile in "$inDir"/*.fasta; do
    dirName=$(basename "$fastaFile")
    dirName="${dirName%.*}"
    outDir="$inDir/$dirName"
    mkdir -p $outDir

    python src/data/extract.py $esm_model "$fastaFile" "$outDir" \
        --repr_layers -1 --include per_tok
done