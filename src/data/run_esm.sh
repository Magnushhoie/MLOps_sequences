#!/bin/bash
# Generates ESM embeddings per sequence and stores to interim/FASTA_NAME/ID.pt files

inDir="data/interim"

for fastaFile in data/interim/*.fasta; do
    dirName=$(basename "$fastaFile")
    dirName="${dirName%.*}"
    outDir="$inDir/$dirName"

    echo "python -u src/data/extract.py esm1_t6_43M_UR50S $fastaFile $outDir \
        --repr_layers -1 --include per_tok"
    python src/data/extract.py esm1_t6_43M_UR50S $fastaFile $outDir \
        --repr_layers -1 --include per_tok
done