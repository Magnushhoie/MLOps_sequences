#!/bin/bash

# Remove the previously built .mar
rm models/model_store/PepCNN.mar

# Create the .mar file
torch-model-archiver --model-name PepCNN \
    --version 1.0 \
    --serialized-file models/MLOps_Sequences/mlops_antimb/pretrained_cnn.pt \
    --export-path models/model_store \
    --handler src/models/model_handler.py

# Initiate the server
    torchserve --start --ncs --model-store models/model_store --models PepCNN.mar

## TO RUN ON ANOTHER TERMINAL
# curl http://127.0.0.1:8080/predictions/PepCNN -T test_serve.fasta
