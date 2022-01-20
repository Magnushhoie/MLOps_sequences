#/bin/bash
touch .env

# Prepare dataset from data/raw
make data

# Train on prepared data in data/processed
# Without weights and biases authentication
export WANDB_MODE=disabled
export WANDB_DISABLE_CODE=true
make train
