MLOps Sequences
==============================

This project is part of our hand-in for the DTU Compute course 02476 - Machine Learning Operations, available here:
https://skaftenicki.github.io/dtu_mlops/

# Project goals:
The goal of our project is to learn and apply the tools in the Machine Learning Operations (MLOps) toolbox to predict whether a given peptide is anti-microbial or not, directly from its amino-acid sequence. Anti-microbial peptides are an important part of the innate immune system of many organisms. There is wide interest in predicting new peptides of this class, as these have useful properties in inhibiting bacteria, fungi and other microorganisms for example in clinical settings. 

In this project we aim to apply the following MLOps tools:
- Pytorch/Pytorch lightning for prediction model architecture
- MLFlow for logging experiment training and hyperparameters
- Optuna/Raytune for hyperparameter optimization
- Git and DVC for code and dataset version control
- Docker containers for reproducibility across systems
- Pytest and Github Actions for continuous integration
- Google Cloud Platform for including model training, dataset processing and deployment
- FastAPI for interfacing with deployed model
- TorchDrift for monitoring of possible data drift in our deployed model

## Dataset:
We will use a dataset of ca. 170.000 peptides which have been experimentally investigated for anti-microbial activity, of which ca. 3.000 peptides have a positive anti-microbial status. The peptides range in length from 5 to 255 amino-acids or tokens.

The dataset has already been benchmarked by the Computational Biology and Bioinformatics Lab at the University of Macau, and is available in several sizes from: 
https://cbbio.online/AxPEP/?action=dataset

## Approach:
We will apply a protein language model known as ESM by Facebook Research, an unsupervised transformer pre-trained to predict amino-acid probabilities in protein sequences. We will fine-tune ESM on our peptide dataset, and add additional layers for classifying peptides based on their anti-microbial status.

Several pre-trained models ranging in size from ca. 40 to 700M parameters is available here:
https://github.com/facebookresearch/esm

The use of the larger ESM-1b model (650M parameters) is also described on HuggingFace:
https://huggingface.co/facebook/esm-1b


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
