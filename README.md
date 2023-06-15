# Clinical LLaMA-LoRA

This repository contains the code for pre-training, fine-tuning, and evaluation for "Fine-Tuning LLaMA for the Clinical Domain" (in submission)

<!-- omit in toc -->
## Table of Contents
- [🛠️ Setup](#️-setup)
  - [Python packages](#python-packages)
  - [Dataset](#dataset)
- [🤖 Training](#-training)
  - [Prepare the MIMIC-IV dataset](#prepare-the-mimic-iv-dataset)
  - [Prepare the downstream datasets](#prepare-the-downstream-dataset)
  - [Training the model](#training-the-model)
- [⚖️ Results](#️-results)
  - [Domain adaptation pretraining](#domain-adaptation-pretraining)
  - [Downstream fine-tuning](#downstream-fine-tuning)


## 🛠️ Setup
### Python packages
This codebase requires the following dependencies:
```
- numpy
- pandas
- pytorch
- transformers
- datasets
- huggingface-hub
- evaluate
- pydantic
- scikit-learn
- python-dotenv
- black
- isort
- PyYAML
- tqdm
- wandb
- jupyterlab
- matplotlib
```

We opted in to using conda as our package manager. The following will install all necessary dependencies for a GPU training:
```
ENV_NAME=clinical_peft
conda create -n ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME}
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install transformers datasets huggingface_hub evaluate -c huggingface -y
conda install sentencepiece pydantic python-dotenv black isort tqdm wandb pandas matplotlib accelerate scikit-learn -c conda-forge -y
```

### Dataset
TBA

## 🤖 Training

### Prepare the MIMIC-IV dataset

TBA

### Prepare the downstream datasets

TBA

### Training the models
TBA

## ⚖️ Results

### Domain adaptation pretraining

TBA

### Downstream fine-tuning

TBA
