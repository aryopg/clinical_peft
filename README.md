# clinical_peft
Parameter-efficient Fine Tuning for Clinical LLMs


# Setup

## Install packages

```
conda install sentencepiece pydantic python-dotenv black isort tqdm wandb pandas matplotlib -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install transformers datasets huggingface_hub -c huggingface -y
conda install accelerate -c conda-forge -y
```