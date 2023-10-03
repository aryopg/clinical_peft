FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

WORKDIR /home

COPY requirements.txt /tmp/requirements.txt
COPY requirements_cpu.txt /tmp/requirements_cpu.txt

# Install required apt packages
RUN apt update -y --fix-missing
RUN apt install -y byobu git wget unzip htop zsh parallel

# Check if an NVIDIA GPU is available
RUN wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda3
RUN rm ./Miniconda3-latest-Linux-x86_64.sh
RUN ln -s /home/miniconda3/bin/conda /usr/bin/conda

RUN conda update -n base -c defaults conda -y
RUN conda init
RUN conda install python=3.10 -y
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
RUN conda install transformers datasets huggingface_hub evaluate -c huggingface -y
RUN conda install sentencepiece pydantic python-dotenv black isort tqdm wandb pandas matplotlib accelerate scikit-learn pynvml -c conda-forge -y
RUN rm -rf /home/miniconda3/pkgs/*
RUN PYTHONDONTWRITEBYTECODE=1

RUN pip install deepspeed
RUN pip install peft==0.4.0
RUN pip install -U tokenizers==0.13.3
RUN pip install -U pydantic==1.10.12