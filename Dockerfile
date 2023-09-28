#--- USAGE

# Build image:
# > sudo docker build -t <tag> .

# Test if container runs:
# > sudo docker container run -it --name <name> -t <tag> python ...

# Push to docker hub:
# > docker push <tag>

# Image
FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

WORKDIR /home

# Install required apt packages
RUN apt update -y --fix-missing
RUN apt install -y byobu git python3 python-is-python3 pip bc htop parallel nano wget unzip

# Install miniconda
RUN echo "$(uname -s)"
RUN if [ "$(uname -s)" = "Linux" ]; then \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
    else \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh; \
    fi
RUN chmod +x miniconda.sh && ./miniconda.sh -b -p /home/miniconda3 && rm miniconda.sh
ENV PATH="/home/miniconda3/bin:${PATH}"

# Create sensorium environment
RUN conda update -n base -c defaults conda -y
RUN conda init
RUN conda install python=3.10 -y
RUN if [ "$(uname -s)" = "Linux" ]; then \
    RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 transformers datasets huggingface_hub evaluate sentencepiece python-dotenv black isort tqdm wandb pandas matplotlib accelerate scikit-learn pynvml -c conda-forge -c huggingface -c pytorch -c nvidia -y; \
    else \
    RUN conda install pytorch torchvision torchaudio transformers datasets huggingface_hub evaluate sentencepiece python-dotenv black isort tqdm wandb pandas matplotlib accelerate scikit-learn -c conda-forge -c huggingface -c pytorch -y \
    fi

RUN rm -rf /home/miniconda3/pkgs/*
ENV PYTHONDONTWRITEBYTECODE=1

RUN if [ "$(uname -s)" = "Linux" ]; then \
    RUN pip install --no-cache-dir deepspeed; \
    fi

RUN pip install --no-cache-dir -U peft==0.4.0
RUN pip install --no-cache-dir -U tokenizers==0.13.3
RUN pip install --no-cache-dir -U pydantic==1.10.12