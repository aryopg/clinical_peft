FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

WORKDIR /home

COPY requirements.txt /tmp/requirements.txt
COPY requirements_cpu.txt /tmp/requirements_cpu.txt

# Install required apt packages
RUN apt update -y --fix-missing
RUN apt install -y byobu git python3 python-is-python3 pip bc htop parallel nano wget unzip python3.10-venv sox ffmpeg libcairo2 libcairo2-dev libgirepository1.0-dev libdbus-1-dev

# Check if an NVIDIA GPU is available
RUN nvidia-smi && \
    pip install -r /tmp/requirements.txt || \
    pip install -r /tmp/requirements_cpu.txt