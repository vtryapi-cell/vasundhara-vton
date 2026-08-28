FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV CATVTON_ROOT=/opt/CatVTON

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

# CUDA-enabled PyTorch
RUN python -m pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Application dependencies
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Verify torch is installed during the build
RUN python -c "import torch; print('TORCH:', torch.__version__); print('CUDA:', torch.version.cuda)"

# CatVTON
RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /opt/CatVTON

COPY app.py .

COPY templates ./templates
COPY static ./static

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "1", "--timeout", "900", "app:app"]
