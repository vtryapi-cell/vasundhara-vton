FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# =========================================================
# SYSTEM DEPENDENCIES
# =========================================================

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# CLEAN CONFLICTING PYTHON PACKAGES
# =========================================================

RUN pip uninstall -y \
    numpy \
    scipy \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface-hub \
    opencv-python \
    opencv-python-headless \
    || true

# =========================================================
# NUMPY / SCIPY
# Keep these compatible with the PyTorch/CatVTON stack
# =========================================================

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1

# =========================================================
# CORE PYTHON DEPENDENCIES
# =========================================================

RUN pip install --no-cache-dir \
    diffusers==0.30.2 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    huggingface_hub==0.24.6 \
    einops \
    ftfy \
    regex \
    tqdm \
    requests \
    pillow \
    opencv-python-headless \
    omegaconf \
    sentencepiece

# =========================================================
# RUNPOD
# =========================================================

RUN pip install --no-cache-dir runpod

# =========================================================
# COPY PROJECT
# =========================================================

COPY . /workspace

# =========================================================
# VERIFY PYTHON DEPENDENCIES DURING BUILD
# If this fails, Docker build stops instead of deploying
# a broken worker.
# =========================================================

RUN python - <<'PY'
import sys

print("=" * 60)
print("VERIFYING PYTHON ENVIRONMENT")
print("=" * 60)

import numpy
print("NumPy       :", numpy.__version__)

import scipy
print("SciPy       :", scipy.__version__)

import torch
print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.version.cuda)
print("CUDA avail. :", torch.cuda.is_available())

import diffusers
print("Diffusers   :", diffusers.__version__)

import transformers
print("Transformers:", transformers.__version__)

import accelerate
print("Accelerate  :", accelerate.__version__)

import safetensors
print("Safetensors :", safetensors.__version__)

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor

print("Diffusers imports: OK")

print("=" * 60)
print("ALL PYTHON DEPENDENCIES OK")
print("=" * 60)
PY

# =========================================================
# VERIFY CATVTON FILES
# =========================================================

RUN test -f /workspace/handler.py && \
    test -f /workspace/CatVTON/model/pipeline.py && \
    echo "CatVTON project files: OK"

# =========================================================
# START RUNPOD WORKER
# =========================================================

CMD ["python", "-u", "/workspace/handler.py"]
