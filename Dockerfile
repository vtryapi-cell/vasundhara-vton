FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=8.0+PTX
ENV RUNPOD_INIT_TIMEOUT=900

# CatVTON's published environment is based on PyTorch 2.4 / torchvision 0.19.
# Detectron2 v0.6 is pinned because it is the version used by CatVTON's
# DensePose integration. The Docker builder has no GPU, so FORCE_CUDA and
# TORCH_CUDA_ARCH_LIST are required for the source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the scientific stack aligned with CatVTON.
RUN python -m pip uninstall -y \
    numpy scipy diffusers transformers accelerate safetensors huggingface-hub \
    tokenizers opencv-python opencv-python-headless \
    2>/dev/null || true

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    Pillow==10.3.0 \
    opencv-python-headless==4.10.0.84

RUN python -m pip install --no-cache-dir \
    diffusers==0.29.2 \
    transformers==4.46.3 \
    accelerate==0.31.0 \
    safetensors==0.4.5 \
    huggingface-hub==0.26.2 \
    tokenizers==0.20.3 \
    sentencepiece==0.2.0 \
    protobuf==4.25.8 \
    tqdm==4.66.4 \
    scikit-image==0.24.0 \
    matplotlib==3.9.1 \
    PyYAML==6.0.1 \
    einops==0.8.2 \
    Ninja==1.11.1.1 \
    cloudpickle==3.0.0 \
    omegaconf==2.3.0 \
    pycocotools==2.0.8

# CatVTON-compatible Detectron2/DensePose.
RUN git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git /opt/detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2/projects/DensePose

# Upstream CatVTON source.
RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /workspace/CatVTON

# Runtime packages.
RUN python -m pip install --no-cache-dir \
    runpod==1.12.0 \
    gradio==4.41.0 \
    requests

COPY . /workspace

# Final version lock. Do not let project files change the CatVTON runtime.
RUN python -m pip install --no-cache-dir \
    "diffusers==0.29.2" \
    "transformers==4.46.3" \
    "accelerate==0.31.0" \
    "safetensors==0.4.5" \
    "huggingface-hub==0.26.2" \
    "tokenizers==0.20.3"

# Build-time smoke tests. No model weights are downloaded here.
RUN python - <<'PY'
import sys
import torch
import torchvision
import numpy
import scipy
import diffusers
import transformers
import accelerate
import detectron2
import densepose
from diffusers.image_processor import VaeImageProcessor

print('=== VASUNDHARA VTON BUILD VALIDATION ===')
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available during build:', torch.cuda.is_available())
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('Diffusers:', diffusers.__version__)
print('Transformers:', transformers.__version__)
print('Accelerate:', accelerate.__version__)
print('Detectron2:', detectron2.__version__)
print('DensePose import: OK')
print('VaeImageProcessor: OK')
print('BUILD ENVIRONMENT PASSED')
PY

# Validate application/source syntax without importing the GPU worker.
RUN test -f /workspace/CatVTON/model/pipeline.py
RUN test -f /workspace/CatVTON/model/cloth_masker.py
RUN test -f /workspace/handler.py
RUN test -f /workspace/vton/model.py
RUN test -f /workspace/vton/train_dataset.py
RUN test -f /workspace/train.py

RUN python - <<'PY'
import ast
from pathlib import Path

files = [
    '/workspace/CatVTON/model/pipeline.py',
    '/workspace/CatVTON/model/cloth_masker.py',
    '/workspace/handler.py',
    '/workspace/vton/model.py',
    '/workspace/vton/train_dataset.py',
    '/workspace/train.py',
]
for name in files:
    ast.parse(Path(name).read_text())
    print(name, 'syntax: OK')
print('VASUNDHARA VTON SOURCE CHECK PASSED')
PY

CMD ["python", "-u", "/workspace/handler.py"]
