FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1
ENV FORCE_CUDA=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System/build dependencies for CatVTON + DensePose/Detectron2
RUN apt-get update && apt-get install -y \
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

# Remove conflicting scientific/diffusion packages from the RunPod base image.
RUN python -m pip uninstall -y \
    numpy scipy diffusers transformers accelerate safetensors huggingface-hub \
    tokenizers opencv-python opencv-python-headless \
    2>/dev/null || true

# Stable scientific stack.
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    Pillow==10.3.0 \
    opencv-python-headless==4.10.0.84

# CatVTON-compatible diffusion stack.
RUN python -m pip install --no-cache-dir \
    diffusers==0.29.2 \
    transformers==4.27.3 \
    accelerate==0.31.0 \
    safetensors==0.4.5 \
    huggingface-hub==0.23.4 \
    tokenizers==0.13.3 \
    sentencepiece \
    protobuf \
    tqdm==4.66.4 \
    scikit-image==0.24.0 \
    matplotlib==3.9.1 \
    PyYAML==6.0.1 \
    Ninja==1.11.1.1

# Detectron2/DensePose required by CatVTON AutoMasker.
RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git /opt/detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2/projects/DensePose

# Upstream CatVTON source.
RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /workspace/CatVTON

# Runtime/application packages.
RUN python -m pip install --no-cache-dir \
    runpod \
    einops \
    requests \
    gradio

# Copy our application.
COPY . /workspace

# ------------------------------------------------------------
# FINAL DEPENDENCY LOCK
# ------------------------------------------------------------
# Run this after COPY so the final image always contains the exact
# diffusion packages required by handler.py.
RUN python -m pip install --no-cache-dir \
    "diffusers==0.29.2" \
    "transformers==4.27.3" \
    "accelerate==0.31.0" \
    "safetensors==0.4.5" \
    "huggingface-hub==0.23.4" \
    "tokenizers==0.13.3"

# ------------------------------------------------------------
# HARD BUILD VALIDATION
# ------------------------------------------------------------
# The Docker build MUST fail here if Diffusers is missing/broken.
# We intentionally do NOT import handler.py during build because handler.py
# loads the GPU model at import time.
RUN python - <<'PY'
import sys
import numpy
import scipy
import torch
import torchvision
import diffusers
import transformers
import accelerate
import safetensors
import huggingface_hub
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

print("========================================")
print("VASUNDHARA VTON FINAL ENVIRONMENT")
print("========================================")
print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("Safetensors:", safetensors.__version__)
print("HuggingFace Hub:", huggingface_hub.__version__)
print("Pillow:", Image.__version__)
print("VaeImageProcessor: OK")
print("========================================")
print("DIFFUSERS ENVIRONMENT CHECK PASSED")
print("========================================")
PY

# Validate CatVTON source without loading weights.
RUN test -f /workspace/CatVTON/model/pipeline.py
RUN test -f /workspace/CatVTON/model/cloth_masker.py

RUN python - <<'PY'
import ast
from pathlib import Path

pipeline = Path('/workspace/CatVTON/model/pipeline.py')
cloth_masker = Path('/workspace/CatVTON/model/cloth_masker.py')
handler = Path('/workspace/handler.py')

ast.parse(pipeline.read_text())
ast.parse(cloth_masker.read_text())
ast.parse(handler.read_text())
print("CatVTON source syntax: OK")
print("Handler source syntax: OK")
print("CATVTON BUILD CHECK PASSED")
PY

# Validate our own VTON code.
RUN test -f /workspace/vton/model.py
RUN test -f /workspace/vton/train_dataset.py
RUN test -f /workspace/train.py

RUN python - <<'PY'
import ast
from pathlib import Path

for name in ['vton/model.py', 'vton/train_dataset.py', 'train.py']:
    ast.parse(Path('/workspace', name).read_text())
    print(name, 'syntax: OK')

print('VASUNDHARA VTON SOURCE CHECK PASSED')
PY

CMD ["python", "-u", "/workspace/handler.py"]
