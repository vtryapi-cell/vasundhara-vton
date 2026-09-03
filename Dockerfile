FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Requirements
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

# Remove potentially incompatible scientific packages
RUN pip uninstall -y numpy scipy || true

# Install requirements
RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# FORCE FINAL SCIENTIFIC STACK
# ---------------------------------------------------------

RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 \
    scipy==1.13.1

# ---------------------------------------------------------
# Verify Python / NumPy / SciPy / Torch / Diffusers
# DURING DOCKER BUILD
# ---------------------------------------------------------

RUN python - <<'PY'
import sys

print("========================================")
print("Python:", sys.version)

import numpy
print("NumPy:", numpy.__version__)

import scipy
print("SciPy:", scipy.__version__)

import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

import transformers
print("Transformers:", transformers.__version__)

import diffusers
print("Diffusers:", diffusers.__version__)

import scipy.sparse
print("SciPy sparse: OK")

import scipy.stats
print("SciPy stats: OK")

print("========================================")
print("SCIENTIFIC STACK OK")
print("========================================")
PY

# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py
COPY vton /workspace/vton

# ---------------------------------------------------------
# Start RunPod worker
# ---------------------------------------------------------

CMD ["python", "-u", "/workspace/handler.py"]
