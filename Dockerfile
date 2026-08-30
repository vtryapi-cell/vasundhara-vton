FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------
# Install application dependencies
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# Force compatible NumPy version
# ---------------------------------------------------------

RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4

# ---------------------------------------------------------
# Verify the Python environment
# ---------------------------------------------------------

RUN python - <<'PY'
import numpy
print("========================================")
print("NUMPY VERSION:", numpy.__version__)
print("========================================")

import torch
print("TORCH VERSION:", torch.__version__)
print("TORCH CUDA:", torch.version.cuda)
print("CUDA AVAILABLE:", torch.cuda.is_available())
print("========================================")

import cv2
print("OPENCV VERSION:", cv2.__version__)
print("========================================")

import scipy
print("SCIPY VERSION:", scipy.__version__)
print("========================================")

import skimage
print("SKIMAGE VERSION:", skimage.__version__)
print("========================================")

import matplotlib
print("MATPLOTLIB VERSION:", matplotlib.__version__)
print("========================================")

import transformers
print("TRANSFORMERS VERSION:", transformers.__version__)
print("========================================")

import diffusers
print("DIFFUSERS VERSION:", diffusers.__version__)
print("========================================")

print("ALL CORE IMPORTS SUCCESSFUL")
print("========================================")
PY

# ---------------------------------------------------------
# Download CatVTON
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# ---------------------------------------------------------
# CatVTON paths
# ---------------------------------------------------------

ENV CATVTON_ROOT=/workspace/CatVTON
ENV PYTHONPATH=/workspace/CatVTON

# ---------------------------------------------------------
# Copy RunPod handler
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py

# ---------------------------------------------------------
# Start worker
# ---------------------------------------------------------

WORKDIR /workspace/CatVTON

CMD ["python", "/workspace/handler.py"]
