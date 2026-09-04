FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1

# ---------------------------------------------------------
# System packages
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# IMPORTANT:
# Clean conflicting preinstalled scientific packages
# ---------------------------------------------------------
RUN python -m pip uninstall -y \
    numpy \
    scipy \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    huggingface-hub \
    opencv-python \
    opencv-python-headless \
    2>/dev/null || true

# ---------------------------------------------------------
# Stable NumPy/SciPy versions for this environment
# ---------------------------------------------------------
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1

# ---------------------------------------------------------
# HuggingFace / Diffusion stack
# ---------------------------------------------------------
RUN python -m pip install --no-cache-dir \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    huggingface-hub==0.24.6 \
    tokenizers==0.19.1

# ---------------------------------------------------------
# Image / utility packages
# ---------------------------------------------------------
RUN python -m pip install --no-cache-dir \
    Pillow \
    opencv-python-headless \
    einops \
    tqdm \
    requests \
    gradio

# ---------------------------------------------------------
# VERIFY ENVIRONMENT DURING BUILD
# If this fails, Docker build stops here instead of
# producing a broken RunPod worker.
# ---------------------------------------------------------
RUN python - <<'PY'
import sys
import numpy
import scipy
import torch
import diffusers
import transformers
import accelerate
from PIL import Image

print("========================================")
print("ENVIRONMENT CHECK")
print("========================================")
print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("Pillow:", Image.__version__)

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor

print("Diffusers imports: OK")
print("========================================")
print("BUILD ENVIRONMENT OK")
print("========================================")
PY

# ---------------------------------------------------------
# Copy project
# ---------------------------------------------------------
COPY . /workspace

# ---------------------------------------------------------
# Verify CatVTON source exists
# ---------------------------------------------------------
RUN test -f /workspace/handler.py
RUN test -f /workspace/CatVTON/model/pipeline.py

# ---------------------------------------------------------
# Final CatVTON import test
# ---------------------------------------------------------
RUN python - <<'PY'
import sys
sys.path.insert(0, "/workspace")

print("Testing CatVTON import...")

from model.pipeline import CatVTONPipeline

print("CatVTONPipeline import: OK")
print("========================================")
print("CATVTON BUILD CHECK PASSED")
print("========================================")
PY

# ---------------------------------------------------------
# Start RunPod worker
# ---------------------------------------------------------
CMD ["python", "-u", "/workspace/handler.py"]
