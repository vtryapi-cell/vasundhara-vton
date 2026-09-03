FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1

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
# Remove conflicting preinstalled Python packages
# ---------------------------------------------------------

RUN python -m pip uninstall -y \
    numpy \
    opencv-python \
    opencv-python-headless \
    matplotlib \
    scipy \
    2>/dev/null || true

# ---------------------------------------------------------
# Install NumPy FIRST
# ---------------------------------------------------------

RUN python -m pip install \
    --no-cache-dir \
    --force-reinstall \
    numpy==1.26.4

# ---------------------------------------------------------
# Verify NumPy before installing other packages
# ---------------------------------------------------------

RUN python -c "import numpy; print('NUMPY VERSION:', numpy.__version__); print('NUMPY PATH:', numpy.__file__)"

# ---------------------------------------------------------
# Application requirements
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# Copy application
# ---------------------------------------------------------

COPY . /workspace

# ---------------------------------------------------------
# Final dependency verification
# ---------------------------------------------------------

RUN python - <<'PY'
import numpy
print("========================================")
print("NumPy:", numpy.__version__)

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import diffusers
print("Diffusers:", diffusers.__version__)

import transformers
print("Transformers:", transformers.__version__)

print("========================================")
print("BASIC IMPORT TEST PASSED")
PY

# ---------------------------------------------------------
# Start RunPod worker
# ---------------------------------------------------------

CMD ["python", "-u", "handler.py"]
