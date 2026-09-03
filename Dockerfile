FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------
# System packages
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Remove potentially incompatible scientific packages
# ---------------------------------------------------------
RUN pip uninstall -y numpy scipy || true

# ---------------------------------------------------------
# Install compatible NumPy + SciPy
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1

# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------
COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# Force final NumPy/SciPy versions
# ---------------------------------------------------------
RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 \
    scipy==1.13.1

# ---------------------------------------------------------
# Verify scientific stack
# ---------------------------------------------------------
RUN python - <<'PY'
import numpy
import scipy
import torch

print("========================================")
print("NUMPY :", numpy.__version__)
print("SCIPY :", scipy.__version__)
print("TORCH :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("CUDA AVAILABLE:", torch.cuda.is_available())
print("========================================")

import numpy._core
import scipy.sparse

print("NumPy import: OK")
print("SciPy import: OK")
print("PyTorch import: OK")
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
