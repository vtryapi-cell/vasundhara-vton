FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# SYSTEM
# =========================================================

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# REMOVE OLD SCIENTIFIC PACKAGES
# =========================================================

RUN pip uninstall -y numpy scipy || true

# =========================================================
# REQUIREMENTS
# =========================================================

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# =========================================================
# FINAL NUMPY / SCIPY PIN
# =========================================================

RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 \
    scipy==1.13.1

# =========================================================
# VERIFY ENVIRONMENT
# =========================================================

RUN python - <<'PY'
import sys

print("")
print("==============================================")
print("VASUNDHARA VTON - BUILD ENVIRONMENT TEST")
print("==============================================")

print("Python:", sys.version)

import numpy
print("NumPy:", numpy.__version__)

assert numpy.__version__ == "1.26.4"

import scipy
print("SciPy:", scipy.__version__)

assert scipy.__version__ == "1.13.1"

import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

import transformers
print("Transformers:", transformers.__version__)

import diffusers
print("Diffusers:", diffusers.__version__)

import accelerate
print("Accelerate:", accelerate.__version__)

import scipy.sparse
print("SciPy sparse: OK")

import scipy.stats
print("SciPy stats: OK")

print("")
print("==============================================")
print("SCIENTIFIC STACK OK")
print("==============================================")
PY

# =========================================================
# CHECK PYTHON DEPENDENCIES
# =========================================================

RUN pip check

# =========================================================
# APPLICATION
# =========================================================

COPY handler.py /workspace/handler.py

COPY vton /workspace/vton

# =========================================================
# CHECK APPLICATION FILES
# =========================================================

RUN echo "==============================================" && \
    echo "VTON FILES" && \
    echo "==============================================" && \
    find /workspace/vton -maxdepth 2 -type f -print && \
    echo "=============================================="

# =========================================================
# CHECK HANDLER IMPORT
# =========================================================

RUN python - <<'PY'
print("Testing handler import...")

import handler

print("handler.py import: OK")
print("==============================================")
print("APPLICATION IMPORT TEST PASSED")
print("==============================================")
PY

# =========================================================
# START RUNPOD
# =========================================================

CMD ["python", "-u", "/workspace/handler.py"]
