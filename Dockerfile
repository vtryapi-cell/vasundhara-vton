FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=0

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV CATVTON_ROOT=/workspace/CatVTON
ENV PYTHONPATH=/workspace/CatVTON:/workspace

# ---------------------------------------------------------
# SYSTEM PACKAGES
# ---------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# APPLICATION REQUIREMENTS
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN python -m pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# FORCE COMPATIBLE NUMPY
# ---------------------------------------------------------

RUN python -m pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4

# ---------------------------------------------------------
# CLONE CATVTON
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# ---------------------------------------------------------
# INSTALL IMPORTANT CATVTON DEPENDENCIES
# ---------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    fvcore==0.1.5.post20221221 \
    cloudpickle==3.0.0 \
    omegaconf==2.3.0 \
    pycocotools==2.0.8 \
    PyYAML==6.0.1 \
    tqdm==4.66.4 \
    peft>=0.17.0

# ---------------------------------------------------------
# VERIFY PYTHON + CUDA
# ---------------------------------------------------------

RUN python - <<'PY'
import sys
import torch
import numpy
import cv2
import scipy
import skimage
import matplotlib
import transformers
import diffusers
import accelerate
import huggingface_hub
import runpod

print("=" * 70)
print("VASUNDHARA VTON BUILD CHECK")
print("=" * 70)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("SciPy:", scipy.__version__)
print("Scikit-image:", skimage.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Transformers:", transformers.__version__)
print("Diffusers:", diffusers.__version__)
print("Accelerate:", accelerate.__version__)
print("HuggingFace Hub:", huggingface_hub.__version__)
print("RunPod:", runpod.__version__ if hasattr(runpod, "__version__") else "installed")

print("=" * 70)
print("BASE ENVIRONMENT OK")
print("=" * 70)
PY

# ---------------------------------------------------------
# VERIFY CATVTON IMPORTS
# ---------------------------------------------------------

RUN python - <<'PY'
import sys

sys.path.insert(0, "/workspace/CatVTON")

print("=" * 70)
print("TESTING CATVTON IMPORTS")
print("=" * 70)

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import (
    resize_and_crop,
    resize_and_padding,
    init_weight_dtype,
)

print("CatVTONPipeline: OK")
print("AutoMasker: OK")
print("resize_and_crop: OK")
print("resize_and_padding: OK")
print("init_weight_dtype: OK")

print("=" * 70)
print("CATVTON IMPORTS OK")
print("=" * 70)
PY

# ---------------------------------------------------------
# COPY SERVERLESS HANDLER
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py

# ---------------------------------------------------------
# VERIFY HANDLER IMPORTS
# ---------------------------------------------------------

RUN python - <<'PY'
import py_compile

print("=" * 70)
print("CHECKING HANDLER SYNTAX")
print("=" * 70)

py_compile.compile(
    "/workspace/handler.py",
    doraise=True
)

print("handler.py syntax: OK")
print("=" * 70)
PY

# ---------------------------------------------------------
# FINAL WORKING DIRECTORY
# ---------------------------------------------------------

WORKDIR /workspace

# ---------------------------------------------------------
# START RUNPOD SERVERLESS WORKER
# ---------------------------------------------------------

CMD ["python", "-u", "/workspace/handler.py"]
