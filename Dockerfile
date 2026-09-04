FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1
ENV FORCE_CUDA=1

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

# Keep the scientific stack compatible with the current PyTorch image.
RUN python -m pip uninstall -y \
    numpy scipy diffusers transformers accelerate safetensors huggingface-hub \
    opencv-python opencv-python-headless \
    2>/dev/null || true

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    Pillow==10.3.0 \
    opencv-python-headless==4.10.0.84

# CatVTON's tested diffusion stack. We deliberately do not install its old
# torch/xformers pins because the RunPod base image already supplies PyTorch.
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

# Detectron2 is required by CatVTON's AutoMasker/DensePose path.
# Build it against the exact PyTorch/CUDA stack in this image.
RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git /opt/detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2
RUN python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2/projects/DensePose

# Clone CatVTON source because the application repository intentionally does
# not vendor the upstream project.
RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /workspace/CatVTON

# Runtime/application packages.
RUN python -m pip install --no-cache-dir \
    runpod \
    einops \
    requests \
    gradio

# Basic environment checks.
RUN python - <<'PY'
import sys
import numpy
import scipy
import torch
import torchvision
import diffusers
import transformers
import accelerate
from PIL import Image
import detectron2
import densepose

print("========================================")
print("ENVIRONMENT CHECK")
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
print("Detectron2: OK")
print("DensePose: OK")
print("Pillow:", Image.__version__)
from diffusers.image_processor import VaeImageProcessor
print("Diffusers imports: OK")
print("========================================")
print("BUILD ENVIRONMENT OK")
print("========================================")
PY

# Copy our application after the upstream CatVTON clone so the two projects
# remain separate. The uploaded repository does not contain CatVTON/.
COPY . /workspace

# The COPY above may not include an upstream CatVTON directory; restore it if
# the build context ever contains a conflicting placeholder directory.
RUN test -f /workspace/CatVTON/model/pipeline.py
RUN test -f /workspace/CatVTON/model/cloth_masker.py
RUN test -f /workspace/vton/model.py
RUN test -f /workspace/vton/train_dataset.py
RUN test -f /workspace/train.py

# Verify upstream CatVTON imports from its own source root.
RUN python - <<'PY'
import sys
sys.path.insert(0, "/workspace/CatVTON")
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding, init_weight_dtype
print("CatVTONPipeline import: OK")
print("AutoMasker import: OK")
print("CatVTON utils import: OK")
print("========================================")
print("CATVTON BUILD CHECK PASSED")
print("========================================")
PY

# Verify our own trainable VTON code separately.
RUN python - <<'PY'
import sys
sys.path.insert(0, "/workspace")
from vton.model import create_model
from vton.train_dataset import VTONDataset
model = create_model(device="cpu")
print("VasundharaVTON import: OK")
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
print("========================================")
print("VASUNDHARA VTON BUILD CHECK PASSED")
print("========================================")
PY

CMD ["python", "-u", "/workspace/handler.py"]
