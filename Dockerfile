FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PIP_NO_CACHE_DIR=1

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

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1

RUN python -m pip install --no-cache-dir \
    diffusers==0.30.3 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    safetensors==0.4.5 \
    huggingface-hub==0.24.6 \
    tokenizers==0.19.1

RUN python -m pip install --no-cache-dir \
    Pillow \
    opencv-python-headless \
    einops \
    tqdm \
    requests \
    gradio

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

COPY . /workspace

RUN test -f /workspace/handler.py
RUN test -f /workspace/CatVTON/model/pipeline.py
RUN test -f /workspace/vton/model.py
RUN test -f /workspace/vton/train_dataset.py
RUN test -f /workspace/train.py

RUN python - <<'PY'
import sys
sys.path.insert(0, "/workspace/CatVTON")

print("Testing CatVTON import...")
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker

print("CatVTONPipeline import: OK")
print("AutoMasker import: OK")
print("========================================")
print("CATVTON BUILD CHECK PASSED")
print("========================================")
PY

RUN python - <<'PY'
import sys
sys.path.insert(0, "/workspace")

print("Testing Vasundhara VTON import...")
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
