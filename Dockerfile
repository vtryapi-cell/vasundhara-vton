FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/huggingface \
    TRANSFORMERS_CACHE=/workspace/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=8.0+PTX \
    RUNPOD_INIT_TIMEOUT=900

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential gcc g++ cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    Pillow==10.3.0 \
    opencv-python-headless==4.10.0.84 \
    diffusers==0.29.2 \
    transformers==4.46.3 \
    accelerate==0.31.0 \
    safetensors==0.4.5 \
    huggingface-hub==0.26.2 \
    tokenizers==0.20.3 \
    sentencepiece==0.2.0 \
    protobuf==4.25.8 \
    tqdm==4.66.4 \
    scikit-image==0.24.0 \
    matplotlib==3.9.1 \
    PyYAML==6.0.1 \
    einops==0.8.2 \
    Ninja==1.11.1.1 \
    cloudpickle==3.0.0 \
    omegaconf==2.3.0 \
    pycocotools==2.0.8 \
    runpod==1.12.0 \
    gradio==4.41.0 \
    requests

# CatVTON-compatible Detectron2/DensePose. Pinned to v0.6.
RUN git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git /opt/detectron2 \
    && python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2 \
    && python -m pip install --no-cache-dir --no-build-isolation -e /opt/detectron2/projects/DensePose

RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /workspace/CatVTON

COPY . /workspace

RUN python -m pip install --no-cache-dir \
    "diffusers==0.29.2" \
    "transformers==4.46.3" \
    "accelerate==0.31.0" \
    "safetensors==0.4.5" \
    "huggingface-hub==0.26.2" \
    "tokenizers==0.20.3"

# Build-time smoke tests: imports only, no model weights.
RUN python - <<'PY'
import sys, ast
from pathlib import Path
import torch, torchvision, numpy, scipy
import diffusers, transformers, accelerate, detectron2, densepose
from diffusers.image_processor import VaeImageProcessor

print('=== VASUNDHARA VTON BUILD VALIDATION ===')
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available during build:', torch.cuda.is_available())
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('Diffusers:', diffusers.__version__)
print('Transformers:', transformers.__version__)
print('Accelerate:', accelerate.__version__)
print('Detectron2:', detectron2.__version__)
print('DensePose: OK')
print('VaeImageProcessor: OK')

paths = [
    '/workspace/CatVTON/model/pipeline.py',
    '/workspace/CatVTON/model/cloth_masker.py',
    '/workspace/handler.py',
    '/workspace/vton/model.py',
    '/workspace/vton/train_dataset.py',
    '/workspace/train.py',
]
for p in paths:
    if not Path(p).is_file():
        raise FileNotFoundError(p)
    ast.parse(Path(p).read_text())
    print('SYNTAX OK:', p)

print('BUILD ENVIRONMENT PASSED')
PY

CMD ["python", "-u", "/workspace/handler.py"]
