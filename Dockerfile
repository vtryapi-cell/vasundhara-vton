FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/huggingface \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RUNPOD_INIT_TIMEOUT=900 \
    MODEL_PATH=/workspace/checkpoints/vasundhara-vton/best.pt

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential gcc g++ cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# VASUNDHARA worker dependencies only.
# No Gradio, CatVTON, Detectron2 or DensePose is used by the own model.
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    Pillow==10.3.0 \
    opencv-python-headless==4.10.0.84 \
    runpod==1.12.0 \
    tqdm==4.66.4

COPY . /workspace

# Build-time validation: verify the own model/training code and imports.
RUN python - <<'PY'
import ast
from pathlib import Path
import torch
import runpod
from PIL import Image

print('=== VASUNDHARA VTON BUILD VALIDATION ===')
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available during build:', torch.cuda.is_available())
print('RunPod:', getattr(runpod, '__version__', 'installed'))

for p in [
    '/workspace/handler.py',
    '/workspace/vton/model.py',
    '/workspace/vton/train_dataset.py',
    '/workspace/train.py',
]:
    if not Path(p).is_file():
        raise FileNotFoundError(p)
    ast.parse(Path(p).read_text())
    print('SYNTAX OK:', p)

from vton.model import create_model
model = create_model(device='cpu')
print('MODEL OK:', sum(p.numel() for p in model.parameters()), 'parameters')
print('BUILD ENVIRONMENT PASSED')
PY

CMD ["python", "-u", "/workspace/handler.py"]
