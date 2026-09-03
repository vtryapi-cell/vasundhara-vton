FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
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
# Python dependencies
# IMPORTANT:
# Do NOT install another torch version.
# Base image already provides PyTorch 2.8 + CUDA 12.8.
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py
COPY vton /workspace/vton

# ---------------------------------------------------------
# Start RunPod worker
# ---------------------------------------------------------

CMD ["python", "-u", "/workspace/handler.py"]
