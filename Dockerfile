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
# Do not upgrade/reinstall system cryptography.
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir --ignore-installed \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# CatVTON
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# ---------------------------------------------------------
# Worker
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py

ENV CATVTON_ROOT=/workspace/CatVTON

CMD ["python", "-u", "/workspace/handler.py"]
