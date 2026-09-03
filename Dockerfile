FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# System dependencies
# =========================================================

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# Python dependencies
# =========================================================

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# =========================================================
# FASHN VTON v1.5
# =========================================================

RUN git clone --depth 1 \
    https://github.com/fashn-AI/fashn-vton-1.5.git \
    /workspace/fashn-vton

RUN pip install --no-cache-dir \
    -e /workspace/fashn-vton

# =========================================================
# Download VTON weights
# =========================================================

RUN mkdir -p /workspace/weights && \
    python /workspace/fashn-vton/scripts/download_weights.py \
    --weights-dir /workspace/weights

# =========================================================
# VASUNDHARA application
# =========================================================

COPY handler.py /workspace/handler.py
COPY vton /workspace/vton

# =========================================================
# RunPod Serverless
# =========================================================

CMD ["python", "-u", "/workspace/handler.py"]
