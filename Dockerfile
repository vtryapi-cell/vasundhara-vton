FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# System packages
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
# JCo-MVTON engine source
# =========================================================

RUN git clone --depth 1 \
    https://github.com/damo-cv/JCo-MVTON.git \
    /workspace/jco-mvton

# =========================================================
# Modified Diffusers 0.33.0
# =========================================================

RUN git clone --depth 1 --branch v0.33.0 \
    https://github.com/huggingface/diffusers.git \
    /workspace/diffusers

RUN cp \
    /workspace/jco-mvton/flux/modeling_utils.py \
    /workspace/diffusers/src/diffusers/models/modeling_utils.py

RUN pip install --no-cache-dir \
    /workspace/diffusers

# =========================================================
# Application
# =========================================================

COPY handler.py /workspace/handler.py
COPY vton /workspace/vton

# =========================================================
# RunPod worker
# =========================================================

CMD ["python", "-u", "/workspace/handler.py"]
