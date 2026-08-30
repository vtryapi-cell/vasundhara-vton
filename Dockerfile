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
# APPLICATION DEPENDENCIES
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN python -m pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# FORCE NUMPY
# ---------------------------------------------------------

RUN python -m pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4

# ---------------------------------------------------------
# CATVTON
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# ---------------------------------------------------------
# CATVTON DEPENDENCIES
# ---------------------------------------------------------

RUN python -m pip install --no-cache-dir \
    fvcore==0.1.5.post20221221 \
    cloudpickle==3.0.0 \
    omegaconf==2.3.0 \
    pycocotools==2.0.8 \
    PyYAML==6.0.1 \
    tqdm==4.66.4 \
    "peft>=0.17.0"

# ---------------------------------------------------------
# COPY HANDLER
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py

# ---------------------------------------------------------
# WORKING DIRECTORY
# ---------------------------------------------------------

WORKDIR /workspace

# ---------------------------------------------------------
# START SERVERLESS WORKER
# ---------------------------------------------------------

CMD ["python", "-u", "/workspace/handler.py"]
