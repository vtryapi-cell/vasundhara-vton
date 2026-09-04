FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# SYSTEM DEPENDENCIES
# =========================================================

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# PYTHON DEPENDENCIES
# =========================================================

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

# Install application dependencies without touching the
# system cryptography package.
RUN pip install --no-cache-dir --ignore-installed \
    -r /workspace/requirements-serverless.txt

# =========================================================
# FIX NUMPY / SCIPY BINARY COMPATIBILITY
# =========================================================

RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4 \
    scipy==1.14.1

# Verify NumPy + SciPy before building the worker
RUN python -c "import numpy; print('NumPy:', numpy.__version__); import scipy; print('SciPy:', scipy.__version__); import scipy.sparse; print('NumPy/SciPy OK')"

# =========================================================
# CATVTON
# =========================================================

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# =========================================================
# WORKER
# =========================================================

COPY handler.py /workspace/handler.py

ENV CATVTON_ROOT=/workspace/CatVTON

# =========================================================
# RUNPOD SERVERLESS
# =========================================================

CMD ["python", "-u", "/workspace/handler.py"]
