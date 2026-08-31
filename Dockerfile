FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

# Remove potentially conflicting preinstalled packages
RUN pip uninstall -y \
    numpy \
    opencv-python \
    opencv-python-headless \
    matplotlib \
    2>/dev/null || true

# Install exact NumPy first
RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4

# Install application dependencies
RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# Make sure NumPy remains the required version
RUN pip install --no-cache-dir --force-reinstall \
    numpy==1.26.4

# ---------------------------------------------------------
# Copy application
# ---------------------------------------------------------

COPY . /workspace

# ---------------------------------------------------------
# Startup import test
# ---------------------------------------------------------

RUN python -c "import numpy; print('NUMPY:', numpy.__version__)"
RUN python -c "import torch; print('TORCH:', torch.__version__)"
RUN python -c "import diffusers; print('DIFFUSERS:', diffusers.__version__)"
RUN python -c "import transformers; print('TRANSFORMERS:', transformers.__version__)"

# ---------------------------------------------------------
# RunPod serverless worker
# ---------------------------------------------------------

CMD ["python", "handler.py"]
