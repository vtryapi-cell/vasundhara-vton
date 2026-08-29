FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# Force compatible NumPy
# ---------------------------------------------------------

RUN pip uninstall -y numpy || true

RUN pip install --no-cache-dir \
    numpy==1.26.4

# ---------------------------------------------------------
# Verify NumPy
# ---------------------------------------------------------

RUN python -c "import numpy; print('NumPy:', numpy.__version__); print('NumPy OK')"

# ---------------------------------------------------------
# Download CatVTON
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# ---------------------------------------------------------
# CatVTON configuration
# ---------------------------------------------------------

ENV CATVTON_ROOT=/workspace/CatVTON
ENV PYTHONPATH=/workspace/CatVTON

# ---------------------------------------------------------
# Handler
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py

WORKDIR /workspace/CatVTON

# ---------------------------------------------------------
# Start worker
# ---------------------------------------------------------

CMD ["python", "/workspace/handler.py"]
