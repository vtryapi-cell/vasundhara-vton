FROM runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------
# Install serverless dependencies
# ---------------------------------------------------------

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir --ignore-installed \
    -r /workspace/requirements-serverless.txt

# ---------------------------------------------------------
# Download CatVTON
# ---------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/Zheng-Chong/CatVTON.git \
    /workspace/CatVTON

# ---------------------------------------------------------
# CatVTON paths
# ---------------------------------------------------------

ENV CATVTON_ROOT=/workspace/CatVTON
ENV PYTHONPATH=/workspace/CatVTON

# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

COPY handler.py /workspace/handler.py

WORKDIR /workspace/CatVTON

CMD ["python", "/workspace/handler.py"]
