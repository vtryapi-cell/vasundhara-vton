FROM runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu24.04

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY requirements-serverless.txt /workspace/requirements-serverless.txt

RUN pip install --no-cache-dir -r /workspace/requirements-serverless.txt

RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /workspace/CatVTON

COPY handler.py /workspace/handler.py

WORKDIR /workspace/CatVTON

ENV PYTHONPATH=/workspace/CatVTON

CMD ["python", "/workspace/handler.py"]
