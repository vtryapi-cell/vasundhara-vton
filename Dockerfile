FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV CATVTON_ROOT=/opt/CatVTON

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install the pinned Python dependencies first.
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Bring in the official CatVTON source.
RUN git clone --depth 1 https://github.com/Zheng-Chong/CatVTON.git /opt/CatVTON

COPY app.py .

# Keep the existing Vasundhara website files if they are present in the repo.
COPY templates ./templates
COPY static ./static

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "1", "--timeout", "900", "app:app"]
