# Use an official NVIDIA CUDA runtime with development tools for compiling libraries
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set up system dependencies including git, python, and image handling tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /lib/apt/lists/*

# Set python3 as the default python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create working directory for the CatVTON repository architecture
WORKDIR /opt

# Clone the CatVTON framework repository
RUN git clone https://github.com /opt/CatVTON

# Set environment variables for app.py mapping
ENV CATVTON_ROOT=/opt/CatVTON

# Set main app working directory
WORKDIR /app

# Copy requirement files first to cache Docker layers optimally
COPY requirements.txt .

# Install Python requirements targeting correct CUDA wheels
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose your Flask port
EXPOSE 5000

# Start Gunicorn server with exactly 1 worker to manage GPU memory lock safely
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
