# Custom base for Wan/ComfyUI RunPod workers
# Slim CUDA 12.8 runtime + Python 3.10 + PyTorch — no models, no ComfyUI
# (runtime image is much smaller than cudnn-devel)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]
WORKDIR /

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
      git wget curl ca-certificates \
      bash libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      build-essential \
      ffmpeg git-lfs \
      python3.10 python3.10-dev python3.10-venv python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN pip install -U pip setuptools wheel packaging

# PyTorch cu128 (matches CUDA 12.8 / RunPod)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
      runpod \
      websocket-client \
      "huggingface_hub[hf_transfer]" \
      requests \
      numpy \
      pillow \
      opencv-python-headless \
      scipy \
      einops \
      safetensors \
      aiohttp \
      pyyaml \
      tqdm

WORKDIR /
CMD ["/bin/bash"]
