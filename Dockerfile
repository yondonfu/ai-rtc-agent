FROM nvidia/cuda:12.1.1-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install prerequisites
RUN apt-get update && \
  apt-get install -y --no-install-recommends build-essential libssl-dev zlib1g-dev libbz2-dev \
  xz-utils libffi-dev liblzma-dev curl python3-openssl git ffmpeg libopus-dev libvpx-dev && \
  rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install your desired Python version
ARG PYTHON_VERSION=3.10
RUN pyenv install $PYTHON_VERSION && \
  pyenv global $PYTHON_VERSION && \
  pyenv rehash

RUN curl -L 'https://github.com/CVCUDA/CV-CUDA/releases/download/v0.11.0-beta/cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl' > cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl

# Install deps
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m streamdiffusion.tools.install-tensorrt

FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

WORKDIR /app

# Install necessary runtime libraries
RUN apt-get update && \
  apt-get install -y --no-install-recommends libopus-dev libvpx-dev ffmpeg && \
  rm -rf /var/lib/apt/lists/*

# Copy Python environment and installed packages
COPY --from=builder /root/.pyenv /root/.pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV HF_HOME=/models
ENV HF_HUB_CACHE=/models/hub
ENV CIVITAI_CACHE=/models/civitai
ENV TRT_ENGINES_CACHE=/models/engines
# Enable NVENC in aiortc
ENV NVENC=true
# Enable NVDEC in aiortc
ENV NVDEC=true
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Copy necessary files
COPY lib /app/lib
COPY download.py /app/download.py
COPY build.py /app/build.py
COPY agent.py /app/agent.py

CMD ["python", "agent.py"]
