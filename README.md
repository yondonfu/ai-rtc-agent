# ai-rtc-agent

# Table of Contents

1. [Overview](#overview)
2. [Pre-Install](#pre-install)
3. [Install](#install)
4. [Download Models](#download-models)
5. [Build TensorRT Engines](#build-tensorrt-engines)
6. [Run](#run)

# Overview

ai-rtc-agent is an experimental project for real-time video stream processing using AI models, hardware accelerated video decoding/encoding via NVDEC/NVENC and WebRTC.

The project relies on:

- A fork of [StreamDiffusion](https://github.com/yondonfu/StreamDiffusion/tree/deepstream) with the following changes:
  - Support for directly loading TensorRT engines without first loading base model weights.
  - Support for keeping output image tensors in CUDA memory so they can be passed directly to NVENC without incurring a CPU-GPU memory copy.
- A fork of [aiortc](https://github.com/yondonfu/aiortc/tree/nvcodec) with the following changes:
  - Support for NVDEC/NVENC decoding/encoding of h264 video streams.

This project only supports Linux + Nvidia GPUs and all testing thus far has been on a Nvidia RTX 4090.

# Pre-Install

## Nvidia Drivers

Make sure you have `libnvidia-encode-*` and `libnvidia-decode-*` installed.

```
# Replace 535 with your desired version
sudo apt install nvidia-driver-535 libnvidia-encode-535 libnvidia-decode-535
```

If you have an existing driver installation and encounter issues with the above you can try removing the existing installation first:

```
sudo apt install nvidia-* libnvidia-*
sudo apt autoremove
```

## Nvidia Container Runtime

If you are using Docker, install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

# Install

## Local

### System Dependencies

```
sudo apt install libopus-dev libvpx-dev
```

Note: These are required by `aiortc`.

### Python Dependencies

Download [CV-CUDA](https://github.com/CVCUDA/CV-CUDA) wheel:

```
curl -L 'https://github.com/CVCUDA/CV-CUDA/releases/download/v0.11.0-beta/cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl' > cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl
```

Install PyTorch:

```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

Install other Python dependencies:

```
pip install -r requirements.txt
```

Install TensorRT dependencies for StreamDiffusion:

```
python -m streamdiffusion.tools.install-tensorrt
```

## Docker

```
docker build -t ai-rtc-agent:latest .
```

# Download Models

## Local

```
# Keep all model files under the `models` directory
export HF_HOME=./models
export HF_HUB_CACHE=./models/hub

python download.py
```

The default directory for CivitAI model files is `./models/civitai` and can be modified using the `CIVITAI_CACHE` env variable.

## Docker

```
docker run -v ./models:/models ai-rtc-agent:latest python download.py
```

# Build TensorRT Engines

## Local

```
python build.py
```  

The default directory for engine plan files is `./models/engines` and can be modified using the `TRT_ENGINES_CACHE` env variable.

## Docker

```
docker run --gpus all -v ./models:/models ai-rtc-agent:latest python build.py
```

# Run

The following will start a publicly accessible server listening on port 8888 (default).

## Local

```
python agent.py
```

## Docker

```
docker run --gpus all --network="host" -v ./models:/models ai-rtc-agent:latest
```

## WebRTC Connection Issues

TODO

# Environment Variables

`AUTH_TOKEN`

`WEBHOOK_URL`

`TWILIO_ACCOUNT_SID`

`TWILIO_AUTH_TOKEN`

`WARMUP_FRAMES`

`TRT_ENGINES_CACHE`

`CIVITAI_CACHE`

`NVENC_PRESET`

`NVENC_TUNING_INFO`

`NVENC_DEFAULT_BITRATE`

`NVENC_MIN_BITRATE`

`NVENC_MAX_BITRATE`