# ai-rtc-agent

# Overview

ai-rtc-agent is an experimental project for real-time video stream processing using AI models, hardware accelerated video decoding/encoding via NVDEC/NVENC and WebRTC.

The project relies on:

- A fork of [StreamDiffusion](https://github.com/yondonfu/StreamDiffusion/tree/deepstream) with the following changes:
  - Support for directly loading TensorRT engines without first loading base model weights.
  - Support for keeping output image tensors in CUDA memory so they can be passed directly to NVENC without incurring a CPU-GPU memory copy.
- A fork of [aiortc](https://github.com/yondonfu/aiortc/tree/nvcodec) with the following changes:
  - Support for NVDEC/NVENC decoding/encoding of h264 video streams.

This project only supports Linux + Nvidia GPUs and all testing thus far has been on a Nvidia RTX 4090.

# Table of Contents

1. [Setup](./docs/setup.md)
2. [Run](./docs/run.md)
3. [Deploy](./docs/deploy.md)
4. [Connect](./docs/connect.md)
5. [Troubleshoot](./docs/troubleshoot.md)