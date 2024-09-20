# ai-rtc-agent

# Overview

ai-rtc-agent is an experimental project for real-time video stream processing using AI models, hardware accelerated video decoding/encoding via NVDEC/NVENC and WebRTC.

The first (and only for now) supported pipeline is per-frame image2image using diffusion models with the [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) framework. The pipeline is used in the [aifluidsim](https://www.aifluidsim.com/) app which accepts a stream URL which can be backed by a self-hosted instance of ai-rtc-agent. The easiest way to get a stream URL is to follow the steps for [deploying on Runpod](./docs/runpod.md) and [connecting](./docs/connect.md). If you want to run and deploy the agent elsewhere, refer to the Table of Contents below.

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