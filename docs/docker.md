# Run with Docker

# Install

## Nvidia Container Runtime

Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Build

```
docker build -t ai-rtc-agent:latest .
```

# Download Models

```
docker run -v ./models:/models ai-rtc-agent:latest python download.py
```

# Build TensorRT Engines

```
docker run --gpus all -v ./models:/models ai-rtc-agent:latest python build.py
```

# Run

The following will start a publicly accessible server listening on port 8888 (default).

```
docker run --gpus all --network="host" -v ./models:/models ai-rtc-agent:latest
```
