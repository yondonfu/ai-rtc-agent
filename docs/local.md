# Run Locally

# Install

## System Dependencies

```
sudo apt install libopus-dev libvpx-dev
```

Note: These are required by `aiortc`.

## Python Dependencies

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

# Download Models

```
# Keep all model files under the `models` directory
export HF_HOME=./models
export HF_HUB_CACHE=./models/hub

python download.py
```

# Build TensorRT Engines

```
python build.py
```  

The default directory for engine plan files is `./models/engines` and can be modified using the `TRT_ENGINES_CACHE` env variable.

# Run

The following will start a publicly accessible server listening on port 8888 (default).

```
python agent.py
```