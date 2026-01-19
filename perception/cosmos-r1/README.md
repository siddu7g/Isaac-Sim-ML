# Cosmos-Reason2 Inference for VLA 

## Quick Start

### 1. Start Docker Container

```bash
docker run -it --gpus all --ipc=host --rm \
  -e HF_TOKEN="your_huggingface_token_here" \
  -v $HOME/cosmos-reason2:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  -p 8000:8000 \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime bash
```

### 2. Install Dependencies

```bash
# Inside container
pip install flask transformers torch accelerate pillow aiohttp
```

### 3. Run HTTP Server

```bash
# Inside container
python scripts/cosmos_http_server.py --port 8000 --host 0.0.0.0 --preload
```
