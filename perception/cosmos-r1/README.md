# Cosmos-Reason2 Inference for VLA 

Clone and follow the instructions to use the model: ![NVIDIA Cosmos-R2](https://github.com/nvidia-cosmos/cosmos-reason2)

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

### 2. Run The Inference Script

```bash
# Inside container
python scripts/inference_sample.py
```
