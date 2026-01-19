# Cosmos-Reason2 HTTP Inference Server for VLA Pipeline

This directory contains the HTTP API wrapper for Cosmos-Reason2 that enables real-time Vision-Language-Action (VLA) inference for UAV control.

## Overview

The `cosmos_http_server.py` script wraps the Cosmos-Reason2 model inference in a Flask HTTP server, allowing the VLA pipeline to query the model for action decisions based on multi-camera views and state information.

## Features

- **Multi-Camera Support**: Accepts RGB images from front, left, and right cameras
- **Real-Time Inference**: Low-latency HTTP API for action decisions
- **Model Preloading**: Optional `--preload` flag to load model at startup
- **Robust Error Handling**: Automatic retries and graceful error recovery
- **Depth Integration**: Supports depth metrics in prompts for obstacle awareness

## Prerequisites

- Docker with GPU support (`--gpus all`)
- NVIDIA GPU with CUDA support
- HuggingFace token for model access
- PyTorch Docker image (or build from Dockerfile)

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

The `--preload` flag loads the model at startup (recommended to avoid first-request delay).

## API Endpoints

### POST `/inference`

Main inference endpoint for VLA action decisions.

**Request Format:**
```json
{
  "prompt": "Mission: Navigate forward while avoiding obstacles\nCurrent State: Position: N=0.00m E=0.00m Alt=5.00m | Velocity: 0.00m/s | Yaw=0.0°\nAvailable Actions: FORWARD, BACK, LEFT, RIGHT, YAW_LEFT, YAW_RIGHT, UP, DOWN, STOP, HOLD\nAction:",
  "images": [
    {
      "name": "front",
      "data": "<base64_encoded_jpeg>",
      "mime_type": "image/jpeg"
    },
    {
      "name": "left",
      "data": "<base64_encoded_jpeg>",
      "mime_type": "image/jpeg"
    },
    {
      "name": "right",
      "data": "<base64_encoded_jpeg>",
      "mime_type": "image/jpeg"
    }
  ],
  "max_tokens": 50,
  "temperature": 0.1
}
```

**Response Format:**
```json
{
  "response": "FORWARD"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "cosmos-reason2-inference",
  "model_loaded": true
}
```

## Integration with VLA Pipeline

This server is designed to work with the PegasusSimulator VLA pipeline:

1. **VLA Client** (`src/pegasus/vla/vla_client.py`) sends requests to this endpoint
2. **Multi-camera images** are encoded as base64 and sent in the request
3. **Prompt includes** mission intent, current state, and action space
4. **Model responds** with a single action (e.g., "FORWARD", "LEFT", "STOP")
5. **VLA Controller** processes the action through safety gates before sending to PX4

## Configuration

### Command-Line Arguments

```bash
python scripts/cosmos_http_server.py [OPTIONS]

Options:
  --host HOST          Host to bind to (default: 0.0.0.0)
  --port PORT          Port to bind to (default: 8000)
  --debug              Enable Flask debug mode
  --preload            Preload model at startup (recommended)
```

### Model Configuration

The server uses `nvidia/Cosmos-Reason2-2B` by default. Model configuration is in the `load_model()` function:

- **Model**: `nvidia/Cosmos-Reason2-2B`
- **Dtype**: `torch.float16` (for GPU memory efficiency)
- **Device**: Auto (uses available GPUs)
- **Vision Tokens**: 256-8192 (configurable)

## Architecture

```
┌─────────────────┐
│  Isaac Sim      │
│  (3 cameras)    │──┐
└─────────────────┘  │
                     │
┌─────────────────┐  │  HTTP POST
│  VLA Pipeline   │──┼──────────────┐
│  (vla_client)   │  │              │
└─────────────────┘  │              ▼
                     │      ┌──────────────────┐
┌─────────────────┐  │      │  Cosmos-R2      │
│  PX4/MAVSDK     │  │      │  HTTP Server    │
│  (offboard)     │◄─┼──────│  (this repo)    │
└─────────────────┘         └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  Cosmos-Reason2  │
                            │  Model (GPU)     │
                            └──────────────────┘
```

## Troubleshooting

### Model Loading Issues

- **Out of Memory**: Reduce `max_vision_tokens` or use a smaller model variant
- **HF Token Error**: Ensure `HF_TOKEN` environment variable is set correctly
- **Download Timeout**: Model download may take time on first run (2B model ~4GB)

### Connection Issues

- **Connection Refused**: Verify port 8000 is exposed (`-p 8000:8000` in Docker)
- **Timeout Errors**: Increase `--timeout` in VLA client if model inference is slow
- **GPU Not Available**: Check `nvidia-smi` and ensure `--gpus all` is used

### Performance

- **First Request Slow**: Use `--preload` to load model at startup
- **Inference Speed**: Typically 1-3 seconds per request depending on GPU
- **Memory Usage**: ~8-12GB GPU memory for Cosmos-Reason2-2B

## Files

- `scripts/cosmos_http_server.py` - Main HTTP server script
- `scripts/inference_sample.py` - Original inference example (reference)
- `Dockerfile` - Optional: build custom Docker image

## License

See main project LICENSE file.

## Notes

- The server runs in development mode (Flask). For production, use a WSGI server like Gunicorn.
- Model is loaded once and reused for all requests (efficient for real-time inference).
- Multi-camera images are processed in order: front, left, right (matching training format).
