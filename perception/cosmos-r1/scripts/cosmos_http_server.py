#!/usr/bin/env python3
"""
cosmos_http_server.py: HTTP API wrapper for Cosmos-Reason2 inference

This script wraps the Cosmos-Reason2 inference logic in an HTTP server
that the VLA pipeline can call.

Usage (inside Cosmos-R1 Docker container):
    python cosmos_http_server.py --port 8000 --host 0.0.0.0
"""

import argparse
import json
import base64
import io
import warnings
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image
import torch
import transformers

warnings.filterwarnings("ignore")

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[ERROR] Flask not installed. Install with: pip install flask")

PIXELS_PER_TOKEN = 32**2


# Global model and processor (loaded once at startup)
_model = None
_processor = None


def load_model():
    """Load Cosmos-Reason2 model and processor (call once at startup)"""
    global _model, _processor
    
    if _model is not None:
        return _model, _processor
    
    print("[MODEL] Loading Cosmos-Reason2 model...")
    model_name = "nvidia/Cosmos-Reason2-2B"
    
    _model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    _processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
    
    # Optional: Limit vision tokens
    min_vision_tokens = 256
    max_vision_tokens = 8192
    _processor.image_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }
    
    print("[MODEL] ✅ Model loaded successfully")
    return _model, _processor


def decode_image_from_base64(img_b64: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes))
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def numpy_to_pil_image(img_array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    # Ensure uint8 [0, 255]
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # Ensure RGB format
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        return Image.fromarray(img_array, mode="RGB")
    elif len(img_array.shape) == 2:
        # Grayscale -> RGB
        return Image.fromarray(img_array, mode="L").convert("RGB")
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")


def create_conversation_for_action(
    prompt: str,
    images: Dict[str, np.ndarray],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create conversation format for Cosmos-Reason2.
    
    Args:
        prompt: User prompt/instruction
        images: Dict of {camera_name: numpy_array} images
        system_prompt: Optional system prompt (default: UAV control assistant)
    
    Returns:
        Conversation list in Qwen3VL format
    """
    if system_prompt is None:
        system_prompt = (
            "You are a UAV navigation assistant. Analyze the camera views and "
            "determine the best action for safe navigation and obstacle avoidance. "
            "Respond with a single action word."
        )
    
    # Build content: images first (IMPORTANT: media before text per training format)
    content = []
    
    # Add images in order: front, left, right (or any order)
    camera_order = ["front", "left", "right"] if all(c in images for c in ["front", "left", "right"]) else list(images.keys())
    
    for cam_name in camera_order:
        if cam_name in images:
            img_pil = numpy_to_pil_image(images[cam_name])
            # Store temporarily or convert to format processor expects
            # The processor can handle PIL Images directly
            content.append({
                "type": "image",
                "image": img_pil,
            })
    
    # Add text prompt after images
    content.append({
        "type": "text",
        "text": prompt,
    })
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    
    return conversation


def run_inference(
    conversation: List[Dict[str, Any]],
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> str:
    """
    Run Cosmos-Reason2 inference on conversation.
    
    Returns:
        Generated text response
    """
    model, processor = load_model()
    
    # Process inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Run inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    return output_text[0] if output_text else ""


if FLASK_AVAILABLE:
    app = Flask(__name__)

    @app.route('/inference', methods=['POST'])
    def inference_endpoint():
        """
        HTTP endpoint for Cosmos-Reason2 inference.
        
        Expected request format:
        {
            "prompt": "Mission: ... Current State: ... Available Actions: ...",
            "image": {"data": "<base64>", "mime_type": "image/jpeg"},  # Optional (backward compat)
            "images": [{"name": "front", "data": "<base64>", ...}, ...],  # Multi-camera
            "max_tokens": 50,
            "temperature": 0.1,
        }
        
        Returns:
        {
            "response": "FORWARD",  # Action text
        }
        """
        try:
            data = request.json
            
            # Extract prompt
            prompt = data.get("prompt", "")
            if not prompt:
                return jsonify({"error": "Missing 'prompt' in request"}), 400

            # Extract images (support both single and multi-camera)
            images_dict = {}
            if "images" in data and isinstance(data["images"], list):
                # Multi-camera: list of {"name": "...", "data": "...", "mime_type": "..."}
                for img_obj in data["images"]:
                    cam_name = img_obj.get("name", "unknown")
                    img_b64 = img_obj.get("data", "")
                    if img_b64:
                        images_dict[cam_name] = decode_image_from_base64(img_b64)
            elif "image" in data:
                # Single image (backward compat)
                img_obj = data["image"]
                img_b64 = img_obj.get("data", "")
                if img_b64:
                    images_dict["front"] = decode_image_from_base64(img_b64)

            if not images_dict:
                return jsonify({"error": "No images provided"}), 400

            # Extract generation parameters
            max_tokens = data.get("max_tokens", 50)
            temperature = float(data.get("temperature", 0.1))

            # Create conversation
            conversation = create_conversation_for_action(prompt, images_dict)

            # Run inference
            response_text = run_inference(conversation, max_new_tokens=max_tokens, temperature=temperature)

            return jsonify({
                "response": response_text.strip(),
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "response": "STOP"  # Fallback action
            }), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        model_loaded = _model is not None
        return jsonify({
            "status": "ok",
            "service": "cosmos-reason2-inference",
            "model_loaded": model_loaded,
        })


def main():
    parser = argparse.ArgumentParser(description="Cosmos-Reason2 HTTP Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--preload", action="store_true", help="Preload model at startup")
    args = parser.parse_args()

    if not FLASK_AVAILABLE:
        print("[ERROR] Flask is required. Install with: pip install flask")
        return

    # Preload model if requested
    if args.preload:
        print("[SERVER] Preloading model...")
        load_model()
        print("[SERVER] ✅ Model preloaded")

    print(f"[SERVER] Starting Cosmos-Reason2 HTTP inference server on {args.host}:{args.port}")
    print(f"[SERVER] Endpoints:")
    print(f"[SERVER]   POST /inference  - Run inference")
    print(f"[SERVER]   GET  /health     - Health check")
    print(f"[SERVER] Note: Model will be loaded on first request unless --preload is used")
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
