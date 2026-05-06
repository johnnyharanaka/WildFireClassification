#!/usr/bin/env python3
"""
Converts HuggingFace DINOv2 Remote Sensing model from safetensors to PyTorch .pth format.

Usage:
    python utils/convert_model_rs.py
    python utils/convert_model_rs.py --model-id KevinCha/dinov2-vit-small-remote-sensing-100ep
    python utils/convert_model_rs.py --safetensors model.safetensors --output model.pth
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from core.config.config import log_info


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL_ID = "KevinCha/dinov2-vit-small-remote-sensing-100ep"
DEFAULT_OUTPUT = "dinov2_vits14_reg4_pretrain_rs_100ep.pth"


# =============================================================================
# Download
# =============================================================================

def download_model(model_id: str = DEFAULT_MODEL_ID, cache_dir: str = None) -> str:
    """
    Download safetensors model from HuggingFace Hub.

    Returns:
        Path to downloaded .safetensors file
    """
    log_info(f"Downloading: {model_id}")

    filepath = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        cache_dir=cache_dir
    )

    log_info(f"Downloaded to: {filepath}")
    return filepath


# =============================================================================
# Conversion
# =============================================================================

def convert_safetensors_to_pth(safetensors_path: str, output_path: str, device: str = "cpu"):
    """
    Convert safetensors to PyTorch .pth format.

    Args:
        safetensors_path: Input .safetensors file
        output_path: Output .pth file
        device: Device for loading ('cpu' or 'cuda')
    """
    log_info(f"\nLoading: {safetensors_path}")
    state_dict = load_safetensors(safetensors_path)

    converted = {}
    removed = []

    for key, value in state_dict.items():
        # Skip teacher model and dino_loss
        if key.startswith('teacher.') or key.startswith('dino_loss.'):
            removed.append(key)
            continue

        # Remove student prefixes
        new_key = key.replace('student.backbone.', '').replace('student.', '')
        converted[new_key] = value
        log_info(f"  {key} -> {new_key}")

    if removed:
        log_info(f"\nRemoved {len(removed)} keys (teacher/dino_loss)")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"\nSaving: {output_path}")
    torch.save(converted, output_path, pickle_protocol=4)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    log_info(f"Saved ({size_mb:.2f} MB, {len(converted)} parameters)")

    # Verify
    log_info("Verifying...")
    loaded = torch.load(output_path, map_location=device, weights_only=False)
    log_info(f"Verified: {len(loaded)} parameters loaded")


# =============================================================================
# Main
# =============================================================================

def main():
    # No arguments: run with defaults
    if len(sys.argv) == 1:
        log_info("=" * 60)
        log_info("Converting DINOv2 Remote Sensing model")
        log_info("=" * 60)
        safetensors_path = download_model()
        convert_safetensors_to_pth(safetensors_path, DEFAULT_OUTPUT)
        log_info("\n" + "=" * 60)
        log_info("Conversion complete!")
        log_info("=" * 60)
        return

    parser = argparse.ArgumentParser(description="Convert HuggingFace safetensors to PyTorch .pth")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .pth path")
    parser.add_argument("--safetensors", default=None, help="Existing .safetensors file (skip download)")
    parser.add_argument("--no-download", action="store_true", help="Don't download, use --safetensors")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for loading")

    args = parser.parse_args()

    log_info("=" * 60)
    log_info("Converting DINOv2 Remote Sensing model")
    log_info("=" * 60)

    # Get safetensors path
    if args.safetensors and os.path.exists(args.safetensors):
        safetensors_path = args.safetensors
        log_info(f"Using: {safetensors_path}")
    elif not args.no_download:
        safetensors_path = download_model(args.model_id, args.cache_dir)
    else:
        log_info("Error: No safetensors file and --no-download set")
        sys.exit(1)

    try:
        convert_safetensors_to_pth(safetensors_path, args.output, args.device)
        log_info("\n" + "=" * 60)
        log_info("Conversion complete!")
        log_info("=" * 60)
    except Exception as e:
        log_info(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
