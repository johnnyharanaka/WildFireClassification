"""
Compare Attention Rollout vs Last-Layer Attention visualization.
"""
# ============================================================================
# CONFIG
# ============================================================================
MAX_IMAGES = 40
MODEL_NAME = "DinoV2RS_Small"
# ============================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from core.models import get_model, load_model
from core.datasets import get_data_loaders
from core.config.config import get_config, get_active_dataset_config
from core.attention_rollout import compute_attention_rollout
from core.attention_map import get_last_layer_attention


def colorize_heatmap(heatmap):
    """Apply JET colormap to heatmap."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def main():
    num_classes = 2
    model_name = MODEL_NAME
    max_images = MAX_IMAGES
    output_dir = "attention_comparison"

    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model = get_model(model_name, num_classes=num_classes, device=device)
    load_model(model, model_name)
    model.eval()

    # Load data
    root_path = get_active_dataset_config('data_path')
    img_size = get_config("models", model_name, "img_size") or 224
    img_size = int(img_size)

    data_loaders, _ = get_data_loaders(root_path, batch_size=1, num_workers=0, img_size=img_size)

    # Denormalization params
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    saved_files = []
    count = 0
    for features, labels, paths in data_loaders["Test"]:
        if count >= max_images:
            break

        # Only Fire class (label 0)
        if labels[0].item() != 0:
            continue

        features = features.to(device)
        feature = features[0]
        feature_unsqueeze = feature.unsqueeze(0)

        # Get original image
        img_np = feature.cpu().permute(1, 2, 0).numpy() * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
        img_np = np.clip(img_np, 0, 1)

        # Get Last-Layer Attention
        heatmap_last_layer = get_last_layer_attention(model, model_name, feature_unsqueeze)

        # Get Attention Rollout
        heatmap_rollout = compute_attention_rollout(model, model_name, feature_unsqueeze)

        # Colorize
        heatmap_last_layer_color = colorize_heatmap(heatmap_last_layer)
        heatmap_rollout_color = colorize_heatmap(heatmap_rollout)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(heatmap_last_layer_color)
        axes[1].set_title('Last-Layer Attention', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(heatmap_rollout_color)
        axes[2].set_title('Attention Rollout', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(output_dir, f"comparison_{count+1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(save_path)

        count += 1
        print(f"Saved {count}/{max_images}: {save_path}")

    # Open all images in Preview (macOS)
    subprocess.run(["open"] + saved_files)


if __name__ == "__main__":
    main()
