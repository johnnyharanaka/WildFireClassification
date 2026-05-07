"""
Histogram of IoU per image for the test set.

Loads a trained model, runs inference on Fire images from the test set,
computes per-image mean IoU (predicted bboxes vs GT), generates a histogram,
and copies images into bad/medium/good folders.

Usage:
    python utils/histogram_iou.py --model DinoV2RS_Small
"""

import os
import sys
import shutil
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config.config import get_config, get_active_dataset_config, log_info, set_root_path
from core.models import get_model, load_model, add_attn_maps
from core.datasets import get_data_loaders
from core.metrics import load_coco_bboxes_from_json, normalize_bbox, compute_iou
from core.visualization.attention_map import get_last_layer_attention
from core.visualization.vis_utils import VisUtils, compute_dynamic_threshold
from core.utils.debug import remove_nested_bboxes


def get_device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_image_mean_iou(gt_bboxes_coco, pred_bboxes):
    """Compute mean IoU for a single image.

    For each GT box, find the predicted box with the highest IoU.
    Return the mean of those best IoUs.

    Args:
        gt_bboxes_coco: List of GT bboxes in COCO format [x, y, w, h, class_id].
        pred_bboxes: List of predicted bboxes in corner format (x1, y1, x2, y2, ...).

    Returns:
        float: Mean IoU across all GT boxes.
    """
    if not gt_bboxes_coco:
        return 0.0

    best_ious = []
    for gt in gt_bboxes_coco:
        gt_x1, gt_y1, gt_x2, gt_y2 = normalize_bbox(gt[:4])

        best_iou = 0.0
        for pred in pred_bboxes:
            pred_box = [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])]
            iou = compute_iou([gt_x1, gt_y1, gt_x2, gt_y2], pred_box)
            best_iou = max(best_iou, iou)
        best_ious.append(best_iou)

    return float(np.mean(best_ious)) if best_ious else 0.0


def main():
    parser = argparse.ArgumentParser(description="Generate IoU histogram for test set.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. DinoV2RS_Small)")
    parser.add_argument("--root-path", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--output-dir", type=str, default="histogram_results", help="Output directory")
    args = parser.parse_args()

    model_name = args.model
    output_dir = args.output_dir

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device_auto()
    log_info(f"Using device: {device}")

    # Root path
    if args.root_path:
        set_root_path(args.root_path)

    root_path = get_active_dataset_config("data_path") or "data"
    num_classes = get_active_dataset_config("num_classes") or 2

    # Image size from config
    img_size = get_config("models", model_name, "img_size")
    img_size = int(img_size) if img_size is not None else 256

    # Load model
    log_info(f"Loading model: {model_name}")
    model = get_model(model_name, num_classes=num_classes, device=device)
    load_model(model, model_name)
    if "Dino" not in model_name:
        model = add_attn_maps(model)
    model.eval()

    # Load test data (batch_size=1 for per-image processing)
    data_loaders, _ = get_data_loaders(
        root_path=root_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
    )
    test_loader = data_loaders["Test"]

    # Load GT bboxes from COCO annotations
    test_path = os.path.join(root_path, "Test")
    gt_bboxes_by_filename = load_coco_bboxes_from_json(test_path, target_size=img_size)
    log_info(f"Loaded GT bboxes for {len(gt_bboxes_by_filename)} images")

    if not gt_bboxes_by_filename:
        log_info("No GT bboxes found. Exiting.")
        return

    # Min bbox area from config
    min_bbox_area = get_config("defaults", "min_bbox_area") or 1600

    # Collect per-image IoU
    image_ious = []  # (filename, mean_iou, image_path)
    total_processed = 0

    log_info(f"Processing test set images...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if len(batch_data) >= 3:
                images, labels, paths = batch_data[0], batch_data[1], batch_data[2]
            else:
                continue

            images = images.to(device)
            batch_size_actual = images.size(0)

            for i in range(batch_size_actual):
                label = labels[i].item() if isinstance(labels, torch.Tensor) else int(labels[i])

                # Only process Fire images (label == 0)
                if label != 0:
                    continue

                img_path = paths[i]
                filename = os.path.basename(img_path)

                # Check if GT bboxes exist for this image
                if filename not in gt_bboxes_by_filename:
                    continue

                gt_bboxes = gt_bboxes_by_filename[filename]

                # Extract attention map
                img_tensor = images[i:i + 1]
                try:
                    attn_np = get_last_layer_attention(model, model_name, img_tensor)
                except Exception as e:
                    log_info(f"Attention extraction failed for {filename}: {e}")
                    continue

                # Generate predicted bboxes
                threshold = compute_dynamic_threshold(attn_np, method="otsu", fire_adapted=True)
                bbs_raw = VisUtils.generate_bounding_box(attn_np, threshold=threshold, min_area=min_bbox_area)

                if len(bbs_raw) > 1:
                    bbs_raw = remove_nested_bboxes(bbs_raw, containment_threshold=0.5)

                if not bbs_raw:
                    # No prediction -> IoU = 0
                    image_ious.append((filename, 0.0, img_path))
                    total_processed += 1
                    continue

                # Compute mean IoU for this image
                mean_iou = compute_image_mean_iou(gt_bboxes, bbs_raw)
                image_ious.append((filename, mean_iou, img_path))
                total_processed += 1

            if (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
                log_info(f"  Processed {total_processed} Fire images so far...")

    log_info(f"Total Fire images processed: {total_processed}")

    if not image_ious:
        log_info("No Fire images with GT bboxes found. Exiting.")
        return

    # Extract IoU values
    ious = np.array([x[1] for x in image_ious])

    # --- Histogram ---
    os.makedirs(output_dir, exist_ok=True)

    bin_edges = np.arange(0, 1.1, 0.1)
    counts, _ = np.histogram(ious, bins=bin_edges)

    # Color mapping: red (0-0.3), yellow (0.3-0.7), green (0.7-1.0)
    colors = []
    for j in range(len(counts)):
        center = (bin_edges[j] + bin_edges[j + 1]) / 2
        if center < 0.3:
            colors.append("#d32f2f")  # red
        elif center < 0.7:
            colors.append("#fbc02d")  # yellow
        else:
            colors.append("#388e3c")  # green

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, counts, width=0.09, color=colors, edgecolor="black", linewidth=0.8)

    ax.set_xlabel("Mean IoU", fontsize=13)
    ax.set_ylabel("Number of Images", fontsize=13)
    ax.set_title(f"IoU Distribution - {model_name} (n={len(ious)})", fontsize=14, weight="bold")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(-0.05, 1.05)

    # Add count labels on top of bars
    for j, (c, cnt) in enumerate(zip(bin_centers, counts)):
        if cnt > 0:
            ax.text(c, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"histogram_iou_{model_name}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    log_info(f"Histogram saved to: {hist_path}")

    # --- Copy images into bad/medium/good folders ---
    for folder in ["bad", "medium", "good"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    bad_count, med_count, good_count = 0, 0, 0
    for filename, iou_val, img_path in image_ious:
        if iou_val < 0.3:
            dest = os.path.join(output_dir, "bad", filename)
            bad_count += 1
        elif iou_val < 0.7:
            dest = os.path.join(output_dir, "medium", filename)
            med_count += 1
        else:
            dest = os.path.join(output_dir, "good", filename)
            good_count += 1

        if os.path.exists(img_path):
            shutil.copy2(img_path, dest)

    # --- Summary ---
    log_info(f"\n{'=' * 60}")
    log_info(f"IoU HISTOGRAM SUMMARY - {model_name}")
    log_info(f"{'=' * 60}")
    log_info(f"Total Fire images: {len(ious)}")
    log_info(f"Mean IoU:   {ious.mean():.4f}")
    log_info(f"Median IoU: {np.median(ious):.4f}")
    log_info(f"Std IoU:    {ious.std():.4f}")
    log_info(f"Min IoU:    {ious.min():.4f}")
    log_info(f"Max IoU:    {ious.max():.4f}")
    log_info(f"-" * 60)
    log_info(f"Bad   (IoU < 0.3):  {bad_count} images")
    log_info(f"Medium (0.3-0.7):   {med_count} images")
    log_info(f"Good  (IoU >= 0.7): {good_count} images")
    log_info(f"-" * 60)
    log_info(f"Histogram: {hist_path}")
    log_info(f"Images copied to: {output_dir}/bad/, medium/, good/")
    log_info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
