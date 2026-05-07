"""
Data curation - filters bad images based on the model's localization quality.

Given a model and its weights, runs prediction on the specified split, computes IoU
between predicted bboxes and ground truth, and saves only images with IoU above
the threshold in a new folder keeping the same dataset structure.

Usage:
    python run_curacao_dados.py --model DinoV2RS_Small --weights models/DinoV2RS_Small.pth
    python run_curacao_dados.py --model DinoV2RS_Small --weights models/DinoV2RS_Small.pth --split Val --iou_threshold 0.4
    python run_curacao_dados.py --model DinoV2RS_Small --weights models/DinoV2RS_Small.pth --split Test --output data_fire_curated
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from core.config.config import get_config, get_active_dataset_config, log_info
from core.models import get_model
from core.datasets import get_data_loaders
from core.visualization.attention_map import get_last_layer_attention
from core.visualization.vis_utils import VisUtils, compute_dynamic_threshold
from core.metrics import load_coco_bboxes_from_json


def compute_best_iou(gt_bboxes_coco, pred_bboxes):
    """
    Compute the best IoU across all GT x Prediction combinations.

    Args:
        gt_bboxes_coco: List of GT bboxes in COCO format [x, y, w, h, class_id]
        pred_bboxes: List of predicted bboxes in corner format (x1, y1, x2, y2, conf)

    Returns:
        float: Best IoU found
    """
    best_iou = 0.0
    for gt in gt_bboxes_coco:
        # Converte COCO [x, y, w, h] -> corners [x1, y1, x2, y2]
        gt_x1, gt_y1 = gt[0], gt[1]
        gt_x2, gt_y2 = gt_x1 + gt[2], gt_y1 + gt[3]
        gt_corners = (gt_x1, gt_y1, gt_x2, gt_y2)

        for pred in pred_bboxes:
            iou = VisUtils.get_IoU(gt_corners, pred[:4])
            best_iou = max(best_iou, iou)

    return best_iou


def main():
    parser = argparse.ArgumentParser(
        description="Data curation - filters bad images by IoU"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g.: DinoV2RS_Small)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the weights file (e.g.: models/DinoV2RS_Small.pth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        help="Split to curate (Test or Val). Default: Test",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="Minimum IoU to keep the image. Default: 0.3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: data_fire_curated",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference. Default: 8",
    )
    parser.add_argument(
        "--keep_not",
        action="store_true",
        default=True,
        help="Keep 'Not' class images (no fire). Default: True",
    )
    parser.add_argument(
        "--no_keep_not",
        action="store_true",
        help="Don't keep 'Not' images (save only filtered Fire)",
    )
    args = parser.parse_args()

    if args.no_keep_not:
        args.keep_not = False

    if args.output is None:
        args.output = f"data_fire_curated_iou{args.iou_threshold}"

    # ── Setup ──
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log_info(f"Device: {device}")

    # ── Load model ──
    log_info(f"Loading model {args.model} with weights from {args.weights}")
    model = get_model(args.model, num_classes=2, device=device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # ── Config ──
    img_size = int(get_config("models", args.model, "img_size") or 256)
    root_path = get_active_dataset_config("data_path") or "data_fire"
    split_path = os.path.join(root_path, args.split)
    annotations_path = os.path.join(split_path, "annotations.json")

    if not os.path.exists(annotations_path):
        log_info(
            f"ERROR: {annotations_path} not found. This split has no COCO annotations."
        )
        return

    log_info(
        f"Split: {args.split} | img_size: {img_size} | IoU threshold: {args.iou_threshold}"
    )

    # ── Load COCO annotations ──
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # GT bboxes scaled to img_size (COCO format [x, y, w, h, class_id])
    gt_bboxes = load_coco_bboxes_from_json(split_path, target_size=img_size)
    log_info(f"GT bboxes loaded for {len(gt_bboxes)} images (Fire)")

    # Mappings
    filename_to_imginfo = {img["file_name"]: img for img in coco_data["images"]}
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    images_with_anns = set(ann["image_id"] for ann in coco_data["annotations"])

    # Annotations grouped by image_id
    anns_by_image_id = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image_id:
            anns_by_image_id[img_id] = []
        anns_by_image_id[img_id].append(ann)

    # ── Load dataloader ──
    data_loaders, datasets = get_data_loaders(
        root_path=root_path,
        batch_size=args.batch_size,
        num_workers=0,
        img_size=img_size,
    )
    data_loader = data_loaders[args.split]

    # ── Bbox config ──
    min_bbox_area = get_config("defaults", "min_bbox_area") or 1600

    # ── Process images ──
    kept_fire = []  # (filename, iou)
    removed_fire = []  # (filename, iou)
    kept_not = []  # filenames
    total_fire = 0
    total_not = 0

    log_info(f"\nProcessing {len(data_loader.dataset)} images...")

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Inference"):
            images, labels, paths = batch_data[0], batch_data[1], batch_data[2]
            images = images.to(device)

            for i in range(images.size(0)):
                filename = os.path.basename(paths[i])
                true_label = labels[i].item()

                # ── "Not" class (no fire) ──
                if true_label == 1:
                    total_not += 1
                    if args.keep_not:
                        kept_not.append(filename)
                    continue

                # ── "Fire" class ──
                total_fire += 1

                # Check if it has GT
                if filename not in gt_bboxes:
                    continue

                gt_boxes = gt_bboxes[filename]

                # Extract attention map and generate bboxes
                try:
                    img_tensor = images[i : i + 1]
                    attn_np = get_last_layer_attention(model, args.model, img_tensor)
                    threshold = compute_dynamic_threshold(
                        attn_np, method="otsu", fire_adapted=True
                    )
                    pred_bboxes = VisUtils.generate_bounding_box(
                        attn_np, threshold=threshold, min_area=min_bbox_area
                    )
                except Exception as e:
                    pred_bboxes = []

                if not pred_bboxes:
                    removed_fire.append((filename, 0.0))
                    continue

                # Compute best IoU
                best_iou = compute_best_iou(gt_boxes, pred_bboxes)

                if best_iou >= args.iou_threshold:
                    kept_fire.append((filename, best_iou))
                else:
                    removed_fire.append((filename, best_iou))

    # ── Statistics ──
    log_info(f"\n{'=' * 60}")
    log_info(f"CURATION RESULTS")
    log_info(f"{'=' * 60}")
    log_info(f"Total Fire: {total_fire}")
    log_info(f"  Kept (IoU >= {args.iou_threshold}): {len(kept_fire)}")
    log_info(f"  Removed (IoU < {args.iou_threshold}): {len(removed_fire)}")
    if kept_fire:
        ious = [iou for _, iou in kept_fire]
        log_info(f"  Mean IoU (kept): {np.mean(ious):.4f}")
    if removed_fire:
        ious_rem = [iou for _, iou in removed_fire]
        log_info(f"  Mean IoU (removed): {np.mean(ious_rem):.4f}")
    log_info(f"Total Not: {total_not}")
    if args.keep_not:
        log_info(f"  Kept: {len(kept_not)}")
    else:
        log_info(f"  Removed (--no_keep_not)")
    log_info(f"{'=' * 60}")

    # ── Create curated dataset ──
    output_split_path = os.path.join(args.output, args.split)
    output_images_path = os.path.join(output_split_path, "images")
    os.makedirs(output_images_path, exist_ok=True)

    source_images_path = os.path.join(split_path, "images")

    # Set of kept filenames
    all_kept_filenames = set(f for f, _ in kept_fire) | set(kept_not)

    # Copy images
    copied = 0
    for filename in tqdm(all_kept_filenames, desc="Copying images"):
        src = os.path.join(source_images_path, filename)
        dst = os.path.join(output_images_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1

    # Create filtered annotations.json
    kept_image_ids = set()
    new_images = []
    for img_info in coco_data["images"]:
        if img_info["file_name"] in all_kept_filenames:
            new_images.append(img_info)
            kept_image_ids.add(img_info["id"])

    new_annotations = []
    for ann in coco_data["annotations"]:
        if ann["image_id"] in kept_image_ids:
            new_annotations.append(ann)

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data["categories"],
    }

    output_annotations_path = os.path.join(output_split_path, "annotations.json")
    with open(output_annotations_path, "w") as f:
        json.dump(new_coco, f, indent=2)

    # Copy annotations/ folder if it exists
    source_anns_dir = os.path.join(split_path, "annotations")
    if os.path.isdir(source_anns_dir):
        output_anns_dir = os.path.join(output_split_path, "annotations")
        os.makedirs(output_anns_dir, exist_ok=True)
        for filename in os.listdir(source_anns_dir):
            # Annotation name matches the image name (without extension)
            img_name_base = os.path.splitext(filename)[0]
            # Check if any kept image has this basename
            match = any(
                os.path.splitext(f)[0] == img_name_base for f in all_kept_filenames
            )
            if match:
                shutil.copy2(
                    os.path.join(source_anns_dir, filename),
                    os.path.join(output_anns_dir, filename),
                )

    log_info(f"\nCurated dataset saved to: {args.output}")
    log_info(f"  Images copied: {copied}")
    log_info(f"  Annotations (images): {len(new_images)}")
    log_info(f"  Annotations (bboxes): {len(new_annotations)}")

    # Save report
    report = {
        "model": args.model,
        "weights": args.weights,
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "img_size": img_size,
        "keep_not": args.keep_not,
        "total_fire": total_fire,
        "kept_fire": len(kept_fire),
        "removed_fire": len(removed_fire),
        "total_not": total_not,
        "kept_not": len(kept_not),
        "total_kept": len(all_kept_filenames),
        "mean_iou_kept": float(np.mean([iou for _, iou in kept_fire]))
        if kept_fire
        else 0.0,
        "mean_iou_removed": float(np.mean([iou for _, iou in removed_fire]))
        if removed_fire
        else 0.0,
        "kept_fire_detail": [
            {"filename": f, "iou": round(iou, 4)}
            for f, iou in sorted(kept_fire, key=lambda x: x[1])
        ],
        "removed_fire_detail": [
            {"filename": f, "iou": round(iou, 4)}
            for f, iou in sorted(removed_fire, key=lambda x: x[1])
        ],
    }

    report_path = os.path.join(args.output, f"curacao_report_{args.split}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log_info(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
