#!/usr/bin/env python3
"""
TSCAM multi-seed training and evaluation script.

Trains TSCAM with multiple seeds and reports mean ± std for all metrics,
using the same evaluation pipeline as the other models in this project
(CorLoc + mAP + classification metrics).

Usage:
    uv run run_tscam.py
    uv run run_tscam.py --seeds 42 123 456 789 1234
    uv run run_tscam.py --variant small --epochs 10 --seeds 42 123 456
    uv run run_tscam.py --eval-only --seeds 42 123
    uv run run_tscam.py --train-only --seeds 42
"""

import os
import sys
import gc
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

# --- sys.path setup ---
# 1) Project root first so `core` resolves to THIS project's core package,
#    not to core/TSCAM/lib/core/ (which also exists and would shadow it).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 2) TSCAM lib appended at the end so `models.deit` is reachable without
#    conflicting with the project's `core` package.
_TSCAM_LIB = os.path.join(_PROJECT_ROOT, "core", "TSCAM", "lib")
if _TSCAM_LIB not in sys.path:
    sys.path.append(_TSCAM_LIB)

import models.deit  # noqa: F401  - registers deit_tscam_* via @register_model
from timm.models import create_model as timm_create_model

from core.config.config import get_active_dataset_config, get_config, log_info
from core.datasets import get_data_loaders
from core.metrics import (
    calculate_corloc,
    calculate_map,
    calculate_detection_confusion,
    load_coco_bboxes_from_json,
)
from core.vis_utils import VisUtils, compute_dynamic_threshold

# TSCAM uses 224×224 images with patch_size=16 → 14×14 feature maps
IMG_SIZE = 224

VARIANT_MAP = {
    "tiny":  "deit_tscam_tiny_patch16_224",
    "small": "deit_tscam_small_patch16_224",
    "base":  "deit_tscam_base_patch16_224",
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device_str: str = None):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_tscam(num_classes: int, variant: str = "small", pretrained: bool = True,
                checkpoint: str = None, device=None):
    """Create TSCAM and optionally load a checkpoint."""
    arch = VARIANT_MAP.get(variant, VARIANT_MAP["small"])
    model = timm_create_model(arch, pretrained=pretrained, num_classes=num_classes)

    if checkpoint and os.path.isfile(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        sd = state.get("state_dict", state.get("model", state))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model_sd = model.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"  Loaded checkpoint: {checkpoint}")

    return model.to(device) if device else model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tscam(seed: int, variant: str, num_classes: int, epochs: int,
                batch_size: int, lr: float, weight_decay: float,
                num_workers: int, device, pretrained: bool = True) -> str:
    """
    Train TSCAM for one seed and save the best checkpoint.

    Returns:
        Path to saved checkpoint (models/TSCAM_{variant}_seed_{seed}.pth)
    """
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"TRAINING TSCAM ({variant})  |  seed={seed}")
    print(f"{'='*60}")

    root_path = (
        get_active_dataset_config("data_path")
        or get_active_dataset_config("root_path")
        or "data/"
    )

    data_loaders, _ = get_data_loaders(
        root_path=root_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=IMG_SIZE,
    )
    train_loader = data_loaders.get("Train")
    val_loader = data_loaders.get("Val")

    if train_loader is None:
        raise RuntimeError(f"Train split not found in {root_path}")

    model = build_tscam(num_classes=num_classes, variant=variant,
                        pretrained=pretrained, device=device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    save_path = Path("models") / f"TSCAM_{variant}_seed_{seed}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_correct = train_total = 0
        train_loss_sum = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
        for batch_data in loop:
            images, labels = batch_data[0].to(device), batch_data[1].to(device)

            optimizer.zero_grad()
            output = model(images)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = train_correct / train_total
        train_loss = train_loss_sum / len(train_loader)

        # --- Val ---
        val_acc = 0.0
        if val_loader:
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    images, labels = batch_data[0].to(device), batch_data[1].to(device)
                    output = model(images)
                    logits = output[0] if isinstance(output, tuple) else output
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total

        print(f"  Epoch {epoch+1}/{epochs}  |  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"  Best val_acc={best_val_acc:.4f}  →  saved to {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# CAM extraction
# ---------------------------------------------------------------------------

def extract_cam(cam_tensor: torch.Tensor, class_idx: int, img_size: int) -> np.ndarray:
    """
    Extract and upsample the class-specific CAM from TSCAM output.

    Args:
        cam_tensor: [num_classes, H_patch, W_patch] for one image.
        class_idx: Class whose CAM to use.
        img_size: Target spatial resolution.

    Returns:
        np.ndarray [img_size, img_size] normalized to [0, 1].
    """
    import torch.nn.functional as F

    cam = cam_tensor[class_idx]  # [H_patch, W_patch] — still a tensor

    # Normalize to [0, 1] on the tensor (avoids numpy/cv2 for upsampling)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    # Upsample with torch — avoids any cv2 module conflict
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return cam_up.detach().cpu().numpy().clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tscam(seed: int, checkpoint: str, variant: str, num_classes: int,
                   batch_size: int, num_workers: int, device,
                   iou_thresholds=None, loc_threshold: float = 0.5,
                   split: str = "Test") -> dict:
    """
    Evaluate a TSCAM checkpoint using the same CorLoc + mAP pipeline
    as evaluate_wsod() in core/metrics.py.
    """
    set_seed(seed)

    if iou_thresholds is None:
        iou_thresholds = [0.5]

    root_path = (
        get_active_dataset_config("data_path")
        or get_active_dataset_config("root_path")
        or "data/"
    )

    print(f"\n{'='*60}")
    print(f"EVALUATING TSCAM ({variant})  |  seed={seed}  |  split={split}")
    print(f"{'='*60}")

    model = build_tscam(num_classes=num_classes, variant=variant,
                        pretrained=False, checkpoint=checkpoint, device=device)
    model.eval()

    data_loaders, _ = get_data_loaders(
        root_path=root_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=IMG_SIZE,
    )
    data_loader = data_loaders.get(split)

    if not data_loader or len(data_loader) == 0:
        print(f"No {split} split found in {root_path}")
        return {}

    all_predictions = []
    bbox_to_id = {}
    image_sizes = {}
    all_pred_labels = []
    all_true_labels = []
    total_images = 0

    with torch.no_grad():
        loop = tqdm(data_loader, desc=f"Eval ({split})")
        for batch_idx, batch_data in enumerate(loop):
            if len(batch_data) >= 3:
                images, labels, image_paths = batch_data[0], batch_data[1], batch_data[2]
                bboxes_gt = batch_data[3] if len(batch_data) > 3 else None
            else:
                continue

            images = images.to(device)
            batch_sz = images.size(0)
            total_images += batch_sz

            # TSCAM eval mode → (logits, cams)
            # cams: [B, num_classes, H_patch, W_patch]
            output = model(images, return_cam=True)
            if isinstance(output, tuple):
                logits, cams = output[0], output[1]
            else:
                logits, cams = output, None
            pred_classes = logits.argmax(dim=1)

            for i in range(batch_sz):
                img_id = (
                    str(image_paths[i])
                    if isinstance(image_paths, (list, tuple))
                    else f"img_{batch_idx}_{i}"
                )
                pred_label = pred_classes[i].item()
                true_label = (
                    labels[i].item() if isinstance(labels, torch.Tensor)
                    else int(labels[i])
                )
                all_pred_labels.append(pred_label)
                all_true_labels.append(true_label)

                # Localization only for Fire images (label == 0)
                if true_label != 0:
                    continue

                # Extract CAM using true class (ground-truth-known localization)
                if cams is None:
                    pred_boxes = [(0, 0, IMG_SIZE, IMG_SIZE, 1.0)]
                    all_predictions.append((img_id, pred_boxes, IMG_SIZE, image_sizes, true_label))
                    bbox_to_id[img_id] = [[0, 0, IMG_SIZE, IMG_SIZE, true_label]]
                    image_sizes[img_id] = (IMG_SIZE, IMG_SIZE)
                    continue
                attn_np = extract_cam(cams[i], true_label, IMG_SIZE)

                threshold = compute_dynamic_threshold(attn_np, method="otsu", fire_adapted=True)
                min_bbox_area = get_config("defaults", "min_bbox_area") or 1600
                bbs_raw = VisUtils.generate_bounding_box(
                    attn_np, threshold=threshold, min_area=min_bbox_area
                )

                if len(bbs_raw) > 1:
                    from core.debug import remove_nested_bboxes
                    bbs_raw = remove_nested_bboxes(bbs_raw, containment_threshold=0.5)

                pred_boxes = []
                for bb in bbs_raw:
                    if len(bb) >= 4:
                        x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
                        conf = bb[4] if len(bb) > 4 else 1.0
                        pred_boxes.append((x1, y1, x2, y2, conf))

                if not pred_boxes:
                    pred_boxes = [(0, 0, IMG_SIZE, IMG_SIZE, 1.0)]

                all_predictions.append((img_id, pred_boxes, IMG_SIZE, image_sizes, true_label))

                if bboxes_gt is not None and i < len(bboxes_gt):
                    gt_boxes = bboxes_gt[i]
                    if isinstance(gt_boxes, torch.Tensor):
                        gt_boxes = gt_boxes.cpu().numpy().tolist()
                    bbox_to_id[img_id] = gt_boxes
                else:
                    bbox_to_id[img_id] = [[0, 0, IMG_SIZE, IMG_SIZE, true_label]]

                h, w = images[i].shape[-2:]
                image_sizes[img_id] = (w, h)

    # Merge real COCO GT bboxes
    split_path = os.path.join(root_path, split)
    real_bboxes = load_coco_bboxes_from_json(split_path, target_size=IMG_SIZE)
    if real_bboxes:
        for img_id in list(bbox_to_id.keys()):
            fname = os.path.basename(img_id)
            if fname in real_bboxes:
                bbox_to_id[img_id] = real_bboxes[fname]

    # --- WSOD metrics ---
    corloc = calculate_corloc(all_predictions, bbox_to_id, loc_threshold=loc_threshold)
    map_value = calculate_map(
        all_predictions, bbox_to_id,
        image_sizes=image_sizes,
        iou_thresholds=iou_thresholds,
    )
    det_m = calculate_detection_confusion(
        all_predictions, bbox_to_id, iou_threshold=iou_thresholds[0]
    )

    # --- Classification metrics ---
    accuracy = precision = recall = f1 = fdr = error_rate = 0.0
    tp = tn = fp = fn = 0
    if all_pred_labels and all_true_labels:
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        error_rate = 1.0 - accuracy
        precision = precision_score(all_true_labels, all_pred_labels,
                                    average="binary", zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels,
                              average="binary", zero_division=0)
        fdr = 1.0 - precision
        f1 = f1_score(all_true_labels, all_pred_labels,
                      average="binary", zero_division=0)
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

    # Verbose output (same format as evaluate_wsod so run_multi_seed regex still works)
    iou_thr = iou_thresholds[0]
    print(f"\nWSOD EVALUATION RESULTS ({split})")
    print(f"Model: TSCAM_{variant}")
    print(f"Total images: {total_images}")
    print(f"Total classes: {num_classes}")
    print("-" * 60)
    print("CLASSIFICATION METRICS:")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Error Rate (ER): {error_rate:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  FDR (False Discovery Rate): {fdr:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print("-" * 60)
    print(f"DETECTION METRICS (IoU >= {iou_thr}):")
    print(f"  TP_det: {det_m['tp_det']}, FP_det: {det_m['fp_det']}, FN_det: {det_m['fn_det']}")
    print(f"  Precision_det: {det_m['precision_det']:.4f}")
    print(f"  Recall_det: {det_m['recall_det']:.4f}")
    print(f"  CorLoc (LocAcc >= {loc_threshold}): {corloc:.4f}")
    print(f"  mAP@IoU{iou_thr}: {map_value:.4f}")
    print("=" * 60)

    return {
        "seed": seed,
        # Classification
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": float(accuracy),
        "error_rate": float(error_rate),
        "precision": float(precision),
        "recall": float(recall),
        "fdr": float(fdr),
        "f1_score": float(f1),
        # Detection
        "tp_det": det_m["tp_det"],
        "fp_det": det_m["fp_det"],
        "fn_det": det_m["fn_det"],
        "precision_det": det_m["precision_det"],
        "recall_det": det_m["recall_det"],
        "corloc": float(corloc),
        "map": float(map_value),
    }


# ---------------------------------------------------------------------------
# Output tables (same format as run_multi_seed.py)
# ---------------------------------------------------------------------------

def print_individual_results(all_results: list, variant: str, iou_threshold: float):
    model_label = f"TSCAM_{variant}"
    print(f"\n{'='*110}")
    print(f"INDIVIDUAL RESULTS - {model_label}")
    print(f"{'='*110}")

    print(f"\nCLASSIFICATION METRICS")
    print("-"*110)
    print(f"{'Seed':<8} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} "
          f"{'Acc':<10} {'ER':<10} {'Prec':<10} {'Recall':<10} {'FDR':<10} {'F1':<10}")
    print("-"*110)

    for r in all_results:
        seed = r.get("seed", "?")
        print(
            f"{seed:<8} "
            f"{r.get('tp','N/A'):<6} {r.get('tn','N/A'):<6} "
            f"{r.get('fp','N/A'):<6} {r.get('fn','N/A'):<6} "
            f"{r['accuracy']:.4f}   {r['error_rate']:.4f}   "
            f"{r['precision']:.4f}   {r['recall']:.4f}   "
            f"{r['fdr']:.4f}   {r['f1_score']:.4f}"
        )

    print(f"\nDETECTION METRICS (IoU >= {iou_threshold})")
    print("-"*100)
    print(f"{'Seed':<8} {'TP_det':<8} {'FP_det':<8} {'FN_det':<8} "
          f"{'Prec_det':<12} {'Recall_det':<12} {'CorLoc':<12} {'mAP':<12}")
    print("-"*100)

    for r in all_results:
        seed = r.get("seed", "?")
        print(
            f"{seed:<8} "
            f"{r.get('tp_det','N/A'):<8} {r.get('fp_det','N/A'):<8} "
            f"{r.get('fn_det','N/A'):<8} "
            f"{r['precision_det']:<12.4f} {r['recall_det']:<12.4f} "
            f"{r['corloc']:<12.2f} {r['map']:<12.4f}"
        )


def print_summary_statistics(all_results: list, variant: str, iou_threshold: float):
    model_label = f"TSCAM_{variant}"
    n = len(all_results)
    print(f"\n{'='*110}")
    print(f"SUMMARY STATISTICS - {model_label} (n={n} seeds)")
    print(f"{'='*110}")

    def calc_stats(key):
        values = [r[key] for r in all_results if r.get(key) is not None]
        if not values:
            return None, None
        return np.mean(values), np.std(values)

    # Classification
    print(f"\nCLASSIFICATION METRICS")
    print("-"*100)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Mean ± Std':<30}")
    print("-"*100)

    for metric, label in [("tp","TP"), ("tn","TN"), ("fp","FP"), ("fn","FN")]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.1f} {std:<15.1f} {mean:.1f} ± {std:.1f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    print("-"*100)
    for metric, label in [
        ("accuracy","Accuracy"), ("error_rate","Error Rate"),
        ("precision","Precision"), ("recall","Recall"),
        ("fdr","FDR"), ("f1_score","F1-score"),
    ]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.4f} {std:<15.4f} {mean:.4f} ± {std:.4f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    # Detection
    print(f"\nDETECTION METRICS (IoU >= {iou_threshold})")
    print("-"*100)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Mean ± Std':<30}")
    print("-"*100)

    for metric, label in [("tp_det","TP_det"), ("fp_det","FP_det"), ("fn_det","FN_det")]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.1f} {std:<15.1f} {mean:.1f} ± {std:.1f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    print("-"*100)
    for metric, label in [
        ("precision_det","Precision_det"), ("recall_det","Recall_det"),
        ("corloc","CorLoc"), ("map","mAP"),
    ]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.4f} {std:<15.4f} {mean:.4f} ± {std:.4f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    print(f"\n{'='*110}")
    print(f"\nPAPER-READY FORMAT:")
    print("-"*60)

    for metric, label, fmt in [
        ("accuracy",  "Accuracy",  lambda m, s: f"{m*100:.2f}% ± {s*100:.2f}%"),
        ("f1_score",  "F1-score",  lambda m, s: f"{m:.4f} ± {s:.4f}"),
        ("map",       f"mAP@{iou_threshold}", lambda m, s: f"{m:.4f} ± {s:.4f}"),
        ("corloc",    "CorLoc",    lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    ]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<12}: {fmt(mean, std)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TSCAM multi-seed training and evaluation")
    parser.add_argument("--run", type=int, default=1,
                        help="Number of runs with random seeds (default: 1)")
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=None,
                        help="Specific seeds to use (overrides --run)")
    parser.add_argument("--variant", type=str, default="small",
                        choices=["tiny", "small", "base"],
                        help="TSCAM size variant (default: small)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per seed (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate (default: 4e-5)")
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument("--split", type=str, default="Test",
                        choices=["Train", "Val", "Test"])
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | cuda | mps (auto-detect if not set)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train, skip evaluation")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do NOT load ImageNet-pretrained DeiT weights")

    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [
        random.randint(0, 2**31 - 1) for _ in range(args.run)
    ]
    if args.seeds is None:
        print(f"Using random seeds: {seeds}")

    device = get_device(args.device)
    num_classes = get_active_dataset_config("num_classes") or 2
    dataset_name = get_active_dataset_config("name") or "?"
    pretrained = not args.no_pretrained

    print(f"\n{'#'*60}")
    print(f"TSCAM MULTI-SEED EXPERIMENT")
    print(f"{'#'*60}")
    print(f"Variant:    TSCAM_{args.variant}")
    print(f"Dataset:    {dataset_name}  ({num_classes} classes)")
    print(f"Seeds:      {seeds}")
    print(f"Epochs:     {args.epochs}")
    print(f"Device:     {device}")
    print(f"Pretrained: {pretrained}")
    print(f"IoU thr:    {args.iou_threshold}")
    print(f"{'#'*60}\n")

    all_results = []
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        ckpt_path = str(models_dir / f"TSCAM_{args.variant}_seed_{seed}.pth")

        # --- Train ---
        if not args.eval_only:
            train_tscam(
                seed=seed,
                variant=args.variant,
                num_classes=num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                device=device,
                pretrained=pretrained,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Evaluate ---
        if not args.train_only:
            if not os.path.isfile(ckpt_path):
                print(f"  Checkpoint not found: {ckpt_path}  →  skipping")
                continue

            metrics = evaluate_tscam(
                seed=seed,
                checkpoint=ckpt_path,
                variant=args.variant,
                num_classes=num_classes,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                iou_thresholds=[args.iou_threshold],
                loc_threshold=args.loc_threshold,
                split=args.split,
            )
            if metrics:
                all_results.append(metrics)
                print(f"  → Acc={metrics['accuracy']:.4f}  "
                      f"F1={metrics['f1_score']:.4f}  "
                      f"mAP={metrics['map']:.4f}  "
                      f"CorLoc={metrics['corloc']:.2f}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Summary ---
    if all_results and not args.train_only:
        print_individual_results(all_results, args.variant, args.iou_threshold)
        print_summary_statistics(all_results, args.variant, args.iou_threshold)


if __name__ == "__main__":
    main()