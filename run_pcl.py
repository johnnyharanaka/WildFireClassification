#!/usr/bin/env python3
"""
PCL (Proposal Cluster Learning) multi-seed training and evaluation script.

Orchestrates PCL WSOD on data_fire. Reports localization metrics only (CorLoc, mAP),
since PCL is a localization method and does not produce binary classification scores.
Classification metrics are N/A.

Usage:
    uv run run_pcl.py                          # 1 seed, 20 000 iters (default)
    uv run run_pcl.py --seeds 42 123 456       # 3 seeds
    uv run run_pcl.py --seeds 42 --max-iter 200  # quick pipeline test
    uv run run_pcl.py --eval-only --seeds 42   # re-evaluate existing checkpoint
    uv run run_pcl.py --train-only --seeds 42  # only train
"""

import argparse
import gc
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

# MPS stability: fallback unsupported ops to CPU, prevent GPU OOM cascade
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_PCL_ROOT     = _PROJECT_ROOT / "core" / "pcl.pytorch"
_PCL_LIB      = _PCL_ROOT / "lib"
_PCL_TOOLS    = _PCL_ROOT / "tools"
_PCL_DATA     = _PCL_ROOT / "data"
_DATA_FIRE    = _PROJECT_ROOT / "data_fire"
_OUTPUTS_DIR  = _PROJECT_ROOT / "Outputs"
_CFG_FILE     = _PCL_ROOT / "configs" / "baselines" / "vgg16_fire.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, cwd=None, check=True):
    """Run a shell command, streaming stdout/stderr."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
    return result


def _find_best_ckpt(output_dir: Path) -> str | None:
    """Return path to the latest checkpoint under output_dir (recursive search).

    Uses modification time so re-runs always pick the freshest checkpoint,
    not one left over from a previous run in the same seed directory.
    """
    ckpts = list(output_dir.rglob("model_step*.pth"))
    if not ckpts:
        ckpts = list(output_dir.rglob("*.pth"))
    if ckpts:
        return str(max(ckpts, key=lambda p: p.stat().st_mtime))
    return None


def _iou(b1, b2):
    """Compute IoU between [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Step 1: Prepare data
# ---------------------------------------------------------------------------

def prepare_data(force: bool = False):
    """Run prepare_fire_data.py once (skip if outputs already exist)."""
    ann_file = _PCL_DATA / "fire" / "annotations" / "fire_train.json"
    if ann_file.exists() and not force:
        print(f"[prepare] {ann_file} exists — skipping (use --force-prep to regenerate)")
        return

    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)

    cmd = [sys.executable, _PCL_TOOLS / "prepare_fire_data.py",
           "--data-root", _DATA_FIRE,
           "--pcl-data", _PCL_DATA]
    if force:
        cmd.append("--force")
    _run(cmd, cwd=_PCL_ROOT)


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------

def train(seed: int, max_iter: int, batch_size: int, num_workers: int,
          extra_set_cfgs: list = None):
    """Run train_net_step.py for one seed."""
    output_dir = _OUTPUTS_DIR / f"fire_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TRAINING PCL  |  seed={seed}  |  max_iter={max_iter}")
    print(f"{'='*60}")

    set_cfgs = [
        "OUTPUT_DIR", str(output_dir),
        "RNG_SEED", str(seed),
        "SOLVER.MAX_ITER", str(max_iter),
    ]
    if extra_set_cfgs:
        set_cfgs.extend(extra_set_cfgs)

    cmd = [
        sys.executable, _PCL_TOOLS / "train_net_step.py",
        "--dataset", "fire",
        "--cfg", _CFG_FILE,
        "--bs", str(batch_size),
        "--nw", str(num_workers),
        "--set", *set_cfgs,
    ]
    _run(cmd, cwd=_PCL_ROOT)
    return output_dir


# ---------------------------------------------------------------------------
# Step 3: Test inference
# ---------------------------------------------------------------------------

def test_inference(seed: int, ckpt_path: str, test_output_dir: Path):
    """Run test_net.py to produce detections.pkl."""
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TEST INFERENCE  |  seed={seed}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, _PCL_TOOLS / "test_net.py",
        "--dataset", "fire",
        "--cfg", _CFG_FILE,
        "--load_ckpt", ckpt_path,
        "--output_dir", str(test_output_dir),
        "--set", "NUM_GPUS", "1",
    ]
    _run(cmd, cwd=_PCL_ROOT)


# ---------------------------------------------------------------------------
# Step 4: Evaluate
# ---------------------------------------------------------------------------

def _load_coco_gt(ann_json: Path):
    """
    Load GT from a COCO JSON.
    Returns:
        img_id_to_path: {img_id: file_name}
        img_id_to_bboxes: {img_id: [[x1,y1,x2,y2], ...]}  (fire images; empty list for non-fire)
        img_id_to_true_label: {img_id: 0 (fire) or 1 (not-fire)}
        fname_to_id: {file_name: img_id}
    """
    with open(ann_json) as f:
        coco = json.load(f)

    img_id_to_path = {img["id"]: img["file_name"] for img in coco["images"]}
    fname_to_id    = {img["file_name"]: img["id"] for img in coco["images"]}

    img_id_to_bboxes = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        img_id_to_bboxes[img_id].append([x, y, x + w, y + h])

    # True label: 0=Fire (has GT bbox), 1=Not-fire
    img_id_to_true_label = {
        img_id: (0 if bboxes else 1)
        for img_id, bboxes in img_id_to_bboxes.items()
    }

    return img_id_to_path, img_id_to_bboxes, img_id_to_true_label, fname_to_id


def _nms_numpy(dets, thresh):
    """Simple NMS on numpy array [N, 5] = [x1,y1,x2,y2,score]."""
    if len(dets) == 0:
        return np.empty((0, 5), dtype=np.float32)
    x1 = dets[:, 0]; y1 = dets[:, 1]
    x2 = dets[:, 2]; y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]


def _extract_fire_predictions(raw_boxes, nms_thresh=0.3, score_thresh=0.0):
    """
    Convert raw PCL output to fire detections.

    raw_boxes: dict with 'scores' (N, C+1) and 'boxes' (N, 4*(C+1))
               where C = cfg.MODEL.NUM_CLASSES = 1 for fire.

    Returns:
        pred_boxes_fire: ndarray (K, 5) = [x1,y1,x2,y2,score] for fire class
        max_fire_score:  float, maximum fire class score across all proposals
    """
    scores = raw_boxes["scores"]   # (N, 2): col 0 = bg, col 1 = fire
    boxes  = raw_boxes["boxes"]    # (N, 8): col 4:8 = fire class boxes

    # Fire class is index 1 (0 = background)
    fire_scores = scores[:, 1]     # (N,)
    fire_boxes  = boxes[:, 4:8]    # (N, 4)

    max_fire_score = float(fire_scores.max()) if len(fire_scores) > 0 else 0.0

    # Filter by score threshold
    inds = np.where(fire_scores > score_thresh)[0]
    if len(inds) == 0:
        # Fall back to top proposal
        top = int(np.argmax(fire_scores))
        dets = np.hstack([
            fire_boxes[top:top+1],
            fire_scores[top:top+1, np.newaxis]
        ]).astype(np.float32)
    else:
        dets = np.hstack([
            fire_boxes[inds],
            fire_scores[inds, np.newaxis]
        ]).astype(np.float32)

    # Apply NMS
    pred_boxes_fire = _nms_numpy(dets, nms_thresh)
    return pred_boxes_fire, max_fire_score


def evaluate(seed: int, test_output_dir: Path,
             iou_thresholds=None, loc_threshold=0.5,
             cls_threshold=0.5) -> dict:
    """
    Load PCL detections.pkl and compute localization metrics only.

    PCL is a WSOD localization method and does not produce binary classification
    scores suitable for fire/not-fire prediction; classification metrics are N/A.

    Args:
        seed: Random seed (for reporting).
        test_output_dir: Directory containing detections.pkl.
        iou_thresholds: IoU thresholds for mAP (default [0.5]).
        loc_threshold: IoU threshold for CorLoc (default 0.5).
        cls_threshold: Score threshold for counting a detection as positive (default 0.5).
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    det_pkl = test_output_dir / "detections.pkl"
    if not det_pkl.exists():
        print(f"  WARNING: {det_pkl} not found — skipping evaluation")
        return {}

    print(f"\n{'='*60}")
    print(f"EVALUATING PCL  |  seed={seed}")
    print(f"{'='*60}")

    # Load detections
    with open(det_pkl, "rb") as f:
        det_data = pickle.load(f)
    all_boxes = det_data["all_boxes"]  # {img_path: {'scores': ..., 'boxes': ...}}

    # Load GT annotations
    ann_json = _PCL_DATA / "fire" / "annotations" / "fire_test.json"
    if not ann_json.exists():
        print(f"  WARNING: {ann_json} not found — run prepare_data() first")
        return {}

    img_id_to_path, img_id_to_bboxes, img_id_to_true_label, fname_to_id = \
        _load_coco_gt(ann_json)

    # Build reverse lookup: abs_path → img_id (for matching PCL output keys)
    test_images_dir = _DATA_FIRE / "Test" / "images"
    abs_to_id = {}
    for img_id, fname in img_id_to_path.items():
        abs_to_id[str(test_images_dir / fname)] = img_id
        abs_to_id[fname] = img_id  # bare filename fallback

    # --- Per-image evaluation ---
    # For CorLoc: (img_id, pred_boxes_list, fire_iou)
    corloc_correct = 0
    corloc_total   = 0

    # For mAP: collect sorted detections
    # map_detections[iou_thr] = list of (score, is_tp)
    all_det_records = []   # list of (score, img_id, [x1,y1,x2,y2]) for AP calc
    ap_gt_counts = 0       # total fire GT boxes

    # Detection confusion
    tp_det = fp_det = fn_det = 0

    for img_path, raw_boxes in all_boxes.items():
        # Resolve img_id
        img_id = abs_to_id.get(str(img_path))
        if img_id is None:
            fname = Path(str(img_path)).name
            img_id = fname_to_id.get(fname)
        if img_id is None:
            continue

        true_label = img_id_to_true_label.get(img_id, 1)
        gt_bboxes  = img_id_to_bboxes.get(img_id, [])

        # Get fire predictions
        pred_boxes_fire, max_fire_score = _extract_fire_predictions(raw_boxes)

        # Only evaluate localization for fire images (those with GT bboxes)
        if true_label != 0:
            continue

        ap_gt_counts += len(gt_bboxes)
        if len(gt_bboxes) == 0:
            continue

        # CorLoc: best predicted box vs GT
        best_pred = pred_boxes_fire[0, :4] if len(pred_boxes_fire) > 0 else None
        if best_pred is not None:
            ious = [_iou(best_pred, gt) for gt in gt_bboxes]
            corloc_correct += int(max(ious) >= loc_threshold)
        corloc_total += 1

        # Collect for AP calculation
        for box in pred_boxes_fire:
            all_det_records.append({
                "score": float(box[4]),
                "img_id": img_id,
                "bbox": box[:4].tolist(),
            })

        # Detection confusion at primary IoU threshold
        iou_thr = iou_thresholds[0]
        gt_matched = [False] * len(gt_bboxes)
        for box in pred_boxes_fire:
            if float(box[4]) < cls_threshold:
                continue
            pred_b = box[:4].tolist()
            matched = False
            for gi, gt_b in enumerate(gt_bboxes):
                if not gt_matched[gi] and _iou(pred_b, gt_b) >= iou_thr:
                    gt_matched[gi] = True
                    matched = True
                    break
            if matched:
                tp_det += 1
            else:
                fp_det += 1
        fn_det += sum(1 for m in gt_matched if not m)

    # --- CorLoc ---
    corloc = (corloc_correct / corloc_total * 100) if corloc_total > 0 else 0.0

    # --- mAP (VOC-style 11-point) ---
    map_values = {}
    for iou_thr in iou_thresholds:
        if not all_det_records:
            map_values[iou_thr] = 0.0
            continue
        # Sort by descending score
        sorted_dets = sorted(all_det_records, key=lambda x: -x["score"])

        # Match against GT
        tp_arr = []
        fp_arr = []
        img_gt_matched = {}  # img_id → list of bools

        for img_id, gt_bboxes in img_id_to_bboxes.items():
            if img_id_to_true_label.get(img_id, 1) == 0:
                img_gt_matched[img_id] = [False] * len(gt_bboxes)

        for det in sorted_dets:
            img_id = det["img_id"]
            pred_b = det["bbox"]
            gt_bboxes = img_id_to_bboxes.get(img_id, [])
            matched_arr = img_gt_matched.get(img_id, [])

            best_iou = 0.0
            best_idx = -1
            for gi, gt_b in enumerate(gt_bboxes):
                iou = _iou(pred_b, gt_b)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_iou >= iou_thr and best_idx >= 0 and not matched_arr[best_idx]:
                matched_arr[best_idx] = True
                tp_arr.append(1)
                fp_arr.append(0)
            else:
                tp_arr.append(0)
                fp_arr.append(1)

        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        recalls    = tp_cum / max(ap_gt_counts, 1)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask]) / 11.0
        map_values[iou_thr] = ap

    map_value = map_values.get(iou_thresholds[0], 0.0)

    # --- Detection metrics ---
    precision_det = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
    recall_det    = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0

    iou_thr = iou_thresholds[0]

    print(f"\nWSOD EVALUATION RESULTS (Test)")
    print(f"Model: PCL_VGG16")
    print(f"Total classes: 1 (Fire)")
    print("-" * 60)
    print("CLASSIFICATION METRICS: N/A")
    print("  (PCL is a localization-only WSOD method; binary classification not supported)")
    print("-" * 60)
    print(f"DETECTION / LOCALIZATION METRICS (IoU >= {iou_thr}):")
    print(f"  TP_det: {tp_det}, FP_det: {fp_det}, FN_det: {fn_det}")
    print(f"  Precision_det: {precision_det:.4f}")
    print(f"  Recall_det: {recall_det:.4f}")
    print(f"  CorLoc (LocAcc >= {loc_threshold}): {corloc:.4f}%")
    print(f"  mAP@IoU{iou_thr}: {map_value:.4f}")
    print("=" * 60)

    return {
        "seed": seed,
        # Classification: N/A for PCL
        "tp": None, "tn": None, "fp": None, "fn": None,
        "accuracy": None, "error_rate": None,
        "precision": None, "recall": None,
        "fdr": None, "f1_score": None,
        # Localization
        "tp_det":        int(tp_det),
        "fp_det":        int(fp_det),
        "fn_det":        int(fn_det),
        "precision_det": float(precision_det),
        "recall_det":    float(recall_det),
        "corloc":        float(corloc),
        "map":           float(map_value),
    }


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def print_individual_results(all_results: list, iou_threshold: float):
    print(f"\n{'='*100}")
    print(f"INDIVIDUAL RESULTS - PCL_VGG16")
    print(f"{'='*100}")

    print(f"\nCLASSIFICATION METRICS: N/A (PCL is a localization-only WSOD method)")

    print(f"\nDETECTION / LOCALIZATION METRICS (IoU >= {iou_threshold})")
    print("-" * 100)
    print(f"{'Seed':<8} {'TP_det':<8} {'FP_det':<8} {'FN_det':<8} "
          f"{'Prec_det':<12} {'Recall_det':<12} {'CorLoc':<12} {'mAP':<12}")
    print("-" * 100)
    for r in all_results:
        print(
            f"{r.get('seed','?'):<8} "
            f"{r.get('tp_det','N/A'):<8} {r.get('fp_det','N/A'):<8} "
            f"{r.get('fn_det','N/A'):<8} "
            f"{r['precision_det']:<12.4f} {r['recall_det']:<12.4f} "
            f"{r['corloc']:<12.2f} {r['map']:<12.4f}"
        )


def print_summary_statistics(all_results: list, iou_threshold: float):
    n = len(all_results)
    print(f"\n{'='*100}")
    print(f"SUMMARY STATISTICS - PCL_VGG16 (n={n} seeds)")
    print(f"{'='*100}")

    def calc_stats(key):
        values = [r[key] for r in all_results if r.get(key) is not None]
        if not values:
            return None, None
        return np.mean(values), np.std(values)

    print(f"\nCLASSIFICATION METRICS: N/A (PCL is a localization-only WSOD method)")

    print(f"\nDETECTION / LOCALIZATION METRICS (IoU >= {iou_threshold})")
    print("-" * 100)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Mean ± Std':<30}")
    print("-" * 100)

    for metric, label in [("tp_det","TP_det"), ("fp_det","FP_det"), ("fn_det","FN_det")]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.1f} {std:<15.1f} {mean:.1f} ± {std:.1f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    print("-" * 100)
    for metric, label in [
        ("precision_det","Precision_det"), ("recall_det","Recall_det"),
        ("corloc","CorLoc (%)"), ("map","mAP"),
    ]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.4f} {std:<15.4f} {mean:.4f} ± {std:.4f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    print(f"\n{'='*100}")
    print(f"\nPAPER-READY FORMAT:")
    print("-" * 60)

    for metric, label, fmt in [
        ("map",    f"mAP@{iou_threshold}", lambda m, s: f"{m:.4f} ± {s:.4f}"),
        ("corloc", "CorLoc",              lambda m, s: f"{m:.2f}% ± {s:.2f}%"),
    ]:
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<12}: {fmt(mean, std)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCL multi-seed training and evaluation")
    parser.add_argument("--run", type=int, default=1,
                        help="Number of runs with random seeds (default: 1)")
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+",
                        default=None, help="Specific seeds (overrides --run)")
    parser.add_argument("--max-iter", type=int, default=20000,
                        help="Max training iterations per seed (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1 for PCL)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loader workers (default: 0)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--loc-threshold", type=float, default=0.5)
    parser.add_argument("--cls-threshold", type=float, default=0.5,
                        help="Fire score threshold for counting detections as positive (default: 0.5)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train, skip evaluation")
    parser.add_argument("--force-prep", action="store_true",
                        help="Force re-running data preparation")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip data preparation step entirely")

    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [
        random.randint(0, 2**31 - 1) for _ in range(args.run)
    ]
    if args.seeds is None:
        print(f"Using random seeds: {seeds}")

    print(f"\n{'#'*60}")
    print(f"PCL MULTI-SEED EXPERIMENT")
    print(f"{'#'*60}")
    print(f"Model:    PCL_VGG16")
    print(f"Dataset:  data_fire")
    print(f"Seeds:    {seeds}")
    print(f"MaxIter:  {args.max_iter}")
    print(f"IoU thr:  {args.iou_threshold}")
    print(f"ClsThr:   {args.cls_threshold}")
    print(f"{'#'*60}\n")

    # --- Data preparation ---
    if not args.skip_prep:
        prepare_data(force=args.force_prep)

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")

        train_output_dir = _OUTPUTS_DIR / f"fire_seed_{seed}"
        test_output_dir  = train_output_dir / "test"

        # --- Train ---
        if not args.eval_only:
            train(
                seed=seed,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            gc.collect()

        # --- Test inference ---
        if not args.train_only:
            ckpt = _find_best_ckpt(train_output_dir)
            if ckpt is None:
                print(f"  No checkpoint found in {train_output_dir} — skipping")
                continue
            print(f"  Using checkpoint: {ckpt}")

            if not args.eval_only or not (test_output_dir / "detections.pkl").exists():
                test_inference(seed=seed, ckpt_path=ckpt,
                               test_output_dir=test_output_dir)

            # --- Evaluate ---
            metrics = evaluate(
                seed=seed,
                test_output_dir=test_output_dir,
                iou_thresholds=[args.iou_threshold],
                loc_threshold=args.loc_threshold,
                cls_threshold=args.cls_threshold,
            )
            if metrics:
                all_results.append(metrics)
                print(f"  → mAP={metrics['map']:.4f}  "
                      f"CorLoc={metrics['corloc']:.2f}%")

            gc.collect()

    # --- Summary ---
    if all_results and not args.train_only:
        print_individual_results(all_results, args.iou_threshold)
        print_summary_statistics(all_results, args.iou_threshold)


if __name__ == "__main__":
    main()
