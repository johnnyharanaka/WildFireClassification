#!/usr/bin/env python3
"""
WSDDN (Weakly Supervised Deep Detection Networks) multi-seed training and evaluation.

Unlike PCL, WSDDN trains with BOTH Fire and Not-Fire images using BCE loss,
so it supports binary classification as well as localization.

Usage:
    uv run run_wsddn.py                        # 1 seed, 20 000 iters (default)
    uv run run_wsddn.py --seeds 42 123 456     # 3 explicit seeds
    uv run run_wsddn.py --seeds 42 --max-iter 500   # quick pipeline test
    uv run run_wsddn.py --eval-only --seeds 42      # re-evaluate existing checkpoint
    uv run run_wsddn.py --train-only --seeds 42     # only train
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

_PROJECT_ROOT  = Path(__file__).resolve().parent
_WSDDN_ROOT    = _PROJECT_ROOT / "core" / "WSDDN-PyTorch"
_WSDDN_CODE    = _WSDDN_ROOT / "code"
_WSDDN_DATA    = _WSDDN_ROOT / "data"
_PCL_DATA      = _PROJECT_ROOT / "core" / "pcl.pytorch" / "data"
_DATA_FIRE     = _PROJECT_ROOT / "data_fire"
_OUTPUTS_DIR   = _PROJECT_ROOT / "Outputs"
_CFG_FILE      = _WSDDN_ROOT / "configs" / "baselines" / "vgg16_fire.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, cwd=None, check=True):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")
    return result


def _find_best_ckpt(output_dir: Path) -> str | None:
    """Return path to most-recently-modified checkpoint under output_dir."""
    ckpts = list(output_dir.rglob("model_step*.pth"))
    if not ckpts:
        ckpts = list(output_dir.rglob("*.pth"))
    if ckpts:
        return str(max(ckpts, key=lambda p: p.stat().st_mtime))
    return None


def _iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _nms_numpy(dets, thresh):
    if len(dets) == 0:
        return np.empty((0, 5), dtype=np.float32)
    x1,y1,x2,y2,scores = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        iou = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1) / (areas[i]+areas[order[1:]]-np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)+1e-9)
        order = order[np.where(iou<=thresh)[0]+1]
    return dets[keep]


# ---------------------------------------------------------------------------
# Step 1: Prepare data (delegate to PCL's prepare script)
# ---------------------------------------------------------------------------

def prepare_data(force: bool = False):
    ann_file = _WSDDN_DATA / "fire" / "annotations" / "fire_train.json"
    if ann_file.exists() and not force:
        print(f"[prepare] {ann_file} exists — skipping (use --force-prep to regenerate)")
    else:
        print("\n" + "="*60 + "\nPREPARING DATA\n" + "="*60)
        _run([sys.executable,
              _PROJECT_ROOT / "core" / "pcl.pytorch" / "tools" / "prepare_fire_data.py",
              "--data-root", _DATA_FIRE,
              "--pcl-data", _WSDDN_DATA],
             cwd=_PROJECT_ROOT / "core" / "pcl.pytorch")

    # Ensure VGG16 weights exist in WSDDN data dir
    wsddn_weights = _WSDDN_DATA / "pretrained_model" / "vgg16_imagenet.pth"
    pcl_weights   = _PCL_DATA   / "pretrained_model" / "vgg16_imagenet.pth"
    if not wsddn_weights.exists() and pcl_weights.exists():
        wsddn_weights.parent.mkdir(parents=True, exist_ok=True)
        wsddn_weights.symlink_to(pcl_weights)
        print(f"[prepare] Symlinked VGG16 weights → {wsddn_weights}")
    elif not wsddn_weights.exists() and not pcl_weights.exists():
        print("[prepare] WARNING: VGG16 weights not found at "
              f"{pcl_weights}\n  Run PCL prepare script first or place weights manually.")


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------

def train(seed: int, max_iter: int, num_workers: int, extra_set_cfgs: list = None):
    output_dir = _OUTPUTS_DIR / f"wsddn_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nTRAINING WSDDN  |  seed={seed}  |  max_iter={max_iter}\n{'='*60}")

    set_cfgs = [
        "OUTPUT_DIR", str(output_dir),
        "RNG_SEED",   str(seed),
        "SOLVER.MAX_ITER", str(max_iter),
    ]
    if extra_set_cfgs:
        set_cfgs.extend(extra_set_cfgs)

    cmd = [
        sys.executable, _WSDDN_CODE / "tasks" / "train.py",
        "--cfg",   _CFG_FILE,
        "--model", "midn",
        "--nw",    str(num_workers),
        "--set",   *set_cfgs,
    ]
    _run(cmd, cwd=_WSDDN_CODE)
    return output_dir


# ---------------------------------------------------------------------------
# Step 3: Test inference
# ---------------------------------------------------------------------------

def test_inference(seed: int, ckpt_path: str, test_output_dir: Path):
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nTEST INFERENCE  |  seed={seed}\n{'='*60}")

    cmd = [
        sys.executable, _WSDDN_CODE / "tasks" / "test.py",
        "--dataset",    "fire",
        "--cfg",        _CFG_FILE,
        "--load_ckpt",  ckpt_path,
        "--output_dir", str(test_output_dir),
        "--model",      "midn",
    ]
    _run(cmd, cwd=_WSDDN_CODE)


# ---------------------------------------------------------------------------
# Step 4: Evaluate
# ---------------------------------------------------------------------------

def _load_coco_gt(split: str):
    ann_json = _WSDDN_DATA / "fire" / "annotations" / f"fire_{split}.json"
    with open(ann_json) as f:
        coco = json.load(f)
    img_id_to_path   = {img["id"]: img["file_name"] for img in coco["images"]}
    fname_to_id      = {img["file_name"]: img["id"] for img in coco["images"]}
    id_to_bboxes     = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        x,y,w,h = ann["bbox"]
        id_to_bboxes[ann["image_id"]].append([x, y, x+w, y+h])
    # label: 0=Fire, 1=Not-fire  (same convention as run_pcl.py)
    id_to_label = {
        img_id: (0 if bboxes else 1)
        for img_id, bboxes in id_to_bboxes.items()
    }
    return img_id_to_path, fname_to_id, id_to_bboxes, id_to_label


def evaluate(seed: int, test_output_dir: Path,
             iou_thresholds=None, loc_threshold=0.5,
             cls_threshold=0.5) -> dict:
    """
    Evaluate WSDDN detections.

    WSDDN is a localization method — classification metrics are N/A.

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

    print(f"\n{'='*60}\nEVALUATING WSDDN  |  seed={seed}\n{'='*60}")

    with open(det_pkl, "rb") as f:
        det_data = pickle.load(f)

    # detections[img_key] = tensor [roi_count, num_classes+1]
    # proposals[img_key]  = ndarray [roi_count, 4] (x1,y1,x2,y2)
    detections     = det_data["all_boxes"]
    proposals_dict = det_data.get("all_proposals", {})

    # Load GT
    img_id_to_path, fname_to_id, id_to_bboxes, id_to_label = _load_coco_gt("test")

    # Reverse lookup: abs_path or basename → img_id
    test_img_dir = _DATA_FIRE / "Test" / "images"
    abs_to_id = {}
    for img_id, fname in img_id_to_path.items():
        abs_to_id[str(test_img_dir / fname)] = img_id
        abs_to_id[fname] = img_id

    # --- Per-image ---
    corloc_correct = 0; corloc_total = 0
    all_det_records = []; ap_gt_counts = 0
    tp_det = fp_det = fn_det = 0

    for img_path, scores_tensor in detections.items():
        img_id = abs_to_id.get(str(img_path))
        if img_id is None:
            img_id = fname_to_id.get(Path(str(img_path)).name)
        if img_id is None:
            continue

        true_label = id_to_label.get(img_id, 1)
        gt_bboxes  = id_to_bboxes.get(img_id, [])

        # scores_tensor: [roi_count, 2] (bg=col0, fire=col1)
        scores = scores_tensor.numpy() if hasattr(scores_tensor, 'numpy') else np.array(scores_tensor)
        fire_scores = scores[:, 1]  # fire class
        proposals   = proposals_dict.get(img_path)
        if proposals is None:
            proposals = proposals_dict.get(str(img_path))

        # --- Localization (fire images only) ---
        if true_label != 0:
            continue

        ap_gt_counts += len(gt_bboxes)
        if len(gt_bboxes) == 0 or proposals is None or len(proposals) == 0:
            corloc_total += 1
            continue

        # Build fire detections: [x1,y1,x2,y2,score]
        fire_dets = np.hstack([proposals, fire_scores[:, np.newaxis]]).astype(np.float32)
        fire_dets = _nms_numpy(fire_dets, thresh=0.3)

        # CorLoc: best box vs GT
        if len(fire_dets) > 0:
            best_pred = fire_dets[0, :4]
            ious = [_iou(best_pred, gt) for gt in gt_bboxes]
            corloc_correct += int(max(ious) >= loc_threshold)
        corloc_total += 1

        # Collect for AP
        for box in fire_dets:
            all_det_records.append({
                "score": float(box[4]),
                "img_id": img_id,
                "bbox": box[:4].tolist(),
            })

        # Detection confusion
        iou_thr = iou_thresholds[0]
        gt_matched = [False] * len(gt_bboxes)
        for box in fire_dets:
            if float(box[4]) < cls_threshold:
                continue
            pred_b = box[:4].tolist()
            matched = False
            for gi, gt_b in enumerate(gt_bboxes):
                if not gt_matched[gi] and _iou(pred_b, gt_b) >= iou_thr:
                    gt_matched[gi] = True; matched = True; break
            tp_det += int(matched); fp_det += int(not matched)
        fn_det += sum(1 for m in gt_matched if not m)

    # --- CorLoc ---
    corloc = (corloc_correct / corloc_total * 100) if corloc_total > 0 else 0.0

    # --- mAP (VOC 11-point) ---
    map_values = {}
    for iou_thr in iou_thresholds:
        if not all_det_records:
            map_values[iou_thr] = 0.0; continue
        sorted_dets = sorted(all_det_records, key=lambda x: -x["score"])
        img_gt_matched = {img_id: [False]*len(bboxes)
                          for img_id, bboxes in id_to_bboxes.items()
                          if id_to_label.get(img_id,1) == 0}
        tp_arr = []; fp_arr = []
        for det in sorted_dets:
            iid = det["img_id"]; pred_b = det["bbox"]
            gt_bboxes = id_to_bboxes.get(iid, [])
            matched_arr = img_gt_matched.get(iid, [])
            best_iou = 0.0; best_idx = -1
            for gi, gt_b in enumerate(gt_bboxes):
                iou = _iou(pred_b, gt_b)
                if iou > best_iou: best_iou = iou; best_idx = gi
            if best_iou >= iou_thr and best_idx >= 0 and not matched_arr[best_idx]:
                matched_arr[best_idx] = True; tp_arr.append(1); fp_arr.append(0)
            else:
                tp_arr.append(0); fp_arr.append(1)
        tp_cum = np.cumsum(tp_arr); fp_cum = np.cumsum(fp_arr)
        recalls    = tp_cum / max(ap_gt_counts, 1)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        ap = sum(np.max(precisions[recalls >= t]) / 11.0
                 for t in np.linspace(0,1,11) if (recalls >= t).any())
        map_values[iou_thr] = ap

    map_value = map_values.get(iou_thresholds[0], 0.0)

    # --- Detection metrics ---
    precision_det = tp_det/(tp_det+fp_det) if (tp_det+fp_det)>0 else 0.0
    recall_det    = tp_det/(tp_det+fn_det) if (tp_det+fn_det)>0 else 0.0

    iou_thr = iou_thresholds[0]

    print(f"\nWSDDN EVALUATION RESULTS (Test)")
    print(f"Model: WSDDN_VGG16  |  seed={seed}")
    print("-"*60)
    print("CLASSIFICATION METRICS: N/A")
    print("  (WSDDN is a localization-only WSOD method; binary classification not supported)")
    print("-"*60)
    print(f"DETECTION / LOCALIZATION METRICS (IoU >= {iou_thr}):")
    print(f"  TP_det: {tp_det}, FP_det: {fp_det}, FN_det: {fn_det}")
    print(f"  Precision_det: {precision_det:.4f}   Recall_det: {recall_det:.4f}")
    print(f"  CorLoc (>= {loc_threshold}): {corloc:.2f}%")
    print(f"  mAP@IoU{iou_thr}: {map_value:.4f}")
    print("="*60)

    return {
        "seed": seed,
        # Classification: N/A for WSDDN
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
# Output tables (same format as run_tscam.py / run_pcl.py)
# ---------------------------------------------------------------------------

def print_individual_results(all_results: list, iou_threshold: float):
    print(f"\n{'='*100}\nINDIVIDUAL RESULTS - WSDDN_VGG16\n{'='*100}")

    print(f"\nCLASSIFICATION METRICS: N/A (WSDDN is a localization-only WSOD method)")

    print(f"\nDETECTION / LOCALIZATION METRICS (IoU >= {iou_threshold})")
    print("-"*100)
    print(f"{'Seed':<8} {'TP_det':<8} {'FP_det':<8} {'FN_det':<8} "
          f"{'Prec_det':<12} {'Recall_det':<12} {'CorLoc':<12} {'mAP':<12}")
    print("-"*100)
    for r in all_results:
        print(f"{r.get('seed','?'):<8} "
              f"{r.get('tp_det','N/A'):<8} {r.get('fp_det','N/A'):<8} "
              f"{r.get('fn_det','N/A'):<8} "
              f"{r['precision_det']:<12.4f} {r['recall_det']:<12.4f} "
              f"{r['corloc']:<12.2f} {r['map']:<12.4f}")


def print_summary_statistics(all_results: list, iou_threshold: float):
    n = len(all_results)
    print(f"\n{'='*100}\nSUMMARY STATISTICS - WSDDN_VGG16 (n={n} seeds)\n{'='*100}")

    def calc_stats(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        if not vals: return None, None
        return np.mean(vals), np.std(vals)

    print(f"\nCLASSIFICATION METRICS: N/A (WSDDN is a localization-only WSOD method)")

    print(f"\nDETECTION / LOCALIZATION METRICS (IoU >= {iou_threshold})")
    print("-"*100)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Mean ± Std':<30}")
    print("-"*100)
    for metric, label in [("tp_det","TP_det"),("fp_det","FP_det"),("fn_det","FN_det")]:
        m, s = calc_stats(metric)
        if m is not None: print(f"{label:<20} {m:<15.1f} {s:<15.1f} {m:.1f} ± {s:.1f}")
        else:             print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")
    print("-"*100)
    for metric, label in [("precision_det","Precision_det"),("recall_det","Recall_det"),
                           ("corloc","CorLoc (%)"),("map","mAP")]:
        m, s = calc_stats(metric)
        if m is not None: print(f"{label:<20} {m:<15.4f} {s:<15.4f} {m:.4f} ± {s:.4f}")
        else:             print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A'}")

    print(f"\n{'='*100}\n\nPAPER-READY FORMAT:")
    print("-"*60)
    for metric, label, fmt in [
        ("map",    f"mAP@{iou_threshold}", lambda m,s: f"{m:.4f} ± {s:.4f}"),
        ("corloc", "CorLoc",              lambda m,s: f"{m:.2f}% ± {s:.2f}%"),
    ]:
        m, s = calc_stats(metric)
        if m is not None: print(f"{label:<12}: {fmt(m,s)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WSDDN multi-seed training and evaluation")
    parser.add_argument("--run", type=int, default=1,
                        help="Number of runs with random seeds (default: 1)")
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+",
                        default=None, help="Specific seeds (overrides --run)")
    parser.add_argument("--max-iter", type=int, default=20000,
                        help="Max training iterations per seed (default: 20000)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0, safer for MPS/macOS)")
    parser.add_argument("--iou-threshold",  type=float, default=0.5)
    parser.add_argument("--loc-threshold",  type=float, default=0.5)
    parser.add_argument("--cls-threshold",  type=float, default=0.5,
                        help="Fire score threshold for binary classification (default: 0.5)")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, re-evaluate existing checkpoints")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train, skip evaluation")
    parser.add_argument("--force-prep", action="store_true",
                        help="Force re-running data preparation")
    parser.add_argument("--skip-prep",  action="store_true",
                        help="Skip data preparation step entirely")

    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [
        random.randint(0, 2**31 - 1) for _ in range(args.run)
    ]
    if args.seeds is None:
        print(f"Using random seeds: {seeds}")

    print(f"\n{'#'*60}\nWSDDN MULTI-SEED EXPERIMENT\n{'#'*60}")
    print(f"Model:    WSDDN_VGG16")
    print(f"Dataset:  data_fire")
    print(f"Seeds:    {seeds}")
    print(f"MaxIter:  {args.max_iter}")
    print(f"IoU thr:  {args.iou_threshold}")
    print(f"ClsThr:   {args.cls_threshold}")
    print(f"{'#'*60}\n")

    if not args.skip_prep:
        prepare_data(force=args.force_prep)

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")

        train_output_dir = _OUTPUTS_DIR / f"wsddn_seed_{seed}"
        test_output_dir  = train_output_dir / "test"

        # Train
        if not args.eval_only:
            train(seed=seed, max_iter=args.max_iter, num_workers=args.num_workers)
            gc.collect()

        # Test inference + evaluate
        if not args.train_only:
            ckpt = _find_best_ckpt(train_output_dir)
            if ckpt is None:
                print(f"  No checkpoint found in {train_output_dir} — skipping")
                continue
            print(f"  Using checkpoint: {ckpt}")

            if not args.eval_only or not (test_output_dir / "detections.pkl").exists():
                test_inference(seed=seed, ckpt_path=ckpt,
                               test_output_dir=test_output_dir)

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

    if all_results and not args.train_only:
        print_individual_results(all_results, args.iou_threshold)
        print_summary_statistics(all_results, args.iou_threshold)


if __name__ == "__main__":
    main()
