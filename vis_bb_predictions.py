#!/usr/bin/env python3
"""
Visualize bounding box predictions from PCL, WSDDN and TSCAM on test images.

A FIXED set of N images is selected first; every model then processes
exactly those same files.

Each model gets its own subfolder under vis_predictions/:

  vis_predictions/
  ├── PCL/
  │   ├── fire_correct/   ← fire image, best IoU >= iou_thresh
  │   ├── fire_wrong/     ← fire image, best IoU <  iou_thresh
  │   └── notfire/        ← non-fire image
  ├── WSDDN/
  │   ├── fire_correct/
  │   ├── fire_wrong/
  │   └── notfire/
  └── TSCAM/
      ├── fire_correct/
      ├── fire_wrong/
      └── notfire/

Legend on every image:
  GREEN box  → Ground-truth annotation
  RED   box  → Model prediction (labelled with confidence + IoU)

Usage:
    uv run vis_bb_predictions.py --model all
    uv run vis_bb_predictions.py --model pcl --seed 104329020
    uv run vis_bb_predictions.py --model wsddn
    uv run vis_bb_predictions.py --model tscam --checkpoint models/TSCAM_small_seed_42.pth
    uv run vis_bb_predictions.py --model all --max-images 20 --iou-thresh 0.5
    uv run vis_bb_predictions.py --model all --max-images 20 --fire-only
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).resolve().parent
_DATA_FIRE = _ROOT / "data_fire"
_PCL_ANN   = (_ROOT / "core" / "pcl.pytorch" / "data" / "fire"
              / "annotations" / "fire_test.json")
_OUTPUTS   = _ROOT / "Outputs"
_VIS_DIR   = _ROOT / "vis_predictions"

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50
THICKNESS  = 2
GT_COLOR   = (34, 177, 76)   # green (BGR)
PRED_COLOR = (30,  30, 220)  # red   (BGR)


def _load_bgr(path: str) -> np.ndarray | None:
    return cv2.imread(str(path))


def _draw_box(img: np.ndarray, box, color, label: str = ""):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def _caption(img: np.ndarray, text: str):
    cv2.putText(img, text, (8, 22), FONT, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), FONT, 0.48, (0,   0,   0),   1, cv2.LINE_AA)


def _save(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path.with_suffix(".png")), img,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _iou(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih   = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter    = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _best_iou(pred_box, gt_boxes) -> float:
    return max((_iou(pred_box, gt) for gt in gt_boxes), default=0.0)


def _nms(dets: np.ndarray, thresh: float) -> np.ndarray:
    if len(dets) == 0:
        return dets
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iou = (np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1) /
               (areas[i] + areas[order[1:]] -
                np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1) + 1e-9))
        order = order[np.where(iou <= thresh)[0] + 1]
    return dets[keep]


# ---------------------------------------------------------------------------
# GT loading & common image selection
# ---------------------------------------------------------------------------

def _load_gt() -> dict[str, list]:
    """Return {fname: [[x1,y1,x2,y2], ...]} from fire_test.json."""
    if not _PCL_ANN.exists():
        print(f"WARNING: GT file not found: {_PCL_ANN}")
        return {}
    with open(_PCL_ANN) as f:
        coco = json.load(f)
    id_to_fname   = {img["id"]: img["file_name"] for img in coco["images"]}
    fname_to_boxes: dict[str, list] = {img["file_name"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        fname = id_to_fname.get(ann["image_id"])
        if fname:
            x, y, w, h = ann["bbox"]
            fname_to_boxes[fname].append([x, y, x + w, y + h])
    return fname_to_boxes


def select_common_images(n: int, fire_only: bool,
                         seed: int = 0) -> list[str]:
    """
    Return a sorted, reproducible list of N filenames that:
      - exist on disk in data_fire/Test/images/
      - are present in GT annotations
      - optionally only fire images (those with at least 1 GT box)

    The same list is used by every model so comparisons are fair.
    """
    fname_to_bboxes = _load_gt()
    test_img_dir    = _DATA_FIRE / "Test" / "images"

    pool = []
    for fname, boxes in fname_to_bboxes.items():
        if fire_only and not boxes:
            continue
        if (test_img_dir / fname).exists():
            pool.append(fname)

    # reproducible shuffle then take first N
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = sorted(pool[:n])   # sort alphabetically so order is stable
    print(f"[common] Selected {len(selected)} images "
          f"({'fire only' if fire_only else 'fire + notfire'}) "
          f"from {len(pool)} available.")
    return selected


# ---------------------------------------------------------------------------
# Shared image-level draw + save
# ---------------------------------------------------------------------------

def _process_image(fname: str, orig_img: np.ndarray, gt_boxes: list,
                   top_dets: list, model_name: str,
                   out_model_dir: Path, iou_thresh: float,
                   show_gt: bool = False):
    """
    Draw predicted (red) boxes and optionally GT (green) boxes.
    All images are saved directly in out_model_dir (no subdirectories).
    top_dets: list of [x1,y1,x2,y2,score]
    """
    vis = orig_img.copy()

    if show_gt:
        for gt_b in gt_boxes:
            _draw_box(vis, gt_b, GT_COLOR, "GT")

    for det in top_dets:
        _draw_box(vis, det[:4], PRED_COLOR, "pred")

    _save(vis, out_model_dir / fname)


# ---------------------------------------------------------------------------
# GT-only visualisation
# ---------------------------------------------------------------------------

def visualise_gt(common_fnames: list[str], out_root: Path,
                 fname_to_bboxes: dict):
    """Save images with only ground-truth boxes (green) to out_root/gt/."""
    test_img_dir = _DATA_FIRE / "Test" / "images"
    out_dir      = out_root / "gt"
    saved = 0

    for fname in common_fnames:
        gt_boxes = fname_to_bboxes.get(fname, [])
        orig = _load_bgr(str(test_img_dir / fname))
        if orig is None:
            continue
        vis = orig.copy()
        for gt_b in gt_boxes:
            _draw_box(vis, gt_b, GT_COLOR, "GT")
        _save(vis, out_dir / fname)
        saved += 1

    print(f"[GT] Saved {saved} images → {out_dir}/")


# ===========================================================================
# PCL
# ===========================================================================

def _pcl_extract(raw_boxes, nms_thresh: float = 0.3) -> np.ndarray:
    scores      = raw_boxes["scores"]   # (N, 2)
    fire_scores = scores[:, 1]
    fire_boxes  = raw_boxes["boxes"][:, 4:8]   # (N, 4)
    inds = np.where(fire_scores > 0)[0]
    if len(inds) == 0:
        inds = np.array([int(np.argmax(fire_scores))])
    dets = np.hstack([fire_boxes[inds],
                      fire_scores[inds, np.newaxis]]).astype(np.float32)
    return _nms(dets, nms_thresh)


def visualise_pcl(common_fnames: list[str], seed_arg: int | None,
                  iou_thresh: float, n_pred: int, out_root: Path,
                  fname_to_bboxes: dict, show_gt: bool = False):

    candidates = ([_OUTPUTS / f"fire_seed_{seed_arg}"] if seed_arg is not None
                  else sorted(_OUTPUTS.glob("fire_seed_*")))
    det_pkl = used_dir = None
    for c in candidates:
        p = c / "test" / "detections.pkl"
        if p.exists():
            det_pkl, used_dir = p, c
            break

    if det_pkl is None:
        print("[PCL] No detections.pkl found — run run_pcl.py first.")
        return

    print(f"[PCL] {det_pkl}")
    with open(det_pkl, "rb") as f:
        data = pickle.load(f)

    # Build lookup: fname → raw_boxes
    fname_to_raw: dict[str, dict] = {}
    for img_path, raw in data["all_boxes"].items():
        fname_to_raw[Path(str(img_path)).name] = raw

    test_img_dir = _DATA_FIRE / "Test" / "images"
    out_dir      = out_root / "PCL"
    model_name   = f"PCL (seed {used_dir.name.split('_')[-1]})"
    saved = 0

    for fname in common_fnames:
        raw = fname_to_raw.get(fname)
        if raw is None:
            print(f"  [PCL] WARNING: {fname} not found in detections.pkl")
            continue
        orig = _load_bgr(str(test_img_dir / fname))
        if orig is None:
            continue
        dets = _pcl_extract(raw)
        _process_image(fname, orig, fname_to_bboxes.get(fname, []),
                       list(dets[:n_pred]), model_name, out_dir, iou_thresh,
                       show_gt=show_gt)
        saved += 1

    print(f"[PCL] Saved {saved} images → {out_dir}/")


# ===========================================================================
# WSDDN
# ===========================================================================

def _wsddn_extract(scores_tensor, proposals, nms_thresh: float = 0.3) -> np.ndarray:
    scores = (scores_tensor.numpy()
              if hasattr(scores_tensor, "numpy")
              else np.array(scores_tensor))
    fire_scores = scores[:, 1]
    if proposals is None or len(proposals) == 0:
        return np.empty((0, 5), dtype=np.float32)
    dets = np.hstack([proposals, fire_scores[:, np.newaxis]]).astype(np.float32)
    return _nms(dets, nms_thresh)


def visualise_wsddn(common_fnames: list[str], seed_arg: int | None,
                    iou_thresh: float, n_pred: int, out_root: Path,
                    fname_to_bboxes: dict, show_gt: bool = False):

    candidates = ([_OUTPUTS / f"wsddn_seed_{seed_arg}"] if seed_arg is not None
                  else sorted(_OUTPUTS.glob("wsddn_seed_*")))
    det_pkl = used_dir = None
    for c in candidates:
        p = c / "test" / "detections.pkl"
        if p.exists():
            det_pkl, used_dir = p, c
            break

    if det_pkl is None:
        print("[WSDDN] No detections.pkl found — run run_wsddn.py first.")
        return

    print(f"[WSDDN] {det_pkl}")
    with open(det_pkl, "rb") as f:
        data = pickle.load(f)

    # Build lookup: fname → (scores_tensor, proposals)
    fname_to_scores: dict = {}
    fname_to_props:  dict = {}
    for img_path, st in data["all_boxes"].items():
        fn = Path(str(img_path)).name
        fname_to_scores[fn] = st
    for img_path, props in data.get("all_proposals", {}).items():
        fn = Path(str(img_path)).name
        fname_to_props[fn] = props

    test_img_dir = _DATA_FIRE / "Test" / "images"
    out_dir      = out_root / "WSDDN"
    model_name   = f"WSDDN (seed {used_dir.name.split('_')[-1]})"
    saved = 0

    for fname in common_fnames:
        st = fname_to_scores.get(fname)
        if st is None:
            print(f"  [WSDDN] WARNING: {fname} not found in detections.pkl")
            continue
        orig = _load_bgr(str(test_img_dir / fname))
        if orig is None:
            continue
        props = fname_to_props.get(fname)
        dets  = _wsddn_extract(st, props)
        _process_image(fname, orig, fname_to_bboxes.get(fname, []),
                       list(dets[:n_pred]), model_name, out_dir, iou_thresh,
                       show_gt=show_gt)
        saved += 1

    print(f"[WSDDN] Saved {saved} images → {out_dir}/")


# ===========================================================================
# TSCAM
# ===========================================================================

def _extract_cam_np(cam_tensor, class_idx: int, img_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    cam = cam_tensor[class_idx]
    lo, hi = cam.min(), cam.max()
    cam = (cam - lo) / (hi - lo) if hi > lo else torch.zeros_like(cam)
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear", align_corners=False,
    )[0, 0]
    return cam_up.detach().cpu().numpy().clip(0.0, 1.0)


def visualise_tscam(common_fnames: list[str],
                    checkpoint: str | None, seed_arg: int | None, variant: str,
                    iou_thresh: float, n_pred: int, out_root: Path,
                    fname_to_bboxes: dict, show_gt: bool = False):

    # locate checkpoint
    if checkpoint and Path(checkpoint).exists():
        ckpt_path = str(checkpoint)
    else:
        cands = (
            [_ROOT / "models" / f"TSCAM_{variant}_seed_{seed_arg}.pth"]
            if seed_arg is not None
            else sorted((_ROOT / "models").glob(f"TSCAM_{variant}_seed_*.pth"))
        )
        ckpt_path = next((str(c) for c in cands if Path(c).exists()), None)

    if ckpt_path is None:
        print(f"[TSCAM] No checkpoint found for variant='{variant}'. "
              f"Train first:  uv run run_tscam.py --variant {variant}")
        return

    print(f"[TSCAM] checkpoint: {ckpt_path}")

    import torch

    sys.path.insert(0, str(_ROOT))
    tscam_lib = str(_ROOT / "core" / "TSCAM" / "lib")
    if tscam_lib not in sys.path:
        sys.path.append(tscam_lib)

    import models.deit  # noqa: registers deit_tscam_* via timm
    from timm.models import create_model as timm_create

    IMG_SIZE = 224
    VARIANT_MAP = {
        "tiny":  "deit_tscam_tiny_patch16_224",
        "small": "deit_tscam_small_patch16_224",
        "base":  "deit_tscam_base_patch16_224",
    }

    from core.config.config import get_active_dataset_config, get_config
    num_classes = get_active_dataset_config("num_classes") or 2

    model = timm_create(VARIANT_MAP[variant], pretrained=False,
                        num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state.get("model", state))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in sd.items()
         if k in model_sd and model_sd[k].shape == v.shape},
        strict=False,
    )
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)

    from core.datasets import get_data_loaders
    from core.vis_utils import VisUtils, compute_dynamic_threshold

    root_path = (get_active_dataset_config("data_path")
                 or get_active_dataset_config("root_path") or "data/")
    data_loaders, _ = get_data_loaders(
        root_path=root_path, batch_size=1, num_workers=0, img_size=IMG_SIZE,
    )
    test_loader = data_loaders.get("Test")
    if test_loader is None:
        print("[TSCAM] Test loader not found.")
        return

    # Build set for fast lookup
    common_set   = set(common_fnames)
    test_img_dir = _DATA_FIRE / "Test" / "images"
    out_dir      = out_root / "TSCAM"
    seed_tag     = Path(ckpt_path).stem.split("seed_")[-1]
    model_name   = f"TSCAM_{variant} (seed {seed_tag})"
    processed    = set()

    with torch.no_grad():
        for batch_data in test_loader:
            if len(processed) >= len(common_fnames):
                break
            if len(batch_data) < 3:
                continue

            images, labels, image_paths = (
                batch_data[0], batch_data[1], batch_data[2]
            )
            fname = Path(str(
                image_paths[0] if isinstance(image_paths, (list, tuple))
                else image_paths[0]
            )).name

            # skip images not in the common set
            if fname not in common_set or fname in processed:
                continue

            images = images.to(device)
            label  = labels[0].item() if hasattr(labels[0], "item") else int(labels[0])

            orig = _load_bgr(str(test_img_dir / fname))
            if orig is None:
                fallback = (image_paths[0] if isinstance(image_paths, (list, tuple))
                            else image_paths[0])
                orig = _load_bgr(str(fallback))
            if orig is None:
                continue
            H_orig, W_orig = orig.shape[:2]

            output = model(images, return_cam=True)
            logits, cams = (
                (output[0], output[1]) if isinstance(output, tuple)
                else (output, None)
            )

            gt_boxes  = fname_to_bboxes.get(fname, [])

            pred_dets_orig: list = []
            if cams is not None:
                attn_np   = _extract_cam_np(cams[0], label, IMG_SIZE)
                threshold = compute_dynamic_threshold(attn_np, method="otsu",
                                                      fire_adapted=True)
                min_area  = get_config("defaults", "min_bbox_area") or 1600
                bbs       = VisUtils.generate_bounding_box(
                    attn_np, threshold=threshold, min_area=min_area)
                if len(bbs) > 1:
                    from core.debug import remove_nested_bboxes
                    bbs = remove_nested_bboxes(bbs, containment_threshold=0.5)
                sx, sy = W_orig / IMG_SIZE, H_orig / IMG_SIZE
                for bb in bbs[:n_pred]:
                    score = bb[4] if len(bb) > 4 else 1.0
                    pred_dets_orig.append(
                        [bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy, score])

            _process_image(fname, orig, gt_boxes,
                           pred_dets_orig, model_name,
                           out_dir, iou_thresh,
                           show_gt=show_gt)
            processed.add(fname)

    print(f"[TSCAM] Saved {len(processed)} images → {out_dir}/")


# ===========================================================================
# Main
# ===========================================================================

# ===========================================================================
# GradCAM (ResNet50)
# ===========================================================================

def visualise_gradcam(common_fnames: list[str],
                      checkpoint: str | None, seed_arg: int | None,
                      iou_thresh: float, n_pred: int, out_root: Path,
                      fname_to_bboxes: dict, show_gt: bool = False):

    sys.path.insert(0, str(_ROOT))

    # locate checkpoint
    if checkpoint and Path(checkpoint).exists():
        ckpt_path = str(checkpoint)
    else:
        cands = (
            [_ROOT / "models" / f"ResNet50_GradCAM_seed_{seed_arg}.pth"]
            if seed_arg is not None
            else sorted((_ROOT / "models").glob("ResNet50_GradCAM_seed_*.pth"))
        )
        ckpt_path = next((str(c) for c in cands if Path(c).exists()), None)

    if ckpt_path is None:
        print("[GradCAM] No checkpoint found. "
              "Train first:  uv run run_gradcam_resnet50.py")
        return

    print(f"[GradCAM] checkpoint: {ckpt_path}")

    import torch
    import torch.nn.functional as F

    from core.config.config import get_active_dataset_config, get_config
    from core.original_models.resnet50_gradcam import ResNet50GradCAM
    from core.vis_utils import VisUtils, compute_dynamic_threshold

    IMG_SIZE   = 224
    num_classes = get_active_dataset_config("num_classes") or 2

    model = ResNet50GradCAM(num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state.get("state_dict", state.get("model", state))
    sd    = {k.replace("module.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in sd.items()
         if k in model_sd and model_sd[k].shape == v.shape},
        strict=False,
    )
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)

    from core.datasets import get_data_loaders

    root_path = (get_active_dataset_config("data_path")
                 or get_active_dataset_config("root_path") or "data/")
    data_loaders, _ = get_data_loaders(
        root_path=root_path, batch_size=1, num_workers=0, img_size=IMG_SIZE,
    )
    test_loader = data_loaders.get("Test")
    if test_loader is None:
        print("[GradCAM] Test loader not found.")
        return

    common_set   = set(common_fnames)
    test_img_dir = _DATA_FIRE / "Test" / "images"
    out_dir      = out_root / "GradCAM"
    seed_tag     = Path(ckpt_path).stem.split("seed_")[-1]
    model_name   = f"GradCAM (seed {seed_tag})"
    processed    = set()

    for batch_data in test_loader:
        if len(processed) >= len(common_fnames):
            break
        if len(batch_data) < 3:
            continue

        images, labels, image_paths = batch_data[0], batch_data[1], batch_data[2]
        fname = Path(str(
            image_paths[0] if isinstance(image_paths, (list, tuple)) else image_paths[0]
        )).name

        if fname not in common_set or fname in processed:
            continue

        images = images.to(device)
        label  = labels[0].item() if hasattr(labels[0], "item") else int(labels[0])

        orig = _load_bgr(str(test_img_dir / fname))
        if orig is None:
            continue
        H_orig, W_orig = orig.shape[:2]

        gt_boxes = fname_to_bboxes.get(fname, [])

        pred_dets_orig: list = []
        # GradCAM requires gradients
        cam_np_raw = model.compute_gradcam(images, label)   # [H_feat, W_feat]
        cam_t = torch.from_numpy(cam_np_raw).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_t, size=(IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam_up.numpy().clip(0.0, 1.0)

        threshold = compute_dynamic_threshold(cam_np, method="otsu",
                                              fire_adapted=True)
        min_area  = get_config("defaults", "min_bbox_area") or 1600
        bbs       = VisUtils.generate_bounding_box(
            cam_np, threshold=threshold, min_area=min_area)
        if len(bbs) > 1:
            from core.debug import remove_nested_bboxes
            bbs = remove_nested_bboxes(bbs, containment_threshold=0.5)

        sx, sy = W_orig / IMG_SIZE, H_orig / IMG_SIZE
        for bb in bbs[:n_pred]:
            score = bb[4] if len(bb) > 4 else 1.0
            pred_dets_orig.append(
                [bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy, score])

        _process_image(fname, orig, gt_boxes, pred_dets_orig,
                       model_name, out_dir, iou_thresh, show_gt=show_gt)
        processed.add(fname)

    print(f"[GradCAM] Saved {len(processed)} images → {out_dir}/")


# ===========================================================================
# DinoV2RS (user's method)
# ===========================================================================

def visualise_dinov2rs(common_fnames: list[str],
                       checkpoint: str | None, seed_arg: int | None,
                       model_name_arg: str,
                       iou_thresh: float, n_pred: int, out_root: Path,
                       fname_to_bboxes: dict, show_gt: bool = False):

    sys.path.insert(0, str(_ROOT))

    # locate checkpoint — priority:
    #   1. explicit --checkpoint arg
    #   2. seeded file  models/{name}_seed_{seed}.pth
    #   3. unseeded file models/{name}.pth  (saved by main.py / load_model)
    if checkpoint and Path(checkpoint).exists():
        ckpt_path = str(checkpoint)
    else:
        cands = (
            [_ROOT / "models" / f"{model_name_arg}_seed_{seed_arg}.pth"]
            if seed_arg is not None
            else sorted((_ROOT / "models").glob(f"{model_name_arg}_seed_*.pth"))
        )
        ckpt_path = next((str(c) for c in cands if Path(c).exists()), None)
        # fallback: unseeded checkpoint saved by main.py
        if ckpt_path is None:
            fallback = _ROOT / "models" / f"{model_name_arg}.pth"
            if fallback.exists():
                ckpt_path = str(fallback)

    if ckpt_path is None:
        print(f"[{model_name_arg}] No checkpoint found. "
              f"Train first:  uv run run_multi_seed.py --model-name {model_name_arg}")
        return

    print(f"[{model_name_arg}] checkpoint: {ckpt_path}")

    import torch

    from core.config.config import get_active_dataset_config, get_config
    from core.models import get_model
    from core.attention_map import get_last_layer_attention
    from core.vis_utils import VisUtils, compute_dynamic_threshold

    # Use the img_size defined in config for this model (e.g. 672 for DinoV2RS_Small)
    _model_cfg = get_config("models", model_name_arg) or {}
    IMG_SIZE    = int(_model_cfg.get("img_size", 224))
    num_classes = get_active_dataset_config("num_classes") or 2

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = get_model(model_name_arg, num_classes=num_classes, device=device)
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state.get("state_dict", state.get("model", state))
    sd    = {k.replace("module.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in sd.items()
         if k in model_sd and model_sd[k].shape == v.shape},
        strict=False,
    )
    model.eval()

    from core.datasets import get_data_loaders

    root_path = (get_active_dataset_config("data_path")
                 or get_active_dataset_config("root_path") or "data/")
    data_loaders, _ = get_data_loaders(
        root_path=root_path, batch_size=1, num_workers=0, img_size=IMG_SIZE,
    )
    test_loader = data_loaders.get("Test")
    if test_loader is None:
        print(f"[{model_name_arg}] Test loader not found.")
        return

    common_set   = set(common_fnames)
    test_img_dir = _DATA_FIRE / "Test" / "images"
    out_dir      = out_root / model_name_arg
    seed_tag     = Path(ckpt_path).stem.split("seed_")[-1]
    label_name   = f"{model_name_arg} (seed {seed_tag})"
    processed    = set()

    for batch_data in test_loader:
        if len(processed) >= len(common_fnames):
            break
        if len(batch_data) < 3:
            continue

        images, labels, image_paths = batch_data[0], batch_data[1], batch_data[2]
        fname = Path(str(
            image_paths[0] if isinstance(image_paths, (list, tuple)) else image_paths[0]
        )).name

        if fname not in common_set or fname in processed:
            continue

        images = images.to(device)
        label  = labels[0].item() if hasattr(labels[0], "item") else int(labels[0])

        orig = _load_bgr(str(test_img_dir / fname))
        if orig is None:
            continue
        H_orig, W_orig = orig.shape[:2]

        gt_boxes = fname_to_bboxes.get(fname, [])

        pred_dets_orig: list = []
        # Attention map is already upsampled to (IMG_SIZE, IMG_SIZE)
        attn_np = get_last_layer_attention(model, model_name_arg, images)

        # CRF refinement if enabled (same as evaluate_wsod)
        crf_config = get_config("crf") or {}
        if crf_config.get("enabled", False):
            from core.gradcam import refine_heatmap_with_crf
            attn_np = refine_heatmap_with_crf(
                attn_np, image=None, use_crf=True, crf_config=crf_config)

        threshold = compute_dynamic_threshold(attn_np, method="otsu",
                                              fire_adapted=True)
        min_area  = get_config("defaults", "min_bbox_area") or 1600
        bbs       = VisUtils.generate_bounding_box(
            attn_np, threshold=threshold, min_area=min_area)
        if len(bbs) > 1:
            from core.debug import remove_nested_bboxes
            bbs = remove_nested_bboxes(bbs, containment_threshold=0.5)

        # Fallback: bbox da imagem inteira (mesmo que evaluate_wsod)
        if not bbs:
            bbs = [(0, 0, IMG_SIZE, IMG_SIZE, 1.0)]

        sx, sy = W_orig / IMG_SIZE, H_orig / IMG_SIZE
        for bb in bbs[:n_pred]:
            conf = bb[4] if len(bb) > 4 else 1.0
            pred_dets_orig.append(
                [bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy, conf])

        _process_image(fname, orig, gt_boxes, pred_dets_orig,
                       label_name, out_dir, iou_thresh, show_gt=show_gt)
        processed.add(fname)

    print(f"[{model_name_arg}] Saved {len(processed)} images → {out_dir}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualise BB predictions for PCL / WSDDN / TSCAM / GradCAM / DinoV2RS "
                    "(same images for every model)"
    )
    parser.add_argument("--model", default="all",
                        choices=["pcl", "wsddn", "tscam", "gradcam", "dinov2rs", "all"],
                        help="Model(s) to visualise (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Model seed/run to use (default: first available)")
    parser.add_argument("--max-images", type=int, default=20,
                        help="Number of shared images to process (default: 20)")
    parser.add_argument("--fire-only", action="store_true",
                        help="Select only fire images (with GT boxes)")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="Seed for reproducible image shuffle (default: 0)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU cut-off for fire_correct/fire_wrong (default: 0.5)")
    parser.add_argument("--n-pred", type=int, default=3,
                        help="Max predicted boxes per image (default: 3)")
    parser.add_argument("--out-dir", default=str(_VIS_DIR),
                        help="Root output directory (default: vis_predictions/)")
    parser.add_argument("--show-gt", action="store_true",
                        help="Draw ground-truth boxes (green) on top of predictions "
                             "(default: off)")
    # TSCAM extras
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for TSCAM / GradCAM / DinoV2RS "
                             "(optional, auto-detected)")
    parser.add_argument("--variant", default="small",
                        choices=["tiny", "small", "base"],
                        help="TSCAM size variant (default: small)")
    # DinoV2RS extras
    parser.add_argument("--dinov2rs-model", default="DinoV2RS_Small",
                        help="DinoV2RS model name (default: DinoV2RS_Small)")

    args    = parser.parse_args()
    out     = Path(args.out_dir)
    sel     = args.model

    print(f"\nOutput root   : {out}")
    print(f"Model(s)      : {sel}")
    print(f"max_images    : {args.max_images} | fire_only: {args.fire_only} "
          f"| iou_thresh: {args.iou_thresh} | n_pred: {args.n_pred} "
          f"| show_gt: {args.show_gt}\n")

    # ---- 1. Build the common image list ----
    common_fnames   = select_common_images(
        n         = args.max_images,
        fire_only = True,
        seed      = args.shuffle_seed,
    )
    fname_to_bboxes = _load_gt()

    if not common_fnames:
        print("No images selected — check data_fire/Test/images/ and GT annotations.")
        return

    # ---- 2. GT-only reference images ----
    visualise_gt(common_fnames, out, fname_to_bboxes)

    # ---- 3. Run each model on those exact images ----
    if sel in ("pcl", "all"):
        visualise_pcl(common_fnames, args.seed,
                      args.iou_thresh, args.n_pred, out, fname_to_bboxes,
                      show_gt=args.show_gt)

    if sel in ("wsddn", "all"):
        visualise_wsddn(common_fnames, args.seed,
                        args.iou_thresh, args.n_pred, out, fname_to_bboxes,
                        show_gt=args.show_gt)

    if sel in ("tscam", "all"):
        visualise_tscam(common_fnames, args.checkpoint, args.seed,
                        args.variant, args.iou_thresh, args.n_pred,
                        out, fname_to_bboxes,
                        show_gt=args.show_gt)

    if sel in ("gradcam", "all"):
        visualise_gradcam(common_fnames, args.checkpoint, args.seed,
                          args.iou_thresh, args.n_pred, out, fname_to_bboxes,
                          show_gt=args.show_gt)

    if sel in ("dinov2rs", "all"):
        visualise_dinov2rs(common_fnames, args.checkpoint, args.seed,
                           args.dinov2rs_model,
                           args.iou_thresh, args.n_pred, out, fname_to_bboxes,
                           show_gt=args.show_gt)

    print(f"\nDone. Images saved in: {out}/")
    print(f"Common image list ({len(common_fnames)} files):")
    for fn in common_fnames:
        print(f"  {fn}")


if __name__ == "__main__":
    main()