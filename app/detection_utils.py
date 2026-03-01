"""Detection utilities for WSOD with Vision Transformers.

Provides attention map extraction, dynamic thresholding, and
bounding box generation from attention heatmaps via connected components.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


BBox = Tuple[int, int, int, int, float]  # (x1, y1, x2, y2, confidence)


def get_last_layer_attention(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    patch_size: int = 14,
) -> np.ndarray:
    """Extract attention map from the last transformer layer.

    Extracts attention weights from the final layer, focusing on how
    the CLS token attends to spatial patches.

    Args:
        model: Transformer model with get_last_self_attention() method.
        img_tensor: Input image tensor [B, C, H, W].
        patch_size: Patch size used by the backbone.

    Returns:
        Attention heatmap [H, W] normalized to [0, 1].
    """
    with torch.no_grad():
        last_attn = model.get_last_self_attention(img_tensor)

        # Token order: [CLS(0), reg1(1)..reg4(4), patches(5+)]
        _B, _heads, tokens, _ = last_attn.shape
        num_reg = 4 if tokens > 5 else 0

        cls_attn = last_attn.mean(dim=1)[0, 0, 1 + num_reg:]  # CLS -> patches

        _, _, H, W = img_tensor.shape
        h_featmap = H // patch_size
        w_featmap = W // patch_size
        n_patches = h_featmap * w_featmap

        cur = cls_attn.shape[0]
        if cur != n_patches:
            if cur > n_patches:
                cls_attn = cls_attn[:n_patches]
            else:
                pad = torch.zeros(n_patches - cur, device=cls_attn.device)
                cls_attn = torch.cat([cls_attn, pad], dim=0)

        attn_map = cls_attn.reshape(h_featmap, w_featmap)

        attn_up = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        attn_np = attn_up.detach().cpu().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

        return attn_np


def compute_dynamic_threshold(
    gradcam: np.ndarray,
    method: str = "otsu",
    percentile: int = 85,
    fire_adapted: bool = True,
) -> float:
    """Calculate dynamic threshold based on activation map.

    Optimized for fire detection — automatically adapts to activation intensity.

    Args:
        gradcam: Activation heatmap, values in [0, 1] or [0, 255].
        method: Calculation method ("otsu", "percentile", "kmeans", "adaptive").
        percentile: Percentile to use when method="percentile".
        fire_adapted: If True, apply fire-specific heuristics.

    Returns:
        Dynamic threshold in [0, 1].

    Raises:
        ValueError: If unknown method is specified.
    """
    gradcam_normalized = gradcam / 255.0 if gradcam.max() > 1.0 else gradcam.copy()
    active_pixels = gradcam_normalized[gradcam_normalized > 0.01]

    if len(active_pixels) == 0:
        return 0.3

    if method == "otsu":
        heatmap_8bit = np.uint8(255 * gradcam_normalized)
        otsu_value, _ = cv2.threshold(heatmap_8bit, 0, 255, cv2.THRESH_OTSU)
        threshold = otsu_value / 255.0

    elif method == "percentile":
        # active_pixels are already in [0, 1]
        threshold = float(
            np.percentile(active_pixels, percentile)
            if percentile != 100
            else active_pixels.max()
        )

    elif method == "kmeans":
        pixels = active_pixels.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        threshold = float(centers.min()) if len(centers) > 0 else 0.4

    elif method == "adaptive":
        mean_val = float(np.mean(active_pixels))
        std_val = float(np.std(active_pixels))
        threshold = max(0.2, min(0.8, mean_val + 0.3 * std_val))

    else:
        raise ValueError(f"Unknown threshold method: {method}")

    if fire_adapted:
        max_activation = float(gradcam_normalized.max())
        activation_ratio = np.sum(gradcam_normalized > 0.3) / gradcam_normalized.size

        if activation_ratio > 0.5:
            threshold = min(0.9, threshold * 1.2)
        if max_activation > 0.9:
            threshold = max(0.2, threshold * 0.85)

        threshold = float(np.clip(threshold, 0.25, 0.75))

    return float(threshold)


def greedy_merge_bboxes(
    bboxes: List[BBox],
    iou_threshold: float = 0.3,
    distance_threshold: float = 50.0,
) -> List[BBox]:
    """Greedy algorithm to merge nearby or overlapping bounding boxes.

    Strategy:
        1. Sort boxes by area descending.
        2. For each box, merge with any neighbor whose IoU or edge distance
           satisfies the thresholds, expanding the current box to the union.

    Args:
        bboxes: List of (x1, y1, x2, y2, confidence) tuples.
        iou_threshold: Minimum IoU to trigger a merge.
        distance_threshold: Maximum edge-to-edge distance in pixels to trigger a merge.

    Returns:
        Merged list of (x1, y1, x2, y2, confidence) tuples.
    """
    if len(bboxes) <= 1:
        return bboxes

    def get_iou(box1: BBox, box2: BBox) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]

        inter_x_min = max(x1_1, x1_2)
        inter_y_min = max(y1_1, y1_2)
        inter_x_max = min(x2_1, x2_2)
        inter_y_max = min(y2_1, y2_2)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def get_edge_distance(box1: BBox, box2: BBox) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        dx = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
        dy = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))
        return float((dx**2 + dy**2) ** 0.5)

    def area(bb: BBox) -> int:
        return (bb[2] - bb[0]) * (bb[3] - bb[1])

    sorted_bboxes = sorted(bboxes, key=area, reverse=True)
    used = [False] * len(sorted_bboxes)
    merged: List[BBox] = []

    for i in range(len(sorted_bboxes)):
        if used[i]:
            continue
        current = sorted_bboxes[i]
        for j in range(i + 1, len(sorted_bboxes)):
            if used[j]:
                continue
            if (
                get_iou(current, sorted_bboxes[j]) >= iou_threshold
                or get_edge_distance(current, sorted_bboxes[j]) <= distance_threshold
            ):
                x1 = min(current[0], sorted_bboxes[j][0])
                y1 = min(current[1], sorted_bboxes[j][1])
                x2 = max(current[2], sorted_bboxes[j][2])
                y2 = max(current[3], sorted_bboxes[j][3])
                conf = (current[4] + sorted_bboxes[j][4]) / 2.0
                current = (x1, y1, x2, y2, conf)
                used[j] = True
        merged.append(current)
        used[i] = True

    return merged


def generate_bounding_box(
    gradcam: np.ndarray,
    threshold: Optional[float] = None,
    min_area: int = 200,
    greedy_merge: bool = True,
    iou_threshold: float = 0.3,
    distance_threshold: float = 50.0,
    dynamic_threshold: bool = True,
    threshold_method: str = "otsu",
) -> List[BBox]:
    """Generate bounding boxes from a GradCAM/attention heatmap.

    Pipeline:
        1. Compute dynamic threshold if not provided.
        2. Binarize and extract connected components.
        3. Filter by minimum area.
        4. Optionally merge nearby/overlapping boxes.
        5. Filter by minimum area again after merge.

    Args:
        gradcam: Activation heatmap normalized to [0, 1].
        threshold: Binarization threshold. If None, computed dynamically.
        min_area: Minimum bounding box area in pixels.
        greedy_merge: Whether to merge nearby boxes after generation.
        iou_threshold: IoU threshold for greedy merge.
        distance_threshold: Edge-distance threshold in pixels for greedy merge.
        dynamic_threshold: Compute threshold dynamically when threshold is None.
        threshold_method: Method for dynamic threshold
            ("otsu", "percentile", "adaptive", "kmeans").

    Returns:
        List of (x1, y1, x2, y2, confidence) tuples.
    """
    if threshold is None:
        threshold = (
            compute_dynamic_threshold(gradcam, method=threshold_method, fire_adapted=True)
            if dynamic_threshold
            else 0.5
        )

    binary = (gradcam >= threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    bboxes: List[BBox] = []
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        confidence = float(gradcam[labels == i].mean())
        bboxes.append((int(x), int(y), int(x + w), int(y + h), confidence))

    def _area(bb: BBox) -> int:
        return (bb[2] - bb[0]) * (bb[3] - bb[1])

    bboxes = [bb for bb in bboxes if _area(bb) >= min_area]

    if greedy_merge and len(bboxes) > 1:
        bboxes = greedy_merge_bboxes(
            bboxes,
            iou_threshold=iou_threshold,
            distance_threshold=distance_threshold,
        )
        bboxes = [bb for bb in bboxes if _area(bb) >= min_area]

    return bboxes
