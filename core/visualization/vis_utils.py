"""Visualization utilities for fire classification and localization.

This module provides utilities for generating, processing, and evaluating bounding boxes
from GradCAM activation maps. It includes dynamic thresholding methods, multiple bbox
generation algorithms, and localization accuracy metrics.
"""

import os
import cv2
import json
import torch
import numpy as np
from collections import defaultdict

from core.config.config import get_config, log_info


def get_root_path():
    """Return the path of the active dataset.

    Returns:
        str: Path to the data directory of the active dataset.
    """
    return get_config("datasets", get_config("active_dataset"), "data_path")


ROOT_PATH = get_root_path()


def compute_dynamic_threshold(gradcam, method="otsu", percentile=85, fire_adapted=True):
    """Calculate dynamic threshold based on activation map.

    Optimized for fire detection - automatically adapts to activation intensity.

    Args:
        gradcam: Activation heatmap (values between 0 and 1).
        method: Calculation method ("otsu", "percentile", "kmeans", "adaptive").
        percentile: Percentile to use if method="percentile" (default: 85).
        fire_adapted: If True, apply fire-specific heuristics.

    Returns:
        float: Dynamic threshold between 0 and 1.

    Raises:
        ValueError: If unknown method is specified.
    """
    if gradcam.max() > 1.0:
        gradcam_normalized = gradcam / 255.0
    else:
        gradcam_normalized = gradcam.copy()

    active_pixels = gradcam_normalized[gradcam_normalized > 0.01]

    if len(active_pixels) == 0:
        return 0.3

    if method == "otsu":
        heatmap_8bit = np.uint8(255 * gradcam_normalized)
        otsu_value, _ = cv2.threshold(heatmap_8bit, 0, 255, cv2.THRESH_OTSU)
        threshold = otsu_value / 255.0

    elif method == "percentile":
        threshold = np.percentile(active_pixels, percentile) / 255.0 if percentile != 100 else active_pixels.max()

    elif method == "kmeans":
        active_pixels_reshaped = active_pixels.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(active_pixels_reshaped, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        threshold = float(centers.min()) if len(centers) > 0 else 0.4

    elif method == "adaptive":
        mean_val = np.mean(active_pixels)
        std_val = np.std(active_pixels)
        threshold = max(0.2, min(0.8, mean_val + 0.3 * std_val))
    else:
        raise ValueError(f"Unknown method: {method}")

    if fire_adapted:
        max_activation = gradcam_normalized.max()
        activation_ratio = np.sum(gradcam_normalized > 0.3) / gradcam_normalized.size

        if activation_ratio > 0.5:
            threshold = min(0.9, threshold * 1.2)

        if max_activation > 0.9:
            threshold = max(0.2, threshold * 0.85)

        threshold = np.clip(threshold, 0.25, 0.75)

    return float(threshold)


def compute_multi_scale_threshold(gradcam, fire_adapted=True):
    """Calculate multiple thresholds at different scales.

    Captures fire foci at different sizes. Useful when there are multiple
    fire sources of varying sizes.

    Args:
        gradcam: Activation heatmap.
        fire_adapted: Optimize for fire detection.

    Returns:
        dict: Dictionary with thresholds at different scales.
    """
    thresholds = {
        "small_objects": compute_dynamic_threshold(gradcam, method="otsu", fire_adapted=fire_adapted),
        "medium_objects": compute_dynamic_threshold(gradcam, method="percentile", percentile=75, fire_adapted=fire_adapted),
        "large_objects": compute_dynamic_threshold(gradcam, method="percentile", percentile=90, fire_adapted=fire_adapted),
    }

    return thresholds


class VisUtils:
    """Utilities for visualization and bounding box generation from GradCAM.

    This class provides static methods for generating bounding boxes using various
    algorithms (connected components, watershed, grabcut), merging nearby boxes,
    and computing localization metrics.
    """

    @staticmethod
    def _generate_bounding_box_connectedcomponents(gradcam, threshold=0.5):
        """Generate bounding boxes using connectedComponentsWithStats.

        Args:
            gradcam: Activation heatmap (0-1 normalized).
            threshold: Binary threshold for activation (default: 0.5).

        Returns:
            list: List of tuples (x1, y1, x2, y2, confidence_score).
        """
        heatmap = np.uint8(255 * gradcam)
        _, thresh = cv2.threshold(
            heatmap, int(255 * threshold), 255, cv2.THRESH_BINARY
        )
        bboxes = []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            region_mask = (labels == i)
            confidence = float(gradcam[region_mask].mean())
            bboxes.append((x, y, x + w, y + h, confidence))

        return bboxes

    @staticmethod
    def _generate_bounding_box_watershed(gradcam, threshold=0.5):
        """Generate bounding boxes using watershed algorithm.

        Better for overlapping regions, separates close objects.

        Args:
            gradcam: Activation heatmap (0-1 normalized).
            threshold: Binary threshold for activation (default: 0.5).

        Returns:
            list: List of tuples (x1, y1, x2, y2, confidence_score).
        """
        heatmap = np.uint8(255 * gradcam)

        _, thresh = cv2.threshold(heatmap, int(255 * threshold), 255, cv2.THRESH_BINARY)

        if thresh.sum() == 0:
            return []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, sure_fg = cv2.threshold(dist_normalized, int(0.6 * dist_normalized.max()), 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        if sure_fg.sum() == 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
            bboxes = []
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                region_mask = (labels == i)
                confidence = float(gradcam[region_mask].mean())
                bboxes.append((x, y, x + w, y + h, confidence))
            return bboxes

        num_fg, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1

        markers[cleaned == 0] = 0

        img_color = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        bboxes = []

        unique_markers = np.unique(markers)

        for marker_id in unique_markers:
            if marker_id <= 0:
                continue

            mask = (markers == marker_id).astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if w > 0 and h > 0:
                    img_area = gradcam.shape[0] * gradcam.shape[1]
                    bbox_area = w * h
                    if bbox_area > img_area * 0.8:
                        continue

                    region_mask = (markers == marker_id)
                    if region_mask.any():
                        confidence = float(gradcam[region_mask].mean())
                    else:
                        confidence = float(gradcam[y:y+h, x:x+w].mean()) if (y+h <= gradcam.shape[0] and x+w <= gradcam.shape[1]) else 0.5

                    bboxes.append((x, y, x + w, y + h, confidence))

        return bboxes

    @staticmethod
    def _generate_bounding_box_grabcut(gradcam, threshold=0.5):
        """Generate bounding boxes using GrabCut with improved initialization.

        Combines good edges from GrabCut with larger sizes (better GT match).

        Args:
            gradcam: Activation heatmap (0-1 normalized).
            threshold: Binary threshold for activation (default: 0.5).

        Returns:
            list: List of tuples (x1, y1, x2, y2, confidence_score).
        """
        heatmap = np.uint8(255 * gradcam)

        img_bgr = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

        init_threshold = int(255 * threshold * 0.7)
        _, thresh = cv2.threshold(heatmap, init_threshold, 255, cv2.THRESH_BINARY)

        if thresh.sum() == 0:
            return []

        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        GC_BG = 0
        GC_FG = 1
        GC_PR_FG = 3
        GC_INIT_WITH_MASK = 3

        mask = np.zeros(heatmap.shape[:2], np.uint8)
        mask[cleaned == 255] = GC_FG
        mask[cleaned == 0] = GC_BG

        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_fg = (mask == GC_FG).astype(np.uint8) * 255
        mask_fg = cv2.dilate(mask_fg, kernel_expand, iterations=1)
        mask[mask_fg > 0] = GC_FG

        try:
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 10, GC_INIT_WITH_MASK)
        except Exception as e:
            log_info(f"Warning: GrabCut failed ({e}), using connected components as fallback")
            return VisUtils._generate_bounding_box_connectedcomponents(gradcam, threshold)

        foreground_mask = np.where((mask == GC_FG) | (mask == GC_PR_FG), 255, 0).astype(np.uint8)

        if foreground_mask.sum() == 0:
            foreground_mask = np.where(mask == GC_FG, 255, 0).astype(np.uint8)

        if foreground_mask.sum() == 0:
            return VisUtils._generate_bounding_box_connectedcomponents(gradcam, threshold)

        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        foreground_mask = cv2.dilate(foreground_mask, kernel_final, iterations=2)

        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_final, iterations=1)

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []

        raw_bboxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > 0 and h > 0:
                img_area = gradcam.shape[0] * gradcam.shape[1]
                bbox_area = w * h
                if bbox_area > img_area * 0.8:
                    continue

                region_heatmap = gradcam[y:y+h, x:x+w]
                confidence = float(region_heatmap.mean())

                raw_bboxes.append((x, y, x + w, y + h, confidence))

        bboxes = []
        used = set()

        for i, bb1 in enumerate(raw_bboxes):
            if i in used:
                continue

            x1_1, y1_1, x2_1, y2_1, conf1 = bb1

            close_bbs = [(i, bb1)]

            for j, bb2 in enumerate(raw_bboxes):
                if i >= j or j in used:
                    continue

                x1_2, y1_2, x2_2, y2_2, conf2 = bb2

                cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
                cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

                if dist < 80:
                    close_bbs.append((j, bb2))
                    used.add(j)

            if len(close_bbs) > 1:
                all_x1 = min(bb[0] for _, bb in close_bbs)
                all_y1 = min(bb[1] for _, bb in close_bbs)
                all_x2 = max(bb[2] for _, bb in close_bbs)
                all_y2 = max(bb[3] for _, bb in close_bbs)
                avg_conf = np.mean([bb[4] for _, bb in close_bbs])
                bboxes.append((all_x1, all_y1, all_x2, all_y2, avg_conf))
            else:
                bboxes.append(bb1)

            used.add(i)

        return bboxes

    @staticmethod
    def generate_bounding_box(gradcam, threshold=None, min_area=200,
        method=None, greedy_merge=True, iou_threshold=0.1,
        distance_threshold=50, dynamic_threshold=True,
        threshold_method="otsu"):
        """Generate bounding boxes from GradCAM with confidence scores.

        Pipeline:
        1. Calculate dynamic threshold if not specified (based on heatmap).
        2. Generate initial BBs using specified method (filter by min_area).
        3. Apply greedy merge if enabled (consolidate nearby/overlapping BBs).
        4. Filter again by min_area after merge (resulting BBs may be larger).

        Args:
            gradcam: Activation heatmap.
            threshold: Threshold for binarization (None = dynamic).
            min_area: Minimum component area (applied before and after merge).
            method: Generation method ("connectedcomponents", "watershed", or "grabcut").
                   If None, reads from config.yaml.
            greedy_merge: If True, merge nearby BBs using greedy algorithm.
            iou_threshold: IoU threshold for merge (default: 0.1).
            distance_threshold: Maximum distance between edges for merge in pixels (default: 50).
            dynamic_threshold: If True and threshold=None, calculate threshold dynamically.
            threshold_method: Method for calculating dynamic threshold:
                - "otsu": Otsu's method (separates foreground/background well).
                - "percentile": Uses percentile of active pixels.
                - "adaptive": Uses mean + standard deviation.
                - "kmeans": Clusters pixels into 2 groups.

        Returns:
            list: List of tuples (x1, y1, x2, y2, confidence_score).
        """
        if threshold is None and dynamic_threshold:
            threshold = compute_dynamic_threshold(gradcam, method=threshold_method, fire_adapted=True)
            #log_info(f"[VisUtils] Dynamic threshold calculated:
            # {threshold:.4f} (method: {threshold_method})")
        elif threshold is None:
            threshold = 0.5
            #log_info(f"[VisUtils] Using default threshold: {threshold:.4f}")

        if method is None:
            method = get_config("defaults", "bbox_method")
            if method is None:
                method = "connectedcomponents"

        method = method.lower().strip()

        if method == "watershed":
            bboxes = VisUtils._generate_bounding_box_watershed(gradcam, threshold)
        elif method == "grabcut":
            bboxes = VisUtils._generate_bounding_box_grabcut(gradcam, threshold)
        else:
            bboxes = VisUtils._generate_bounding_box_connectedcomponents(gradcam, threshold)

        if min_area is not None and len(bboxes) > 0:
            def get_box_area(box):
                x1, y1, x2, y2 = box[:4]
                return (x2 - x1) * (y2 - y1)
            bboxes = [bb for bb in bboxes if get_box_area(bb) >= min_area]

        # Check if greedy merging is enabled via config (can be overridden by parameter)
        greedy_merge_enabled = get_config('defaults', 'greedy_merge_enabled')
        if greedy_merge_enabled is None:
            greedy_merge_enabled = True
        if greedy_merge_enabled and greedy_merge and len(bboxes) > 1:
            iou_thresh_cfg = get_config('defaults', 'greedy_iou_threshold')
            dist_thresh_cfg = get_config('defaults', 'greedy_distance_threshold')
            iou_thresh = iou_threshold if iou_threshold != 0.1 else (iou_thresh_cfg if iou_thresh_cfg is not None else 0.1)
            dist_thresh = distance_threshold if distance_threshold != 50 else (dist_thresh_cfg if dist_thresh_cfg is not None else 50)
            bbs_bf = len(bboxes)
            bboxes = VisUtils.greedy_merge_bboxes(bboxes, iou_threshold=iou_thresh, distance_threshold=dist_thresh)
            #log_info(f"[Greedy merge] {bbs_bf} → {len(bboxes)} BBs (iou
            # ={iou_thresh}, dist={dist_thresh})")

        return bboxes

    @staticmethod
    def remove_nested_bboxes(bboxes, containment_threshold=0.8):
        """Remove smaller bounding boxes completely inside larger ones.

        Args:
            bboxes: List of BBs in format (x1, y1, x2, y2, confidence) or (x1, y1, x2, y2).
            containment_threshold: Percentage of smaller BB that must be inside larger BB (default: 0.8 = 80%).

        Returns:
            list: Filtered list of BBs (removing smaller ones when overlapping).
        """
        if len(bboxes) <= 1:
            return bboxes

        def get_containment(small_box, large_box):
            """Calculate how much of the smaller BB is inside the larger BB (0-1)."""
            s_x1, s_y1, s_x2, s_y2 = small_box[:4]
            l_x1, l_y1, l_x2, l_y2 = large_box[:4]
            inter_x_min = max(s_x1, l_x1)
            inter_y_min = max(s_y1, l_y1)
            inter_x_max = min(s_x2, l_x2)
            inter_y_max = min(s_y2, l_y2)
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            small_area = (s_x2 - s_x1) * (s_y2 - s_y1)
            return inter_area / small_area if small_area > 0 else 0.0

        def get_box_area(box):
            """Calculate area of a bounding box."""
            x1, y1, x2, y2 = box[:4]
            return (x2 - x1) * (y2 - y1)

        sorted_indices = sorted(range(len(bboxes)),
                                key=lambda i: get_box_area(bboxes[i]),
                                reverse=True)

        keep = [True] * len(bboxes)

        for i, idx_i in enumerate(sorted_indices):
            if not keep[idx_i]:
                continue
            for j in range(i + 1, len(sorted_indices)):
                idx_j = sorted_indices[j]
                if not keep[idx_j]:
                    continue
                if get_containment(bboxes[idx_j], bboxes[idx_i]) >= containment_threshold:
                    keep[idx_j] = False

        return [bboxes[i] for i in range(len(bboxes)) if keep[i]]

    @staticmethod
    def postprocess_BB(bboxes, min_confidence=0.0, remove_nested=True, containment_threshold=0.8):
        """Complete post-processing of bounding boxes.

        Steps:
        1. Filter by min_confidence.
        2. Remove nested BBs.

        Args:
            bboxes: List of BBs in format (x1, y1, x2, y2, confidence).
            min_confidence: Minimum confidence to keep BB (default: 0.0).
            remove_nested: If True, remove nested BBs (default: True).
            containment_threshold: Threshold to consider BB nested (default: 0.8).

        Returns:
            list: List of BBs after post-processing.
        """
        if len(bboxes) == 0:
            return []

        filtered_bbs = [bb for bb in bboxes if (bb[4] if len(bb) > 4 else 0.0) >= min_confidence]

        if remove_nested and len(filtered_bbs) > 1:
            filtered_bbs = VisUtils.remove_nested_bboxes(filtered_bbs, containment_threshold)

        return filtered_bbs

    @staticmethod
    def greedy_merge_bboxes(bboxes, iou_threshold=0.3, distance_threshold=50):
        """Greedy algorithm to merge nearby/overlapping bounding boxes.

        Strategy:
        1. Sort BBs by size (descending).
        2. For each BB, try to merge with neighbors that have IoU > threshold.
        3. Create new BB that encompasses both.

        Args:
            bboxes: List of BBs in format (x1, y1, x2, y2, confidence).
            iou_threshold: IoU threshold to consider merge (0-1).
            distance_threshold: Maximum distance between edges to consider merge (pixels).

        Returns:
            list: List of BBs after merge (usually smaller).
        """
        if len(bboxes) <= 1:
            return bboxes

        def get_iou(box1, box2):
            """Calculate IoU between two boxes."""
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

        def get_edge_distance(box1, box2):
            """Calculate the shortest distance between edges of two boxes.

            Returns 0 if boxes overlap, and the minimum distance between edges otherwise.
            """
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            x1_2, y1_2, x2_2, y2_2 = box2[:4]

            dx = max(x1_1, x1_2) - min(x2_1, x2_2)
            dy = max(y1_1, y1_2) - min(y2_1, y2_2)

            if dx < 0 and dy < 0:
                return 0.0

            if dx > 0 and dy > 0:
                return (dx**2 + dy**2) ** 0.5
            else:
                return max(dx, dy)

        def merge_boxes(box1, box2):
            """Merge two boxes into one that encompasses both."""
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            x1_2, y1_2, x2_2, y2_2 = box2[:4]
            conf1 = box1[4] if len(box1) > 4 else 1.0
            conf2 = box2[4] if len(box2) > 4 else 1.0

            new_x1 = min(x1_1, x1_2)
            new_y1 = min(y1_1, y1_2)
            new_x2 = max(x2_1, x2_2)
            new_y2 = max(y2_1, y2_2)

            new_conf = max(conf1, conf2)

            return (new_x1, new_y1, new_x2, new_y2, new_conf)

        merged_bbs = list(bboxes)
        merged = True

        while merged:
            merged = False
            i = 0
            while i < len(merged_bbs):
                j = i + 1
                while j < len(merged_bbs):
                    box_i = merged_bbs[i]
                    box_j = merged_bbs[j]

                    iou = get_iou(box_i, box_j)
                    edge_dist = get_edge_distance(box_i, box_j)

                    if iou > iou_threshold or edge_dist < distance_threshold:
                        merged_box = merge_boxes(box_i, box_j)
                        merged_bbs[i] = merged_box
                        merged_bbs.pop(j)
                        merged = True
                        break

                    j += 1

                if not merged:
                    i += 1
                else:
                    break

        return merged_bbs

    @staticmethod
    def draw_rectangle(
            tensor: torch.Tensor, bb: tuple, color: tuple = (1, 1, 1), fill: bool = False
    ) -> torch.Tensor:
        """Draw a rectangle on a tensor.

        Args:
            tensor: Input tensor (C, H, W) or (H, W).
            bb: Bounding box coordinates (x1, y1, x2, y2).
            color: RGB color tuple (default: (1, 1, 1) for white).
            fill: If True, fill the rectangle; otherwise draw only edges.

        Returns:
            torch.Tensor: Tensor with drawn rectangle.
        """
        x1, y1, x2, y2 = bb
        x2, y2 = x2 - 1, y2 - 1

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        for c in range(tensor.shape[0]):
            channel_color = color[c] if isinstance(color, tuple) else color

            if fill:
                tensor[c, y1: y2 + 1, x1: x2 + 1] = channel_color
            else:
                tensor[c, y1, x1: x2 + 1] = channel_color
                tensor[c, y2, x1: x2 + 1] = channel_color
                tensor[c, y1: y2 + 1, x1] = channel_color
                tensor[c, y1: y2 + 1, x2] = channel_color

        return tensor

    @staticmethod
    def get_IoU(bbox_a, bbox_b):
        """Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox_a: First bounding box (x1, y1, x2, y2).
            bbox_b: Second bounding box (x1, y1, x2, y2).

        Returns:
            float: IoU value between 0 and 1.
        """
        xa = max(bbox_a[0], bbox_b[0])
        ya = max(bbox_a[1], bbox_b[1])
        xb = min(bbox_a[2], bbox_b[2])
        yb = min(bbox_a[3], bbox_b[3])

        intern_area = max(0, xb - xa) * max(0, yb - ya)
        if intern_area == 0:
            return 0.0

        bbox_a_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        bbox_b_area = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

        return intern_area / float(bbox_a_area + bbox_b_area - intern_area)

    @staticmethod
    def get_newIoU(gt, pred_bb):
        """Calculate modified IoU with additional validation criteria.

        Returns 0 if intersection is less than 75% of both predicted and ground truth areas.

        Args:
            gt: Ground truth bounding box (x1, y1, x2, y2).
            pred_bb: Predicted bounding box (x1, y1, x2, y2).

        Returns:
            float: Modified IoU value between 0 and 1.
        """
        xa = max(gt[0], pred_bb[0])
        ya = max(gt[1], pred_bb[1])
        xb = min(gt[2], pred_bb[2])
        yb = min(gt[3], pred_bb[3])

        intern_area = max(0, xb - xa) * max(0, yb - ya)
        if intern_area == 0:
            return 0.0

        gt_bb_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
        pred_bb_area = (pred_bb[2] - pred_bb[0]) * (pred_bb[3] - pred_bb[1])
        if intern_area < pred_bb_area * 0.75 and intern_area < gt_bb_area * 0.75:
            return 0.0

        return intern_area / gt_bb_area

    @staticmethod
    def get_LocAcc(gt, pred_bb):
        """Calculate Localization Accuracy (MaxBoxAccV2).

        Standard metric in WSOL (Weakly Supervised Object Localization).
        LocAcc = intersection_area / gt_area

        Unlike IoU which divides by union, LocAcc measures how well
        the prediction covers the ground truth (recall-oriented).

        Args:
            gt: Ground truth bounding box (x1, y1, x2, y2).
            pred_bb: Predicted bounding box (x1, y1, x2, y2).

        Returns:
            float: Localization accuracy between 0 and 1.
        """
        xa = max(gt[0], pred_bb[0])
        ya = max(gt[1], pred_bb[1])
        xb = min(gt[2], pred_bb[2])
        yb = min(gt[3], pred_bb[3])

        intersection = max(0, xb - xa) * max(0, yb - ya)
        if intersection == 0:
            return 0.0

        gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

        return intersection / gt_area if gt_area > 0 else 0.0

    @staticmethod
    def redimension_bboxes(bboxes, new_size, or_size=256, original_width=None, original_height=None):
        """Redimension bounding boxes considering Resize + PadResize (with padding).

        Args:
            bboxes: Bounding box (x1, y1, width, height) in original coordinates.
            new_size: Final image size (after padding).
            or_size: Reference dimension (smaller side of original image).
            original_width: Original image width (to calculate padding offset).
            original_height: Original image height (to calculate padding offset).

        Returns:
            tuple: Redimensioned bounding box (x1, y1, width, height).
        """
        x1, y1, width, height = bboxes

        if original_width is None or original_height is None:
            scale = new_size / or_size
            return int(x1 * scale), int(y1 * scale), int(width * scale), int(height * scale)

        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            new_w = new_size
            new_h = int(new_size / aspect_ratio)
        else:
            new_h = new_size
            new_w = int(new_size * aspect_ratio)

        scale_w = new_w / original_width
        scale_h = new_h / original_height

        pad_w = new_size - new_w
        pad_h = new_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        final_x1 = x1 * scale_w + pad_left
        final_y1 = y1 * scale_h + pad_top
        final_width = width * scale_w
        final_height = height * scale_h

        return int(final_x1), int(final_y1), int(final_width), int(final_height)

    @staticmethod
    def get_image_ids(split="Test"):
        """Load bounding box annotations if available.

        Supports three formats:
        1. COCO JSON (annotations.json) - for Fire dataset.
        2. Consolidated annotations.txt file - for NWPU VHR-10 dataset (NEW).
        3. Individual .txt files - fallback for old NWPU format.

        Args:
            split: Dataset split to load annotations from (default: "Test").

        Returns:
            tuple: (bbox_to_id, filename_to_id, image_sizes)
                - bbox_to_id: Dict mapping image_id to list of bboxes.
                - filename_to_id: Dict mapping filename to image_id.
                - image_sizes: Dict mapping image_id to (width, height) original size.
        """
        bbox_to_id = defaultdict(list)
        filename_to_id = {}
        image_sizes = {}

        root_path = get_root_path()
        test_path = os.path.join(root_path, split)

        if not os.path.exists(test_path):
            return bbox_to_id, filename_to_id, image_sizes

        json_found = False

        annotations_path = os.path.join(test_path, "annotations.json")
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, "r") as f:
                    coco_gt = json.load(f)

                for img in coco_gt["annotations"]:
                    bbox = tuple(img["bbox"])
                    if "category_id" in img:
                        bbox = bbox + (img["category_id"],)
                    bbox_to_id[img["image_id"]].append(bbox)
                filename_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}
                for img in coco_gt["images"]:
                    image_sizes[img["id"]] = (img.get("width", 256), img.get("height", 256))
                json_found = True
                log_info(f"Loaded annotations in COCO format: {annotations_path}")
                log_info(f"Images: {len(coco_gt['images'])}, Annotations: {len(coco_gt['annotations'])}")
            except Exception as e:
                log_info(f"Warning: Could not load annotations from {annotations_path}: {e}")

        if not json_found:
            for subdir in os.listdir(test_path):
                subdir_annotations = os.path.join(test_path, subdir, "annotations.json")
                if os.path.exists(subdir_annotations):
                    try:
                        with open(subdir_annotations, "r") as f:
                            coco_gt = json.load(f)

                        for img in coco_gt["annotations"]:
                            bbox = tuple(img["bbox"])
                            if "category_id" in img:
                                bbox = bbox + (img["category_id"],)
                            bbox_to_id[img["image_id"]].append(bbox)
                        filename_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}
                        for img in coco_gt["images"]:
                            image_sizes[img["id"]] = (img.get("width", 256), img.get("height", 256))
                        json_found = True
                        log_info(f"Loaded annotations in COCO format: {subdir_annotations}")
                        log_info(f"Images: {len(coco_gt['images'])}, Annotations: {len(coco_gt['annotations'])}")
                        break
                    except Exception as e:
                        log_info(f"Warning: Could not load annotations from {subdir_annotations}: {e}")

        consolidated_found = False
        if not json_found:
            consolidated_path = os.path.join(test_path, "annotations.txt")
            if os.path.exists(consolidated_path):
                from PIL import Image
                try:
                    with open(consolidated_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split()
                            if len(parts) < 2:
                                continue

                            image_name = parts[0]
                            annotation_str = ' '.join(parts[1:])

                            coords = annotation_str.split(',')
                            if len(coords) >= 5:
                                try:
                                    x1 = int(coords[0].strip('( '))
                                    y1 = int(coords[1].strip(') '))
                                    x2 = int(coords[2].strip('( '))
                                    y2 = int(coords[3].strip(') '))
                                    class_id = int(coords[4].strip())

                                    bbox = (x1, y1, x2 - x1, y2 - y1, class_id)
                                    bbox_to_id[image_name].append(bbox)
                                    filename_to_id[image_name] = image_name

                                    if image_name not in image_sizes:
                                        for subdir in os.listdir(test_path):
                                            subdir_path = os.path.join(test_path, subdir)
                                            if not os.path.isdir(subdir_path):
                                                continue
                                            image_path = os.path.join(subdir_path, image_name)
                                            if os.path.exists(image_path):
                                                try:
                                                    with Image.open(image_path) as img:
                                                        image_sizes[image_name] = img.size
                                                    break
                                                except Exception as e:
                                                    log_info(f"Warning: Could not read image size from {image_path}: {e}")
                                except (ValueError, IndexError) as e:
                                    log_info(f"Warning: Could not parse annotation: {line} - {e}")
                                    continue

                    consolidated_found = True
                    log_info(f"Loaded annotations from consolidated file: {consolidated_path}")
                    log_info(f"  Total images: {len(filename_to_id)}")
                    log_info(f"  Total bounding boxes: {sum(len(bboxes) for bboxes in bbox_to_id.values())}")

                except Exception as e:
                    log_info(f"Warning: Could not load consolidated annotations from {consolidated_path}: {e}")

        txt_count = 0
        if not json_found and not consolidated_found:
            from PIL import Image
            for subdir in os.listdir(test_path):
                subdir_path = os.path.join(test_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                for filename in os.listdir(subdir_path):
                    if filename.endswith('.txt'):
                        txt_path = os.path.join(subdir_path, filename)
                        image_name = filename.replace('.txt', '.jpg')
                        image_path = os.path.join(subdir_path, image_name)

                        if os.path.exists(image_path):
                            try:
                                with Image.open(image_path) as img:
                                    image_sizes[image_name] = img.size
                            except Exception as e:
                                log_info(f"Warning: Could not read image size from {image_path}: {e}")
                                continue

                        try:
                            with open(txt_path, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue

                                    parts = line.split(',')
                                    if len(parts) >= 5:
                                        try:
                                            x1 = int(parts[0].strip('( '))
                                            y1 = int(parts[1].strip(') '))
                                            x2 = int(parts[2].strip('( '))
                                            y2 = int(parts[3].strip(') '))
                                            class_id = int(parts[4].strip())
                                            bbox = (x1, y1, x2 - x1, y2 - y1, class_id)
                                            bbox_to_id[image_name].append(bbox)
                                            filename_to_id[image_name] = image_name
                                        except (ValueError, IndexError):
                                            continue
                        except Exception as e:
                            log_info(f"Warning: Could not parse {txt_path}: {e}")
                            continue

                        txt_count += 1

            if txt_count > 0:
                log_info(f"Loaded annotations from {txt_count} individual .txt files (old NWPU format)")

        if not json_found and not consolidated_found and txt_count == 0:
            log_info("Warning: No annotations found (dataset may be classification-only)")

        return bbox_to_id, filename_to_id, image_sizes

    @staticmethod
    def filter_bboxes_by_classifier(classifier_model, feature, bboxes, device, fire_class=0, confidence_threshold=0.5, img_size=224):
        """Filter bounding boxes by passing cropped regions through the classifier.

        Only keeps bounding boxes that the classifier predicts as "Fire" with high confidence.

        Args:
            classifier_model: Trained classifier model
            feature: Tensor of shape [1, 3, H, W] (single image)
            bboxes: List of bounding boxes in format (x1, y1, x2, y2, confidence)
            device: Device (cuda/cpu)
            fire_class: Class ID for fire (usually 0)
            confidence_threshold: Minimum confidence to keep a bounding box (default: 0.5)
            img_size: Input size for classifier (default: 224)

        Returns:
            filtered_bboxes: Bounding boxes that passed the classifier filter
        """
        if len(bboxes) == 0:
            return bboxes

        filtered_bboxes = []
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        for bb in bboxes:
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            try:
                black_image = torch.zeros_like(feature)
                black_image[:, y1:y2, x1:x2] = feature[:, y1:y2, x1:x2]

                h, w = black_image.shape[1], black_image.shape[2]
                if h != img_size or w != img_size:
                    black_resized = torch.nn.functional.interpolate(
                        black_image.unsqueeze(0),
                        size=(img_size, img_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                else:
                    black_resized = black_image

                with torch.no_grad():
                    black_batch = black_resized.unsqueeze(0).to(device)
                    if isinstance(classifier_model, dict) or hasattr(classifier_model, '__call__'):
                        output = classifier_model(black_batch)
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output

                        pred_probs = torch.softmax(logits, dim=1)
                        pred_class = torch.argmax(pred_probs, dim=1).item()
                        pred_confidence = pred_probs[0, pred_class].item()

                        if pred_class == fire_class and pred_confidence >= confidence_threshold:
                            filtered_bboxes.append(bb)
            except Exception as e:
                filtered_bboxes.append(bb)

        return filtered_bboxes
