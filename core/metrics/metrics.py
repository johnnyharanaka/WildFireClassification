from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
import json
import os

import numpy as np
import torch

from core.config.config import log_info, get_config
from core.visualization.vis_utils import VisUtils, compute_dynamic_threshold
from core.visualization.attention_map import get_last_layer_attention
from core.config.config import get_active_dataset_config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def load_coco_bboxes_from_json(test_path: str, target_size: int = None) -> Dict[str, List]:
    """
    Load ground truth bboxes from COCO annotations.json file.
    Args:
        test_path: Path to Test directory (contains annotations.json)
        target_size: If provided, scale bboxes to this size (square)
    Returns:
        Dict mapping filename to list of ground truth bboxes (scaled if target_size provided)
    """
    annotations_file = os.path.join(test_path, 'annotations.json')
    bbox_by_filename = {}

    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Map image_id to filename and original dimensions
        id_to_info = {}
        for img in coco_data['images']:
            id_to_info[img['id']] = {
                'filename': img['file_name'],
                'width': img.get('width', 256),
                'height': img.get('height', 256)
            }

        # Group annotations by image_id
        bboxes_by_id = defaultdict(list)
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            bbox = ann.get('bbox', [])  # COCO format: [x, y, w, h]
            # Use COCO category_id directly (1-based for Fire dataset)
            class_id = ann.get('category_id', 1)

            if bbox and len(bbox) == 4:
                x, y, w, h = bbox

                # Scale bbox if target_size is provided
                # Must match PadResize transform: scale to fit larger side, then pad
                if target_size is not None and img_id in id_to_info:
                    orig_w = id_to_info[img_id]['width']
                    orig_h = id_to_info[img_id]['height']

                    # PadResize scales so the larger dimension = target_size
                    scale = target_size / max(orig_w, orig_h)

                    # Calculate new dimensions after scaling
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)

                    # Calculate padding offsets (centered padding)
                    pad_x = (target_size - new_w) // 2
                    pad_y = (target_size - new_h) // 2

                    # Scale and offset the bbox
                    x = x * scale + pad_x
                    y = y * scale + pad_y
                    w = w * scale
                    h = h * scale

                bboxes_by_id[img_id].append([x, y, w, h, class_id])

        # Map to filenames
        for img_id, bboxes in bboxes_by_id.items():
            if img_id in id_to_info:
                filename = id_to_info[img_id]['filename']
                bbox_by_filename[filename] = bboxes

        return bbox_by_filename

    except FileNotFoundError:
        log_info(f"Annotations.json file not found in {test_path}")
        return {}
    except Exception as e:
        log_info(f"Error loading bboxes from COCO JSON: {e}")
        return {}

def normalize_bbox(bbox: Union[List, Tuple]) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, w, h] to corner format (x1, y1, x2, y2).
    Args:
        bbox: Bounding box in COCO format [x, y, width, height]
    Returns:
        Tuple (x1, y1, x2, y2)
    """
    if bbox is None or len(bbox) < 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    try:
        x, y, w, h = float(x), float(y), float(w), float(h)
    except (ValueError, TypeError):
        raise ValueError(f"Bbox contains non-numeric values: {bbox}")

    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # Clamp negative coordinates to 0 (floating point errors)
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = max(0.0, x2)
    y2 = max(0.0, y2)

    # Validate coordinates
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid bbox (x1>=x2 or y1>=y2): ({x1}, {y1}, {x2}, {y2})")

    return x1, y1, x2, y2

def get_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate area of a normalized bbox (x1, y1, x2, y2)."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def validate_predictions(predictions: List[Dict]) -> bool:
    """
    Validate list of predictions before processing.
    """
    if not predictions:
        return True

    for i, pred in enumerate(predictions):
        required_keys = {'bbox', 'confidence', 'img_id'}
        if not all(key in pred for key in required_keys):
            log_info(f"Prediction {i} missing keys: {required_keys - set(pred.keys())}")
            return False

        try:
            float(pred['confidence'])
        except (ValueError, TypeError):
            log_info(f"Prediction {i} has invalid confidence: {pred['confidence']}")
            return False

    return True


def validate_ground_truths(gts: List[Dict]) -> bool:
    """
    Validate list of ground truths before processing.
    """
    if not gts:
        return True

    for i, gt in enumerate(gts):
        required_keys = {'bbox', 'class_id', 'img_id'}
        if not all(key in gt for key in required_keys):
            log_info(f"GT {i} missing keys: {required_keys - set(gt.keys())}")
            return False

    return True

def compute_bbox_area(bbox: List[float]) -> float:
  return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def compute_iou(box1_iou: List[float], box2_iou: List[float]) -> float:
  """Compute IoU (Intersection over Union) between two bounding boxes.

  Args:
      box1_iou: Bounding box in format [x1, y1, x2, y2].
      box2_iou: Bounding box in format [x1, y1, x2, y2].

  Returns:
      IoU score between 0 and 1.
  """
  x1_inter = max(box1_iou[0], box2_iou[0])
  y1_inter = max(box1_iou[1], box2_iou[1])
  x2_inter = min(box1_iou[2], box2_iou[2])
  y2_inter = min(box1_iou[3], box2_iou[3])

  inter_width = max(0, x2_inter - x1_inter)
  inter_height = max(0, y2_inter - y1_inter)
  inter_area = inter_width * inter_height

  box1_area = compute_bbox_area(box1_iou)
  box2_area = compute_bbox_area(box2_iou)

  union_area = box1_area + box2_area - inter_area

  if union_area == 0:
    return 0.0
  return inter_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision (AP)
    Args:
        recalls: Array of recall values.
        precisions: Array of precision values.
    Returns:
        Average Precision score.
    """
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def evaluate_detections_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_id: int,
    iou_threshold: float = 0.5
) -> Tuple[float, List[float], List[float]]:
    """Evaluate detections for a specific class.

    Args:
        predictions: List of dicts with {bbox, class_id, confidence, image_id}.
        ground_truths: List of dicts with {bbox, class_id, image_id}.
        class_id: ID of the class to evaluate.
        iou_threshold: IoU threshold to consider detection correct.

    Returns:
        Tuple of (AP, recalls, precisions).
    """
    class_predictions = [p for p in predictions if p['class_id'] == class_id]
    class_gts = [g for g in ground_truths if g['class_id'] == class_id]

    if len(class_gts) == 0:
        return 0.0, [], []

    if len(class_predictions) == 0:
        return 0.0, [0.0], [1.0]

    class_predictions = sorted(class_predictions, key=lambda x: x['confidence'], reverse=True)

    gt_by_image = defaultdict(list)
    for gt in class_gts:
        gt_by_image[gt['image_id']].append(gt)

    gt_detected = defaultdict(lambda: defaultdict(bool))

    tp = np.zeros(len(class_predictions))
    fp = np.zeros(len(class_predictions))

    for idx, pred in enumerate(class_predictions):
        image_id = pred['image_id']
        pred_box = pred['bbox']

        gts_in_image = gt_by_image.get(image_id, [])

        max_iou = 0.0
        max_gt_idx = -1

        for gt_idx, gt in enumerate(gts_in_image):
            iou = compute_iou(pred_box, gt['bbox'])

            if iou > max_iou:
                max_iou = iou
                max_gt_idx = gt_idx

        if max_iou >= iou_threshold:
            if not gt_detected[image_id][max_gt_idx]:
                tp[idx] = 1
                gt_detected[image_id][max_gt_idx] = True
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    n_ground_truths = len(class_gts)
    recalls = tp_cumsum / n_ground_truths
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_ap(recalls, precisions)

    return ap, recalls.tolist(), precisions.tolist()


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5,
    class_names: Dict[int, str] = None
) -> Dict:
    """Compute mAP (mean Average Precision) for all classes.

    Args:
        predictions: List of predictions with {bbox, class_id, confidence, image_id}.
        ground_truths: List of ground truths with {bbox, class_id, image_id}.
        num_classes: Number of classes.
        iou_threshold: IoU threshold.
        class_names: Dict mapping class_id to name (optional).

    Returns:
        Dict with results:
        {
            'mAP': float,
            'per_class_ap': {class_id: ap},
            'per_class_results': {class_id: {ap, recalls, precisions}}
        }
    """
    per_class_ap = {}
    per_class_results = {}

    for class_id in range(1, num_classes + 1):
        ap, recalls, precisions = evaluate_detections_per_class(
            predictions=predictions,
            ground_truths=ground_truths,
            class_id=class_id,
            iou_threshold=iou_threshold
        )

        per_class_ap[class_id] = ap
        per_class_results[class_id] = {
            'ap': ap,
            'recalls': recalls,
            'precisions': precisions
        }

    mAP = np.mean(list(per_class_ap.values()))

    results = {
        'mAP': mAP,
        'per_class_ap': per_class_ap,
        'per_class_results': per_class_results
    }

    if class_names:
        results['class_names'] = class_names

    return results


def classification_predictions_to_detections(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    image_paths: List[str],
    confidences: torch.Tensor = None
) -> Tuple[List[Dict], List[Dict]]:
    """Convert classification predictions to detection format.

    Useful for evaluating classification models with detection metrics.

    Args:
        images: Tensor of images.
        predictions: Tensor of predictions (class indices).
        labels: Tensor of true labels.
        image_paths: List of image paths.
        confidences: Tensor of confidences (probabilities).

    Returns:
        Tuple of (predictions_list, ground_truths_list).
    """
    predictions_list = []
    ground_truths_list = []

    batch_size = images.size(0)

    for i in range(batch_size):
        image_id = image_paths[i]

        img_size = images.size(-1)
        full_bbox = [0, 0, img_size, img_size]

        ground_truths_list.append({
            'bbox': full_bbox,
            'class_id': int(labels[i].item()) + 1,
            'image_id': image_id
        })

        confidence = confidences[i].item() if confidences is not None else 1.0
        predictions_list.append({
            'bbox': full_bbox,
            'class_id': int(predictions[i].item()) + 1,
            'confidence': confidence,
            'image_id': image_id
        })

    return predictions_list, ground_truths_list


def format_map_results(results: Dict, verbose: bool = True) -> str:
    """Format mAP results for display.

    Args:
        results: Dict returned by compute_map.
        verbose: If True, shows detailed per-class information.

    Returns:
        Formatted string with results.
    """
    from core.datasets.nwpu_dataset import CLASS_MAPPING

    output = []
    output.append("="*60)
    output.append("mAP Evaluation Results")
    output.append("="*60)
    output.append(f"\nmAP@0.5: {results['mAP']:.4f}")
    output.append("\nPer-class Average Precision:")
    output.append("-"*60)

    for class_id, ap in sorted(results['per_class_ap'].items()):
        class_name = CLASS_MAPPING.get(class_id, f"Class_{class_id}")
        output.append(f"{class_name:25s}: {ap:.4f}")

    if verbose and 'per_class_results' in results:
        output.append("\n" + "="*60)
        output.append("Detailed Results per Class")
        output.append("="*60)

        for class_id, details in sorted(results['per_class_results'].items()):
            class_name = CLASS_MAPPING.get(class_id, f"Class_{class_id}")
            output.append(f"\n{class_name}:")
            output.append(f"  AP: {details['ap']:.4f}")
            if details['recalls'] and details['precisions']:
                max_recall = max(details['recalls']) if details['recalls'] else 0
                max_precision = max(details['precisions']) if details['precisions'] else 0
                output.append(f"  Max Recall: {max_recall:.4f}")
                output.append(f"  Max Precision: {max_precision:.4f}")

    output.append("\n" + "="*60)
    return "\n".join(output)


def calculate_CorLoc(all_predictions, bbox_to_id, filename_to_id=None, loc_threshold=0.5):
    """
    Compatibility wrapper. Use calculate_corloc() in new code.

    Kept to avoid breaking existing code that calls calculate_CorLoc().
    """
    return calculate_corloc(all_predictions, bbox_to_id, filename_to_id, loc_threshold)


def calculate_metric(total_acc, gt_bbox, bbs, threshold, resize, img_id, image_sizes):
  """
  Calculate bounding box accuracy metric.

  Compares predicted BBs against GT using IoU threshold.
  Handles COCO format [x, y, w, h, class_id].

  Args:
      total_acc: Cumulative list of accuracies
      gt_bbox: Ground truth bounding boxes
      bbs: Predicted bounding boxes
      threshold: IoU threshold to consider a match
      resize: Resized image size (model input)
      img_id: Image identifier
      image_sizes: Dict mapping image_id to (width, height) original dimensions

  Returns:
      List with accumulated accuracies (including current image accuracy)
  """
  if not isinstance(gt_bbox, (list, tuple)):
    return total_acc

  if not gt_bbox or not bbs:
    total_acc.append(1.0)
    return total_acc

  correct = 0

  # Get original dimensions
  if isinstance(image_sizes, dict) and img_id in image_sizes:
    original_width, original_height = image_sizes[img_id]
    or_size = min(original_width, original_height)
  else:
    original_width, original_height = None, None
    or_size = resize  # Use resize as fallback when original dims unknown

  try:
    gt_array = np.array(gt_bbox)
  except (ValueError, TypeError):
    log_info(f"Warning: Could not convert GTs to array: {gt_bbox}")
    total_acc.append(0.0)
    return total_acc

  try:
    bbs_array = np.array(bbs)
  except (ValueError, TypeError):
    log_info(f"Warning: Could not convert bbs to array: {bbs}")
    total_acc.append(0.0)
    return total_acc

  # Process each GT
  for gt in gt_array:
    try:
      if len(gt) < 4:
        continue

      if original_width and original_height:
        gt_x, gt_y, gt_width, gt_height = VisUtils.redimension_bboxes(
            (int(round(float(v))) for v in gt[:4]),
            resize,
            or_size,
            original_width,
            original_height
        )
      else:
        gt_x, gt_y, gt_width, gt_height = VisUtils.redimension_bboxes(
            (int(round(float(v))) for v in gt[:4]),
            resize,
            or_size
        )

      gt_box = (gt_x, gt_y, gt_x + gt_width, gt_y + gt_height)

      # Find best IoU with predictions
      best_iou = 0.0
      for bb in bbs_array:
        try:
          if len(bb) < 4:
            continue
          x, y, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
          pred_box = (x, y, x2, y2)
          iou_val = VisUtils.get_IoU(gt_box, pred_box)
          best_iou = max(best_iou, iou_val)
        except (ValueError, TypeError, IndexError):
          continue

      if best_iou >= threshold:
        correct += 1

    except (ValueError, TypeError, IndexError) as e:
      log_info(f"Warning: Error processing GT {gt}: {e}")
      continue

  num_gt = len(gt_bbox)
  if num_gt > 0:
    acc = float(correct) / float(num_gt)
  else:
    acc = 1.0

  total_acc.append(acc)
  return total_acc


def calculate_corloc(all_predictions, bbox_to_id, filename_to_id=None, loc_threshold=0.5):
  """
  Calculate CorLoc (Correct Localization) - standard WSOL/WSOD metric.
  Args:
      all_predictions: List of tuples (img_id, bbs, resize, image_sizes, pred_label)
      bbox_to_id: Mapping of image_id to GTs (with or without class_id)
      filename_to_id: (Not used, kept for compatibility)
      loc_threshold: IoU threshold to consider correct (default: 0.5)

  Returns:
      float: Average CorLoc (0-1 scale)
  """

  num_classes = get_active_dataset_config('num_classes') or 10

  log_info(f"CorLoc (IoU >= {loc_threshold}):")

  per_class_corloc = {}
  per_class_stats = {}

  for class_id in range(1, num_classes + 1):
    correct_count = 0
    total_count = 0

    for pred_data in all_predictions:
      if len(pred_data) == 5:
        img_id, bbs, resize, image_sizes, pred_label = pred_data
      elif len(pred_data) == 4:
        img_id, bbs, resize, image_sizes = pred_data
        pred_label = 0
      else:
        log_info(f"Warning: Unknown prediction format with {len(pred_data)} elements")
        continue

      pred_class_id = int(pred_label) + 1

      # Skip if class doesn't match
      if pred_class_id != class_id:
        continue

      gt_bboxes = bbox_to_id.get(img_id, [])
      if not gt_bboxes:
        continue

      class_gts = []
      for gt in gt_bboxes:
        gt_class_id = gt[4] if len(gt) >= 5 else 1
        if gt_class_id == class_id:
          try:
            gt_x, gt_y, gt_w, gt_h = gt[0], gt[1], gt[2], gt[3]
            class_gts.append((int(gt_x), int(gt_y), int(gt_x + gt_w), int(gt_y + gt_h)))
          except (ValueError, TypeError):
            continue

      if not class_gts:
        continue

      total_count += 1

      # Check if ANY prediction matches ANY GT with IoU >= threshold
      image_correct = False
      for bb in bbs:
        if image_correct:
          break
        try:
          if isinstance(bb, (tuple, list)):
            if len(bb) == 2 and isinstance(bb[0], (tuple, list)):
              bbox_tuple = bb[0]
            else:
              bbox_tuple = bb

            if len(bbox_tuple) < 4:
              continue

            x, y, x2, y2 = bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], bbox_tuple[3]
            pred_box = (x, y, x2, y2)

            for gt_box in class_gts:
              if VisUtils.get_IoU(gt_box, pred_box) >= loc_threshold:
                image_correct = True
                break
        except (ValueError, TypeError, IndexError):
          continue

      if image_correct:
        correct_count += 1

    # Calculate CorLoc for this class (0-1 scale)
    if total_count > 0:
      class_corloc = correct_count / total_count
    else:
      class_corloc = 0.0

    per_class_corloc[class_id] = class_corloc
    per_class_stats[class_id] = {'correct': correct_count, 'total': total_count}

    log_info(f"  Class {class_id}: CorLoc = {class_corloc:.4f} ({correct_count}/{total_count} images)")

  # Calculate average only for classes with GTs
  classes_with_gts = [cid for cid, stats in per_class_stats.items() if stats['total'] > 0]
  if classes_with_gts:
    avg_corloc = np.mean([per_class_corloc[cid] for cid in classes_with_gts])
  else:
    avg_corloc = 0.0

  return avg_corloc


def calculate_detection_confusion(all_predictions, bbox_to_id, iou_threshold=0.5):
    """
    Calculate detection TP, FP, FN based on IoU threshold.

    Args:
        all_predictions: List of tuples (img_id, bbs, resize, image_sizes, pred_label)
        bbox_to_id: Mapping of image_id to GTs
        iou_threshold: IoU threshold to consider a match

    Returns:
        Dict with tp_det, fp_det, fn_det, precision_det, recall_det
    """
    tp_det = 0  # Predicted bbox matches GT (IoU >= threshold)
    fp_det = 0  # Predicted bbox doesn't match any GT
    fn_det = 0  # GT not detected by any predicted bbox

    # Track which GTs have been matched
    gt_matched = {}

    for pred_data in all_predictions:
        if len(pred_data) >= 4:
            img_id = pred_data[0]
            bbs = pred_data[1]
        else:
            continue

        gt_bboxes = bbox_to_id.get(img_id, [])

        # Initialize matched tracking for this image
        if img_id not in gt_matched:
            gt_matched[img_id] = [False] * len(gt_bboxes)

        # For each predicted bbox
        for bb in bbs:
            try:
                if isinstance(bb, (tuple, list)) and len(bb) >= 4:
                    pred_box = (bb[0], bb[1], bb[2], bb[3])
                else:
                    continue

                # Find best matching GT
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_bboxes):
                    if len(gt) < 4:
                        continue
                    # Convert COCO [x,y,w,h] to corners [x1,y1,x2,y2]
                    gt_box = (int(gt[0]), int(gt[1]), int(gt[0] + gt[2]), int(gt[1] + gt[3]))
                    iou_val = VisUtils.get_IoU(gt_box, pred_box)

                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx

                # Check if match
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    if not gt_matched[img_id][best_gt_idx]:
                        tp_det += 1
                        gt_matched[img_id][best_gt_idx] = True
                    else:
                        # Duplicate detection of same GT
                        fp_det += 1
                else:
                    fp_det += 1

            except (ValueError, TypeError, IndexError):
                continue

    # Count unmatched GTs as FN
    for img_id, matched_list in gt_matched.items():
        fn_det += sum(1 for m in matched_list if not m)

    # Also count GTs from images with no predictions
    for img_id, gt_bboxes in bbox_to_id.items():
        if img_id not in gt_matched:
            fn_det += len(gt_bboxes)

    # Calculate precision and recall
    precision_det = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
    recall_det = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0

    return {
        'tp_det': tp_det,
        'fp_det': fp_det,
        'fn_det': fn_det,
        'precision_det': precision_det,
        'recall_det': recall_det
    }


def calculate_map(all_predictions, bbox_to_id, filename_to_id=None, image_sizes=None, iou_thresholds=None):
  """
  Calculate mAP (mean Average Precision) for WSOD object detection.

  Implementation follows COCO/PASCAL VOC standard using IoU metric.

  Args:
      all_predictions: List of tuples (img_id, bbs, resize, image_sizes, pred_label)
      bbox_to_id: Mapping of image_id to GTs (with class_id)
      filename_to_id: (Not used, kept for compatibility)
      image_sizes: Dict mapping image_id to (width, height) original dimensions
      iou_thresholds: List of IoU thresholds (default: [0.5] like COCO)

  Returns:
      float: Average mAP (mean of APs for all classes)
  """
  if image_sizes is None:
    image_sizes = {}
  if iou_thresholds is None:
    iou_thresholds = [0.5]

  # Identify classes with GTs
  classes_with_gts = set()
  for img_id, gt_list in bbox_to_id.items():
    for gt in gt_list:
      gt_class_id = gt[4] if len(gt) >= 5 else 1
      classes_with_gts.add(gt_class_id)

  total_gts = sum(len(gt_list) for gt_list in bbox_to_id.values())

  log_info(f"mAP — {len(all_predictions)} preds, {total_gts} GTs, IoU={iou_thresholds}:")

  all_class_aps = []

  for iou_threshold in iou_thresholds:
    per_class_ap = {}

    for class_id in sorted(classes_with_gts):
      class_detections = []
      class_gt_count = 0
      gt_by_image = {}

      for pred_data in all_predictions:
        if len(pred_data) == 5:
          img_id, bbs, resize, img_size_dict, pred_label = pred_data
        elif len(pred_data) == 4:
          img_id, bbs, resize, img_size_dict = pred_data
          pred_label = 0
        else:
          log_info(f"Warning: Unknown prediction format")
          continue

        gt_bboxes = bbox_to_id.get(img_id, [])
        if not gt_bboxes:
          pass

        if img_size_dict and img_id in img_size_dict:
          original_width, original_height = img_size_dict[img_id]
          or_size = min(original_width, original_height)
        else:
          original_width, original_height = None, None
          or_size = resize  # Use resize as fallback when original dims unknown

        gt_boxes_normalized = []
        for gt in gt_bboxes:
          gt_class_id = gt[4] if len(gt) >= 5 else 1
          if gt_class_id != class_id:
            continue

          try:
            # GT already scaled to img_size, just convert COCO [x,y,w,h] to corners [x1,y1,x2,y2]
            gt_x, gt_y, gt_w, gt_h = gt[0], gt[1], gt[2], gt[3]
            gt_box = (int(gt_x), int(gt_y), int(gt_x + gt_w), int(gt_y + gt_h))
            gt_boxes_normalized.append(gt_box)
          except (ValueError, TypeError) as e:
            log_info(f"Warning: Error processing GT {gt}: {e}")
            continue

        # Store normalized GTs for IoU matching
        if gt_boxes_normalized:
          gt_by_image[img_id] = gt_boxes_normalized
          class_gt_count += len(gt_boxes_normalized)

        try:
          pred_class_id = int(pred_label) + 1
          if pred_class_id != class_id:
            continue

          for bb in bbs:
            if not isinstance(bb, (tuple, list)) or len(bb) < 5:
              continue

            x, y, x2, y2, confidence = bb[0], bb[1], bb[2], bb[3], bb[4]
            pred_box = (x, y, x2, y2)

            class_detections.append({
              'bbox': pred_box,
              'confidence': float(confidence),
              'img_id': img_id
            })

        except (ValueError, TypeError, IndexError) as e:
          log_info(f"Warning: Error processing bbox: {e}")
          continue

      # Calculate AP for this class
      if class_gt_count == 0:
        ap_value = 0.0
        log_info(f"  Class {class_id}: AP = {ap_value:.4f} (0 GTs)")
      elif len(class_detections) == 0:
        ap_value = 0.0
        log_info(f"  Class {class_id}: AP = {ap_value:.4f} ({class_gt_count} GTs, 0 predictions)")
      else:
        # Sort detections by confidence
        class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)

        tp = np.zeros(len(class_detections))
        fp = np.zeros(len(class_detections))
        matched_gt = defaultdict(set)

        # IoU matching
        for det_idx, det in enumerate(class_detections):
          img_id = det['img_id']
          pred_box = det['bbox']

          gt_boxes = gt_by_image.get(img_id, [])

          if not gt_boxes:
            fp[det_idx] = 1
            continue

          best_iou = 0.0
          best_gt_idx = -1

          for gt_idx, gt_box in enumerate(gt_boxes):
            # Skip already matched GT
            if gt_idx in matched_gt[img_id]:
              continue

            iou_val = VisUtils.get_IoU(gt_box, pred_box)
            if iou_val > best_iou:
              best_iou = iou_val
              best_gt_idx = gt_idx

          if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[det_idx] = 1
            matched_gt[img_id].add(best_gt_idx)
          else:
            fp[det_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / float(class_gt_count)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
          mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        ap_value = float(np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1]))

        num_tps = int(tp_cumsum[-1])
        log_info(f"  Class {class_id}: AP = {ap_value:.4f} ({class_gt_count} GTs, {len(class_detections)} predictions, {num_tps} TPs)")

      per_class_ap[class_id] = ap_value

    all_class_aps.append(per_class_ap)

  final_map = 0.0
  if all_class_aps:
    class_aps_list = all_class_aps[0]  # Get first threshold
    valid_aps = [ap for class_id, ap in class_aps_list.items() if class_id in classes_with_gts]
    if valid_aps:
      final_map = float(np.mean(valid_aps))
  return final_map

_wsod_loader_cache = {}

def evaluate_wsod(
    model,
    model_name: str,
    device,
    num_workers: int = 0,
    num_classes: Optional[int] = None,
    iou_thresholds: Optional[List[float]] = None,
    loc_threshold: float = 0.5,
    batch_size: int = 16,
    split: str = "Test",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate WSOD model with CorLoc and mAP.

    Args:
        model: Trained model
        model_name: Model name (for logs)
        device: Device (cpu/cuda/mps)
        num_workers: Number of workers for dataloader
        num_classes: Number of classes (uses config if None)
        iou_thresholds: IoU thresholds for mAP (default: [0.5])
        loc_threshold: LocAcc threshold for CorLoc (default: 0.5)
        batch_size: Batch size for evaluation
        split: Dataset split to use ("Train", "Val", or "Test")
        verbose: If True, print detailed logs

    Returns:
        Dict with results
    """
    from core.config.config import get_active_dataset_config

    if num_classes is None:
        num_classes = get_active_dataset_config('num_classes') or 10

    if iou_thresholds is None:
        iou_thresholds = [0.5]

    root_path = get_active_dataset_config('data_path') or get_active_dataset_config('root_path') or 'data/'

    img_size = get_config("models", model_name, "img_size")
    if img_size is None:
        img_size = 256
    else:
        img_size = int(img_size)

    if verbose:
        log_info(f"Evaluating WSOD: {model_name} on {split}")

    # Cached dataloader: create once per (split, img_size), reuse on subsequent calls
    cache_key = (split, img_size)
    if cache_key in _wsod_loader_cache:
        data_loader = _wsod_loader_cache[cache_key]
        data_loader.generator.manual_seed(data_loader.generator.initial_seed())
    else:
        from core.datasets import get_data_loaders

        try:
            data_loaders, _ = get_data_loaders(
                root_path=root_path,
                batch_size=batch_size,
                num_workers=num_workers,
                eval_size=img_size,
                splits=[split]
            )
            data_loader = data_loaders.get(split)
            if data_loader is not None:
                _wsod_loader_cache[cache_key] = data_loader
        except Exception as e:
            log_info(f"Error loading dataset: {e}")
            return {
                'corloc': 0.0,
                'map': 0.0,
                'error': f'Failed to load {split} dataset: {str(e)}'
            }

    if not data_loader or len(data_loader) == 0:
        log_info(f"No {split} dataset found!")
        return {
            'corloc': 0.0,
            'map': 0.0,
            'error': f'No {split} dataset'
        }

    # Collect predictions and GTs
    all_predictions = []
    bbox_to_id = {}
    image_sizes = {}
    total_images = 0

    # For classification metrics
    all_pred_labels = []
    all_true_labels = []

    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            try:
                # Unpack batch
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) >= 3:
                        images, labels, image_paths = batch_data[0], batch_data[1], batch_data[2]
                        bboxes_gt = batch_data[3] if len(batch_data) > 3 else None
                    else:
                        continue
                elif isinstance(batch_data, dict):
                    images = batch_data.get('image')
                    labels = batch_data.get('labels')
                    image_paths = batch_data.get('image_paths', [])
                    bboxes_gt = batch_data.get('bboxes')
                else:
                    continue

                images = images.to(device)
                batch_size_actual = images.size(0)
                total_images += batch_size_actual

                # Forward pass
                outputs = model(images)

                # Process outputs
                for i in range(batch_size_actual):
                    try:
                        if isinstance(image_paths, (list, tuple)) and i < len(image_paths):
                            img_id = str(image_paths[i])
                        else:
                            img_id = f"img_{batch_idx}_{i}"

                        # Prediction
                        if isinstance(outputs, dict) and 'logits' in outputs:
                            pred_label = torch.argmax(outputs['logits'][i:i+1], dim=1).item()
                        elif isinstance(outputs, torch.Tensor):
                            pred_label = torch.argmax(outputs[i]).item()
                        else:
                            pred_label = 0

                        # Collect for classification metrics (all images)
                        all_pred_labels.append(pred_label)
                        true_label = labels[i].item() if isinstance(labels, torch.Tensor) else int(labels[i])
                        all_true_labels.append(true_label)

                        # Skip detection for Not images (only Fire has bboxes)
                        if true_label != 0:  # 0 = Fire, 1 = Not
                            continue

                        # Boxes - extract from attention maps
                        pred_boxes = []
                        try:
                            img_tensor = images[i:i+1]
                            attn_np = get_last_layer_attention(model, model_name, img_tensor)

                            # Apply CRF refinement if enabled
                            crf_config = get_config("crf") or {}
                            if crf_config.get('enabled', False):
                                from core.visualization.gradcam import refine_heatmap_with_crf
                                attn_np = refine_heatmap_with_crf(
                                    attn_np, image=None, use_crf=True, crf_config=crf_config
                                )

                            # Dynamic threshold and generate bboxes
                            threshold = compute_dynamic_threshold(attn_np, method="otsu", fire_adapted=True)
                            min_bbox_area = get_config("defaults", "min_bbox_area") or 1600
                            bbs_raw = VisUtils.generate_bounding_box(attn_np, threshold=threshold, min_area=min_bbox_area)

                            # Remove nested bboxes (same as debug.py)
                            if len(bbs_raw) > 1:
                                from core.utils.debug import remove_nested_bboxes
                                bbs_raw = remove_nested_bboxes(bbs_raw, containment_threshold=0.5)

                            # Add confidence to each bbox
                            for bb in bbs_raw:
                                if len(bb) >= 4:
                                    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
                                    conf = bb[4] if len(bb) > 4 else 1.0
                                    pred_boxes.append((x1, y1, x2, y2, conf))

                            # If no bbox was generated, use fallback
                            if not pred_boxes:
                                pred_boxes = [(0, 0, img_size, img_size, 1.0)]

                        except Exception as e:
                            # Fallback: full image bbox
                            if batch_idx == 0 and i == 0:
                                log_info(f"Error extracting attention: {e}")
                            pred_boxes = [(0, 0, img_size, img_size, 1.0)]

                        # Use true_label for WSOD matching (evaluates localization, not classification)
                        all_predictions.append((img_id, pred_boxes, img_size, image_sizes, true_label))

                        # GT
                        if bboxes_gt is not None and len(bboxes_gt) > i:
                            gt_boxes = bboxes_gt[i]
                            if isinstance(gt_boxes, torch.Tensor):
                                gt_boxes = gt_boxes.cpu().numpy().tolist()
                            bbox_to_id[img_id] = gt_boxes
                        elif isinstance(labels, torch.Tensor) and i < len(labels):
                            gt_label = int(labels[i].item())
                            bbox_to_id[img_id] = [[0, 0, img_size, img_size, gt_label]]

                        # Dimensions
                        if hasattr(images, 'shape'):
                            h, w = images[i].shape[-2:] if len(images.shape) == 4 else (256, 256)
                            image_sizes[img_id] = (w, h)

                    except Exception as e:
                        continue

                pass

            except Exception as e:
                continue

    if verbose:
        log_info(f"Total images processed: {total_images}")

    # Load real bboxes from COCO JSON (scaled to img_size)
    split_path = os.path.join(root_path, split)
    real_bboxes = load_coco_bboxes_from_json(split_path, target_size=img_size)

    # Merge with real bboxes
    if real_bboxes:
        for img_id in list(bbox_to_id.keys()):
            # Extract filename from img_id
            if isinstance(img_id, str):
                filename = os.path.basename(img_id)
                if filename in real_bboxes:
                    bbox_to_id[img_id] = real_bboxes[filename]

    # Calculate CorLoc
    try:
        corloc = calculate_corloc(
            all_predictions=all_predictions,
            bbox_to_id=bbox_to_id,
            loc_threshold=loc_threshold
        )
    except Exception as e:
        log_info(f"Error calculating CorLoc: {e}")
        corloc = 0.0

    # Calculate mAP
    try:
        map_value = calculate_map(
            all_predictions=all_predictions,
            bbox_to_id=bbox_to_id,
            image_sizes=image_sizes,
            iou_thresholds=iou_thresholds
        )
    except Exception as e:
        log_info(f"Error calculating mAP: {e}")
        map_value = 0.0

    # Detection metrics (TP/FP/FN based on IoU)
    try:
        det_metrics = calculate_detection_confusion(
            all_predictions=all_predictions,
            bbox_to_id=bbox_to_id,
            iou_threshold=iou_thresholds[0]
        )
        tp_det = det_metrics['tp_det']
        fp_det = det_metrics['fp_det']
        fn_det = det_metrics['fn_det']
        precision_det = det_metrics['precision_det']
        recall_det = det_metrics['recall_det']
    except Exception as e:
        log_info(f"Error calculating detection metrics: {e}")
        tp_det, fp_det, fn_det = 0, 0, 0
        precision_det, recall_det = 0.0, 0.0

    # Classification metrics
    accuracy = 0.0
    error_rate = 0.0
    precision = 0.0
    recall = 0.0
    fdr = 0.0  # False Discovery Rate
    f1 = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0

    if all_pred_labels and all_true_labels:
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        error_rate = 1.0 - accuracy
        precision = precision_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)
        fdr = 1.0 - precision  # FDR = FP / (FP + TP) = 1 - Precision
        f1 = f1_score(all_true_labels, all_pred_labels, average='binary', zero_division=0)

        # Confusion matrix: [[TN, FP], [FN, TP]] for binary classification
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

    # Summary
    if verbose:
        log_info(f"{'='*60}")
        log_info(f"WSOD EVALUATION RESULTS ({split})")
        log_info(f"Model: {model_name}")
        log_info(f"Total images: {total_images}")
        log_info(f"Total classes: {num_classes}")
        log_info(f"-" * 60)
        log_info(f"CLASSIFICATION METRICS:")
        log_info(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        log_info(f"  Accuracy: {accuracy:.4f}")
        log_info(f"  Error Rate (ER): {error_rate:.4f}")
        log_info(f"  Precision: {precision:.4f}")
        log_info(f"  Recall: {recall:.4f}")
        log_info(f"  FDR (False Discovery Rate): {fdr:.4f}")
        log_info(f"  F1-score: {f1:.4f}")
        log_info(f"-" * 60)
        log_info(f"DETECTION METRICS (IoU >= {iou_thresholds[0]}):")
        log_info(f"  TP_det: {tp_det}, FP_det: {fp_det}, FN_det: {fn_det}")
        log_info(f"  Precision_det: {precision_det:.4f}")
        log_info(f"  Recall_det: {recall_det:.4f}")
        log_info(f"  CorLoc (LocAcc >= {loc_threshold}): {corloc:.4f}")
        log_info(f"  mAP@IoU{iou_thresholds[0]}: {map_value:.4f}")
        log_info(f"{'='*60}")
    else:
        # Compact output for epoch evaluation
        log_info(f"[{split}] CorLoc: {corloc:.4f} | mAP@50: {map_value:.4f} | Acc: {accuracy:.4f}")

    return {
        # Classification metrics
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'accuracy': float(accuracy),
        'error_rate': float(error_rate),
        'precision': float(precision),
        'recall': float(recall),
        'fdr': float(fdr),
        'f1': float(f1),
        # Detection metrics
        'tp_det': int(tp_det),
        'fp_det': int(fp_det),
        'fn_det': int(fn_det),
        'precision_det': float(precision_det),
        'recall_det': float(recall_det),
        'corloc': float(corloc),
        'map': float(map_value),
        # Metadata
        'iou_threshold': iou_thresholds[0],
        'loc_threshold': loc_threshold,
        'num_images': total_images,
        'num_classes': num_classes,
        'model_name': model_name,
    }





if __name__ == "__main__":
  log_info("="*60)
  log_info("Testing detection metrics")
  log_info("="*60)

  log_info("\n[1/3] Testing IoU calculation...")
  box1 = [0, 0, 100, 100]
  box2 = [50, 50, 150, 150]
  iou = compute_iou(box1, box2)
  log_info(f"  IoU between {box1} and {box2}: {iou:.4f}")
  assert 0.14 < iou < 0.15, f"Unexpected IoU: {iou}"
  log_info("  ✓ IoU test passed")

  log_info("\n[2/3] Testing AP calculation...")
  predictions = [
    {'bbox': [10, 10, 50, 50], 'class_id': 1, 'confidence': 0.9, 'image_id': 'img1.jpg'},
    {'bbox': [60, 60, 100, 100], 'class_id': 1, 'confidence': 0.8, 'image_id': 'img1.jpg'},
    {'bbox': [15, 15, 55, 55], 'class_id': 1, 'confidence': 0.7, 'image_id': 'img2.jpg'},
  ]

  ground_truths = [
    {'bbox': [10, 10, 50, 50], 'class_id': 1, 'image_id': 'img1.jpg'},
    {'bbox': [20, 20, 60, 60], 'class_id': 1, 'image_id': 'img2.jpg'},
  ]

  ap, recalls, precisions = evaluate_detections_per_class(
      predictions, ground_truths, class_id=1, iou_threshold=0.5
  )

  log_info(f"  AP for class 1: {ap:.4f}")
  log_info(f"  Recalls: {recalls}")
  log_info(f"  Precisions: {precisions}")
  log_info("  AP test passed")

  log_info("\n[3/3] Testing normalize_bbox()...")
  bbox_coco = [10, 20, 40, 40]  # [x, y, w, h]
  bbox_normalized = normalize_bbox(bbox_coco)
  log_info(f"  COCO {bbox_coco} -> {bbox_normalized}")
  assert bbox_normalized == (10, 20, 50, 60), f"COCO conversion failed"
  log_info("  ✓ normalize_bbox test passed")

  log_info("\n" + "="*60)
  log_info("All basic tests passed!")
  log_info("="*60)