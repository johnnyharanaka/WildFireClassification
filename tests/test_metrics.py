"""Tests for core/metrics.py — pure-numeric WSOD/classification metrics."""
import numpy as np
import pytest
import torch

from core.metrics import (
    classification_predictions_to_detections,
    calculate_corloc,
    calculate_detection_confusion,
    calculate_map,
    compute_ap,
    compute_bbox_area,
    compute_iou,
    compute_map,
    evaluate_detections_per_class,
    get_bbox_area,
    normalize_bbox,
    validate_ground_truths,
    validate_predictions,
)


# --------------------------------------------------------------------- IoU --
class TestComputeIoU:
    def test_identical_boxes_iou_is_one(self):
        box = [0, 0, 100, 100]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        assert compute_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0

    def test_known_partial_overlap(self):
        # 50x50 overlap on 100x100+100x100 union → 2500 / (10000+10000-2500) = 1/7
        iou = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert iou == pytest.approx(2500 / 17500, rel=1e-6)

    def test_zero_area_box_returns_zero(self):
        assert compute_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0

    def test_box_inside_other(self):
        # inner is fully contained → IoU = inner_area / outer_area = 100 / 10000
        iou = compute_iou([0, 0, 100, 100], [10, 10, 20, 20])
        assert iou == pytest.approx(100 / 10000, rel=1e-6)


# ---------------------------------------------------------------- bbox util --
class TestBboxHelpers:
    def test_compute_bbox_area(self):
        assert compute_bbox_area([0, 0, 10, 5]) == 50

    def test_get_bbox_area_alias(self):
        assert get_bbox_area((1, 2, 11, 12)) == 100

    def test_normalize_bbox_coco_to_corners(self):
        assert normalize_bbox([10, 20, 40, 40]) == (10.0, 20.0, 50.0, 60.0)

    def test_normalize_bbox_clamps_negatives(self):
        x1, y1, x2, y2 = normalize_bbox([-5, -5, 10, 10])
        assert x1 == 0.0 and y1 == 0.0 and x2 == 5.0 and y2 == 5.0

    def test_normalize_bbox_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            normalize_bbox(["a", "b", "c", "d"])

    def test_normalize_bbox_rejects_zero_area(self):
        with pytest.raises(ValueError):
            normalize_bbox([10, 10, 0, 0])  # x+w=10, y+h=10 → x1==x2, invalid

    def test_normalize_bbox_rejects_too_short(self):
        with pytest.raises(ValueError):
            normalize_bbox([1, 2])


# ----------------------------------------------------------------------- AP --
class TestComputeAP:
    def test_perfect_precision_recall_gives_one(self):
        recalls = np.array([0.5, 1.0])
        precisions = np.array([1.0, 1.0])
        assert compute_ap(recalls, precisions) == pytest.approx(1.0)

    def test_zero_precision_gives_zero(self):
        ap = compute_ap(np.array([0.0]), np.array([0.0]))
        assert ap == 0.0


# ------------------------------------------------------- per-class detection --
class TestEvaluateDetectionsPerClass:
    def test_no_gts_returns_zero(self):
        ap, _, _ = evaluate_detections_per_class([], [], class_id=1)
        assert ap == 0.0

    def test_no_predictions_with_gts(self):
        gts = [{"bbox": [0, 0, 10, 10], "class_id": 1, "image_id": "a"}]
        ap, recalls, precisions = evaluate_detections_per_class([], gts, class_id=1)
        assert ap == 0.0
        assert recalls == [0.0] and precisions == [1.0]

    def test_single_perfect_match_gives_ap_one(self):
        preds = [{"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 0.9, "image_id": "a"}]
        gts = [{"bbox": [0, 0, 10, 10], "class_id": 1, "image_id": "a"}]
        ap, _, _ = evaluate_detections_per_class(preds, gts, class_id=1, iou_threshold=0.5)
        assert ap == pytest.approx(1.0)

    def test_duplicate_detection_counts_as_fp(self):
        preds = [
            {"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 0.9, "image_id": "a"},
            {"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 0.8, "image_id": "a"},
        ]
        gts = [{"bbox": [0, 0, 10, 10], "class_id": 1, "image_id": "a"}]
        ap, recalls, precisions = evaluate_detections_per_class(preds, gts, class_id=1)
        # First TP, second FP → recall=[1,1], precision=[1, 0.5]
        assert recalls[-1] == pytest.approx(1.0)
        assert precisions[-1] == pytest.approx(0.5)


# --------------------------------------------------------------- compute_map --
class TestComputeMap:
    def test_perfect_predictions_yield_map_one(self):
        preds = [
            {"bbox": [0, 0, 10, 10], "class_id": 1, "confidence": 0.99, "image_id": "img1"},
            {"bbox": [20, 20, 40, 40], "class_id": 2, "confidence": 0.99, "image_id": "img2"},
        ]
        gts = [
            {"bbox": [0, 0, 10, 10], "class_id": 1, "image_id": "img1"},
            {"bbox": [20, 20, 40, 40], "class_id": 2, "image_id": "img2"},
        ]
        result = compute_map(preds, gts, num_classes=2, iou_threshold=0.5)
        assert result["mAP"] == pytest.approx(1.0)
        assert result["per_class_ap"][1] == pytest.approx(1.0)
        assert result["per_class_ap"][2] == pytest.approx(1.0)

    def test_empty_predictions_with_class_names(self):
        result = compute_map([], [], num_classes=2, class_names={1: "a", 2: "b"})
        assert "class_names" in result


# ----------------------------------------------------------------- validate --
class TestValidators:
    def test_validate_predictions_empty_is_ok(self):
        assert validate_predictions([]) is True

    def test_validate_predictions_missing_keys(self):
        assert validate_predictions([{"bbox": [0, 0, 1, 1]}]) is False

    def test_validate_predictions_non_numeric_confidence(self):
        bad = [{"bbox": [0, 0, 1, 1], "confidence": "high", "img_id": 1}]
        assert validate_predictions(bad) is False

    def test_validate_predictions_ok(self):
        good = [{"bbox": [0, 0, 1, 1], "confidence": 0.9, "img_id": 1}]
        assert validate_predictions(good) is True

    def test_validate_ground_truths_empty(self):
        assert validate_ground_truths([]) is True

    def test_validate_ground_truths_missing_keys(self):
        assert validate_ground_truths([{"bbox": [0, 0, 1, 1]}]) is False


# ------------------------------------------------- classification → detection --
class TestClassificationToDetections:
    def test_shapes_and_class_offset(self):
        images = torch.zeros(2, 3, 64, 64)
        preds = torch.tensor([0, 1])
        labels = torch.tensor([1, 0])
        confs = torch.tensor([0.9, 0.7])
        paths = ["a.jpg", "b.jpg"]

        pred_list, gt_list = classification_predictions_to_detections(
            images, preds, labels, paths, confs
        )

        assert len(pred_list) == 2 and len(gt_list) == 2
        # class_id is +1 over raw label
        assert pred_list[0]["class_id"] == 1
        assert gt_list[0]["class_id"] == 2
        assert pred_list[0]["bbox"] == [0, 0, 64, 64]
        assert pred_list[0]["confidence"] == pytest.approx(0.9)
        assert pred_list[0]["image_id"] == "a.jpg"


# --------------------------------------------------------------- CorLoc/mAP --
class TestCorLoc:
    """calculate_corloc operates on WSOD-style pred tuples and uses VisUtils.get_IoU."""

    def test_perfect_localization_gives_one(self):
        # GT format: [x, y, w, h, class_id]
        bbox_to_id = {"img1": [[0, 0, 100, 100, 1]]}
        # Prediction format: (img_id, bbs, resize, image_sizes, pred_label)
        # bb is corners (x1,y1,x2,y2[,conf]); pred_label is 0-based, class_id=label+1
        all_predictions = [("img1", [(0, 0, 100, 100, 1.0)], 224, {}, 0)]
        score = calculate_corloc(all_predictions, bbox_to_id, loc_threshold=0.5)
        assert score == pytest.approx(1.0)

    def test_no_overlap_gives_zero(self):
        bbox_to_id = {"img1": [[0, 0, 50, 50, 1]]}
        all_predictions = [("img1", [(200, 200, 300, 300, 1.0)], 224, {}, 0)]
        score = calculate_corloc(all_predictions, bbox_to_id, loc_threshold=0.5)
        assert score == 0.0

    def test_class_mismatch_skipped(self):
        # GT has class 1, prediction is class 2 (label=1) → skipped, no count
        bbox_to_id = {"img1": [[0, 0, 100, 100, 1]]}
        all_predictions = [("img1", [(0, 0, 100, 100, 1.0)], 224, {}, 1)]
        score = calculate_corloc(all_predictions, bbox_to_id, loc_threshold=0.5)
        # No samples for class 1 (pred_class != gt class), no class-2 GTs → 0
        assert score == 0.0


class TestCalculateMap:
    def test_perfect_predictions(self):
        # GTs in COCO [x,y,w,h,class_id]
        bbox_to_id = {"img1": [[0, 0, 100, 100, 1]]}
        # bb format: (x1,y1,x2,y2,confidence)
        all_predictions = [("img1", [(0, 0, 100, 100, 1.0)], 224, {}, 0)]
        m = calculate_map(all_predictions, bbox_to_id, iou_thresholds=[0.5])
        assert m == pytest.approx(1.0)

    def test_empty_predictions_gives_zero(self):
        bbox_to_id = {"img1": [[0, 0, 100, 100, 1]]}
        m = calculate_map([], bbox_to_id, iou_thresholds=[0.5])
        assert m == 0.0


class TestDetectionConfusion:
    def test_perfect_detection(self):
        bbox_to_id = {"img1": [[0, 0, 100, 100]]}  # COCO [x,y,w,h]
        all_predictions = [("img1", [(0, 0, 100, 100)], 224, {}, 0)]
        result = calculate_detection_confusion(all_predictions, bbox_to_id, iou_threshold=0.5)
        assert result["tp_det"] == 1
        assert result["fp_det"] == 0
        assert result["fn_det"] == 0
        assert result["precision_det"] == pytest.approx(1.0)
        assert result["recall_det"] == pytest.approx(1.0)

    def test_unmatched_gt_counts_as_fn(self):
        bbox_to_id = {"img1": [[0, 0, 50, 50]]}
        # Prediction far away
        all_predictions = [("img1", [(500, 500, 600, 600)], 224, {}, 0)]
        result = calculate_detection_confusion(all_predictions, bbox_to_id, iou_threshold=0.5)
        assert result["tp_det"] == 0
        assert result["fp_det"] == 1
        assert result["fn_det"] == 1

    def test_image_with_no_predictions_counts_fns(self):
        bbox_to_id = {"img1": [[0, 0, 50, 50], [10, 10, 20, 20]]}
        result = calculate_detection_confusion([], bbox_to_id, iou_threshold=0.5)
        assert result["fn_det"] == 2
