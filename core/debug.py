import os
import json
import glob

import cv2
# Set matplotlib backend to non-interactive to avoid GUI issues when running in threads
import matplotlib
matplotlib.use('Agg')  # Use Agg backend - saves figures without displaying
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from core.config.config import get_config, log_info
from .datasets import get_data_loaders
from .attention_rollout import compute_multiscale_attention_maps
from .attention_map import get_last_layer_attention
from .gradcam import (
    compute_gradcam,
    refine_heatmap_with_crf,
)
from .vis_utils import VisUtils, compute_dynamic_threshold

def get_root_path():
    """Returns the active dataset path.

    Returns:
        str: Path to the active dataset.
    """
    return get_config("datasets", get_config("active_dataset"), "data_path")


ROOT_PATH = get_root_path()

def remove_nested_bboxes(bboxes, containment_threshold=0.8):
    """Removes smaller bounding boxes when completely contained within larger ones.

    Args:
        bboxes: List of bounding boxes in format (x1, y1, x2, y2, confidence) or (x1, y1, x2, y2).
        containment_threshold: Percentage of smaller box that must be inside larger box (default: 0.8 = 80%).

    Returns:
        List of filtered bounding boxes with nested ones removed.
    """
    if len(bboxes) <= 1:
        return bboxes

    def get_containment(small_box, large_box):
        """Calculates how much of the smaller box is inside the larger box.

        Args:
            small_box: Smaller bounding box.
            large_box: Larger bounding box.

        Returns:
            float: Fraction (0-1) of smaller box area inside larger box.
        """
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

        if small_area == 0:
            return 0.0

        return inter_area / small_area

    def get_box_area(box):
        """Calculates bounding box area.

        Args:
            box: Bounding box.

        Returns:
            float: Area of the bounding box.
        """
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

            containment = get_containment(bboxes[idx_j], bboxes[idx_i])

            if containment > containment_threshold:
                keep[idx_j] = False

    filtered_bboxes = [bboxes[i] for i in range(len(bboxes)) if keep[i]]

    return filtered_bboxes


def classify_cropped_regions(model, feature, filtered_bbs, img_size, device, mean, std, save_embeddings=False):
    """Classifies each cropped bounding box region with black background outside the box.

    Args:
        model: Classification model.
        feature: Image feature tensor.
        filtered_bbs: List of filtered bounding boxes.
        img_size: Target image size.
        device: Device to run inference on.
        mean: Normalization mean values.
        std: Normalization standard deviation values.
        save_embeddings: Whether to save embeddings (default: False).

    Returns:
        tuple: (results, embeddings_data) where results is a list of classification results
            and embeddings_data contains embedding information if save_embeddings is True.
    """
    results = []
    embeddings_list = []
    confidence_list = []
    class_list = []
    class_names = {0: 'Fire', 1: 'No Fire'}

    for idx, bb in enumerate(filtered_bbs):
        x, y, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

        if (x2 - x) < 10 or (y2 - y) < 10:
            continue

        black_image = torch.zeros_like(feature)
        black_image[:, y:y2, x:x2] = feature[:, y:y2, x:x2]
        h, w = black_image.shape[1], black_image.shape[2]

        if h != img_size or w != img_size:
            padded = torch.zeros_like(black_image).reshape(3, img_size, img_size) if h == 0 else torch.zeros(3, img_size, img_size)
            y_offset = (img_size - h) // 2
            x_offset = (img_size - w) // 2

            if y_offset + h <= img_size and x_offset + w <= img_size:
                padded[:, y_offset:y_offset+h, x_offset:x_offset+w] = black_image
            black_resized = padded
        else:
            black_resized = black_image

        with torch.no_grad():
            black_batch = black_resized.unsqueeze(0).to(device)
            logits, embeddings = model(black_batch, return_embedding=True)

            if isinstance(logits, dict):
                logits = logits.get('logits', logits)

            pred_probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(pred_probs, dim=1).item()
            pred_confidence = pred_probs[0, pred_class].item()

            if save_embeddings:
                embeddings_list.append(embeddings.cpu().numpy().flatten())
                confidence_list.append(pred_confidence)
                class_list.append(pred_class)

        black_viz = black_resized.cpu().permute(1, 2, 0).numpy() * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
        black_viz = np.clip(black_viz, 0, 1)

        class_name = class_names.get(pred_class, f'Class {pred_class}')
        results.append({
            'image': black_viz,
            'label': class_name,
            'confidence': pred_confidence,
            'bb_idx': idx
        })

    embeddings_data = None
    if save_embeddings and len(embeddings_list) > 0:
        embeddings_data = {
            'embeddings': np.array(embeddings_list),
            'confidences': np.array(confidence_list),
            'classes': np.array(class_list),
            'class_names': class_names
        }

    return results, embeddings_data


class BBoxVisualizer:
    """Bounding box visualization class for debugging."""

    def __init__(self, root_path):
        """Initializes the visualizer.

        Args:
            root_path: Root path to the dataset.
        """
        self.root_path = root_path

    @staticmethod
    def generate_heatmaps(model, model_name, feature_unsqueeze):
        """Generates heatmaps (AttentionRollout, GradCAM, and multiscale) from the model.

        Args:
            model: The model to generate heatmaps from.
            model_name: Name of the model.
            feature_unsqueeze: Feature tensor with batch dimension.

        Returns:
            tuple: (heatmap_attention, heatmap_gradcam)
        """
        heatmap_attention = None
        heatmap_gradcam = None

        if "DinoV2" in model_name or "Dino" in model_name or "DeiT" in model_name or "ViT" in model_name or "Swin" in model_name:
            heatmap_gradcam = None
            try:
                heatmap_attention = get_last_layer_attention(model, model_name, feature_unsqueeze)
            except Exception as e:
                log_info(f"⚠ Last layer attention extraction failed: {e}")
                heatmap_attention = None
        else:
            heatmap_attention = None
            heatmap_gradcam = None

        return heatmap_attention, heatmap_gradcam

    @staticmethod
    def colorize_heatmap(heatmap):
        """Applies colormap to a heatmap.

        Args:
            heatmap: Heatmap array to colorize.

        Returns:
            np.ndarray: Colorized heatmap in RGB format, or None if input is None.
        """
        if heatmap is None:
            return None
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    @staticmethod
    def apply_crf_refinement(heatmap):
        """Applies CRF refinement to the heatmap.

        Args:
            heatmap: Heatmap array to refine.

        Returns:
            np.ndarray: Refined heatmap, or None if input is None.
        """
        if heatmap is None:
            return None
        crf_config = get_config("crf") or {}
        if crf_config.get('enabled', False):
            heatmap = refine_heatmap_with_crf(
                heatmap, image=None, use_crf=True, crf_config=crf_config
            )
        return heatmap

    @staticmethod
    def _plot_combined_heatmap(ax, heatmap_attention, heatmap_gradcam, img_np):
        """Plots combined heatmap on provided axis.

        Args:
            ax: Matplotlib axis to plot on.
            heatmap_attention: Attention heatmap.
            heatmap_gradcam: GradCAM heatmap.
            img_np: Original image as numpy array.

        Returns:
            np.ndarray: Combined colorized heatmap, or None if unavailable.
        """
        heatmap_combined_color = None

        if heatmap_attention is not None and heatmap_gradcam is not None:
            heatmap_attention_norm = (heatmap_attention - heatmap_attention.min()) / (heatmap_attention.max() - heatmap_attention.min() + 1e-8)
            heatmap_gradcam_norm = (heatmap_gradcam - heatmap_gradcam.min()) / (heatmap_gradcam.max() - heatmap_gradcam.min() + 1e-8)

            layer1 = heatmap_attention
            layer2 = heatmap_gradcam
            layer3 = heatmap_attention_norm * heatmap_gradcam_norm
            heatmap_combined = 0.3 * layer1 + 0.0 * layer2 + 1.0 * layer3
            heatmap_combined = np.clip(heatmap_combined, 0, 1)

            heatmap_combined_color = BBoxVisualizer.colorize_heatmap(heatmap_combined)
            ax.imshow(heatmap_combined_color)
            ax.set_title('Combined Heatmap\n(0.2×Attn + 0.2×GradCAM + 0.6×(Attn⊙GradCAM))',
                        fontsize=11, color='purple', weight='bold')
            ax.axis('off')
        elif heatmap_attention is not None or heatmap_gradcam is not None:
            available_heatmap = heatmap_attention if heatmap_attention is not None else heatmap_gradcam
            available_heatmap_norm = (available_heatmap - available_heatmap.min()) / (available_heatmap.max() - available_heatmap.min() + 1e-8)
            heatmap_combined_color = BBoxVisualizer.colorize_heatmap(available_heatmap_norm)
            ax.imshow(heatmap_combined_color)
            ax.set_title('Combined Heatmap\n(only one method available)', fontsize=11, color='purple', style='italic')
            ax.axis('off')
        else:
            ax.imshow(img_np)
            ax.set_title('Combined unavailable', fontsize=12, style='italic', color='gray')
            ax.axis('off')

        return heatmap_combined_color

    @staticmethod
    def _plot_classifier_labels(ax, img_np, model, feature, filtered_bbs, img_size,
                               device, mean, std, save_cropped_predictions):
        """Plots classifier labels on bounding boxes.

        Args:
            ax: Matplotlib axis to plot on.
            img_np: Original image as numpy array.
            model: Classification model.
            feature: Image feature tensor.
            filtered_bbs: List of filtered bounding boxes.
            img_size: Target image size.
            device: Device to run inference on.
            mean: Normalization mean values.
            std: Normalization standard deviation values.
            save_cropped_predictions: Whether to perform classification on crops.
        """
        ax.imshow(img_np)
        if save_cropped_predictions and len(filtered_bbs) > 0:
            cropped_results, _ = classify_cropped_regions(
                model, feature, filtered_bbs, img_size, device, mean, std, save_embeddings=False
            )

            if len(cropped_results) > 0:
                fire_count = 0
                for bb_idx, (bb, crop_data) in enumerate(zip(filtered_bbs, cropped_results)):
                    if crop_data["label"].lower() == 'fire':
                        x, y, x2, y2 = bb[:4]
                        width = x2 - x
                        height = y2 - y

                        label_text = f'{width}x{height}\n{crop_data["label"]}\nprob: {crop_data["confidence"]:.3f}'

                        rect = patches.Rectangle(
                            (x, y), width, height,
                            linewidth=2, edgecolor='blue', facecolor='none', label='Fire'
                        )
                        ax.add_patch(rect)
                        ax.text(x, y-5, label_text,
                               color='blue', fontsize=8, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                        fire_count += 1

                ax.set_title(f'Classifier Labels (Fire only - {fire_count} boxes)', fontsize=12, color='blue')
            else:
                ax.text(0.5, 0.5, 'No valid crops', ha='center', va='center')
                ax.set_title('Classifier Labels', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No BBs to classify', ha='center', va='center', fontsize=10)
            ax.set_title('Classifier Labels', fontsize=12, style='italic', color='gray')

        ax.axis('off')

    @staticmethod
    def _plot_postprocessed_bboxes(ax, img_np, bbs):
        """Plots bounding boxes after post-processing.

        Args:
            ax: Matplotlib axis to plot on.
            img_np: Original image as numpy array.
            bbs: List of post-processed bounding boxes.
        """
        ax.imshow(img_np)

        for bb_idx, bb in enumerate(bbs):
            x, y, x2, y2 = bb[:4]
            confidence = bb[4] if len(bb) > 4 else 0.0
            width = x2 - x
            height = y2 - y

            label_text = f'{width:.0f}x{height:.0f}\nconf: {confidence:.3f}'

            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2.5, edgecolor='purple', facecolor='none', label='Post-processed'
            )
            ax.add_patch(rect)
            ax.text(x, y-5, label_text,
                   color='purple', fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(f'Post-processed BBs ({len(bbs)} after filtering)', fontsize=12, color='purple', weight='bold')
        ax.axis('off')

    @staticmethod
    def _calculate_average_iou(gt_bboxes, bbs, img_id, image_sizes, img_size, or_size):
        """Calculates average IoU between ground truth and predictions.

        Args:
            gt_bboxes: List of ground truth bounding boxes.
            bbs: List of predicted bounding boxes.
            img_id: Image identifier.
            image_sizes: Dictionary mapping image IDs to original dimensions.
            img_size: Target image size.
            or_size: Original size for scaling.

        Returns:
            float: Average IoU score across all ground truth boxes.
        """
        max_ious = []
        for gt in gt_bboxes:
            if img_id in image_sizes:
                original_width, original_height = image_sizes[img_id]
                gt_x, gt_y, gt_width, gt_height = VisUtils.redimension_bboxes(
                    (int(round(v)) for v in gt[:4]),
                    img_size,
                    or_size,
                    original_width,
                    original_height
                )
            else:
                gt_x, gt_y, gt_width, gt_height = VisUtils.redimension_bboxes(
                    (int(round(v)) for v in gt[:4]), img_size, or_size
                )
            best_iou = 0
            for bb in bbs:
                x, y, x2, y2 = bb[:4]
                iou = VisUtils.get_IoU(
                    (gt_x, gt_y, gt_x + gt_width, gt_y + gt_height),
                    (x, y, x2, y2)
                )
                best_iou = max(best_iou, iou)
            max_ious.append(best_iou)

        return np.mean(max_ious) if max_ious else 0.0

    def save_bbox_visualizations(self, model, model_name, device, num_workers, output_dir='bbox_debug',
                                 num_samples=40, mask_threshold=0.5, score_threshold=0.3,
                                 save_cropped_predictions=True, remove_nested_bbs=True, split="Test",
                                 save_all_layers=False):
        """Saves bounding box visualizations for debugging purposes.

        Generates visualization images comparing predicted bounding boxes against ground truth.

        Args:
            model: Classification model to evaluate.
            model_name: Name identifier of the model.
            device: PyTorch device to run inference on.
            num_workers: Number of data loader workers.
            output_dir: Directory to save visualization images (default: 'bbox_debug').
            num_samples: Maximum number of samples to visualize (default: 40).
            mask_threshold: Threshold for heatmap binarization (default: 0.5).
            score_threshold: Unused parameter (kept for backward compatibility).
            save_cropped_predictions: Whether to classify cropped regions (default: True).
            remove_nested_bbs: Whether to remove nested bounding boxes (default: True).
            split: Dataset split to use (default: 'Test').
            save_all_layers: Whether to save all 12 transformer layer visualizations (default: False).
        """
        os.makedirs(output_dir, exist_ok=True)
        log_info(f"\n{'='*60}")
        log_info(f"Saving bounding box visualizations to: {output_dir}")
        log_info(f"{'='*60}\n")

        img_size = 224
        if get_config("models", model_name, "img_size") is not None:
            img_size = int(get_config("models", model_name, "img_size"))

        data_loaders, datasets = get_data_loaders(self.root_path, batch_size=1, num_workers=num_workers, img_size=img_size)
        bbox_to_id, filename_to_id, image_sizes = VisUtils.get_image_ids(split=split)

        model.eval()
        saved_count = 0
        fire_count = 0
        no_fire_count = 0
        samples_per_class = num_samples // 2

        # List to store embedding data for JSON export
        embeddings_data = []

        for idx, (features, labels, paths) in enumerate(data_loaders[split]):
            # Stop if we have enough samples from both classes
            if fire_count >= samples_per_class and no_fire_count >= samples_per_class:
                break

            features = features.to(device)
            labels_tensor = labels.to(device)
            feature = features[0]
            label = labels[0]
            path = paths[0]

            feature_unsqueeze = feature.unsqueeze(0)

            heatmap_attention, heatmap_gradcam = self.generate_heatmaps(
                model, model_name, feature_unsqueeze
            )

            # Generate multiscale attention maps (early/mid/late layers)
            multiscale_maps = None
            if "DinoV2" in model_name or "Dino" in model_name:
                try:
                    multiscale_maps = compute_multiscale_attention_maps(
                        model, model_name, feature_unsqueeze, use_mean_patches=False
                    )
                except Exception as e:
                    log_info(f"⚠ Multiscale attention maps failed: {e}")
                    multiscale_maps = None

            heatmap_attention_color = None

            # Colorize attention heatmap
            if heatmap_attention is not None:
                heatmap_attention = cv2.resize(heatmap_attention, (feature_unsqueeze.shape[2], feature_unsqueeze.shape[3]))
                heatmap_attention = self.apply_crf_refinement(heatmap_attention)
                heatmap_attention_color = self.colorize_heatmap(heatmap_attention)

            heatmap = heatmap_attention

            if heatmap is not None:
                # Use dynamic threshold (Otsu) instead of fixed threshold
                gradcam_threshold = compute_dynamic_threshold(heatmap, method="otsu", fire_adapted=True)
                bbs = VisUtils.generate_bounding_box(heatmap, threshold=gradcam_threshold)
            else:
                bbs = []

            threshold_info = f"Mask threshold: {mask_threshold}"

            if remove_nested_bbs and len(bbs) > 1:
                bbs_antes = len(bbs)
                bbs = remove_nested_bboxes(bbs, containment_threshold=0.5)
                if len(bbs) < bbs_antes:
                    log_info(f"BBs removed (nested): {bbs_antes} → {len(bbs)}")

            # Check class and update counters
            true_label = label.item()

            # Skip if we already have enough samples of this class
            if true_label == 0 and fire_count >= samples_per_class:
                continue
            elif true_label == 1 and no_fire_count >= samples_per_class:
                continue

            img_id = filename_to_id.get(os.path.basename(path))

            # Only generate bbox visualizations for Fire class (label == 0)
            should_visualize = (true_label == 0)

            if should_visualize:
                # For Fire class, we need gt_bboxes for visualization
                if not img_id:
                    continue
                gt_bboxes = bbox_to_id.get(img_id, [])
                if len(gt_bboxes) == 0:
                    continue
            else:
                # For No Fire class, we just collect embeddings (no visualization)
                # Extract embedding and skip to end
                with torch.no_grad():
                    # Get the class prediction
                    logits = model(feature_unsqueeze)
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    # Extract backbone embedding
                    if hasattr(model, 'backbone'):
                        features_dict = model.backbone.forward_features(feature_unsqueeze)
                        if isinstance(features_dict, dict):
                            cls_embedding = features_dict.get("x_norm_clstoken", features_dict.get("x_norm_patchtokens"))
                        else:
                            cls_embedding = features_dict

                        embedding_np = cls_embedding.cpu().numpy().flatten().tolist()

                        embeddings_data.append({
                            "image_name": os.path.basename(path),
                            "true_class": true_label,
                            "embedding": embedding_np
                        })

                no_fire_count += 1
                if (fire_count + no_fire_count) % 5 == 0:
                    log_info(f"Processed {fire_count} Fire + {no_fire_count} No Fire = {fire_count + no_fire_count} total")
                continue

            if img_id in image_sizes:
                original_width, original_height = image_sizes[img_id]
                or_size = min(original_width, original_height)
            else:
                or_size = 256

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = feature.cpu().permute(1, 2, 0).numpy() * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
            img_np = np.clip(img_np, 0, 1)

            # Use 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            axes[0].imshow(img_np)
            axes[0].set_title('Original Image (224x224)', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(img_np)
            for gt in gt_bboxes:
                if img_id in image_sizes:
                    gt_x, gt_y, gt_width, gt_height = VisUtils.redimension_bboxes(
                        (int(round(v)) for v in gt[:4]),
                        img_size, or_size, original_width, original_height
                    )
                else:
                    gt_x, gt_y, gt_width, gt_height = VisUtils.redimension_bboxes(
                        (int(round(v)) for v in gt[:4]),
                        img_size, or_size
                    )
                rect = patches.Rectangle(
                    (gt_x, gt_y), gt_width, gt_height,
                    linewidth=2, edgecolor='green', facecolor='none', label='GT'
                )
                axes[1].add_patch(rect)
                axes[1].text(gt_x, gt_y-5, f'{gt_width}x{gt_height}',
                             color='green', fontsize=8, weight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            axes[1].set_title(f'Ground Truth ({len(gt_bboxes)} boxes)', fontsize=12, color='green')
            axes[1].axis('off')

            axes[2].imshow(img_np)
            min_confidence = get_config("defaults", "min_confidence") or 0.0

            filtered_bbs = VisUtils.postprocess_BB(
                bbs,
                min_confidence=min_confidence,
                remove_nested=remove_nested_bbs,
                containment_threshold=0.5
            )

            bbs_antes = len(bbs)

            for bb_idx, bb in enumerate(filtered_bbs):
                x, y, x2, y2 = bb[:4]
                confidence = bb[4] if len(bb) > 4 else 0.0
                width = x2 - x
                height = y2 - y

                label_text = f'{width}x{height}\nconf: {confidence:.3f}'

                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=2, edgecolor='red', facecolor='none', label='Pred'
                )
                axes[2].add_patch(rect)
                axes[2].text(x, y-5, label_text,
                             color='red', fontsize=8, weight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            axes[2].set_title(f'Predictions ({len(filtered_bbs)}/{len(bbs)} boxes, min_conf={min_confidence:.2f})', fontsize=12, color='red')
            axes[2].axis('off')

            # Display Last Layer Attention
            if heatmap_attention_color is not None:
                axes[3].imshow(heatmap_attention_color)
                axes[3].set_title('Last Layer Attention', fontsize=12, color='orange', weight='bold')
                axes[3].axis('off')
            else:
                axes[3].imshow(img_np)
                axes[3].set_title('Last Layer Attention unavailable', fontsize=12, style='italic', color='gray')
                axes[3].axis('off')

            avg_iou = self._calculate_average_iou(
                gt_bboxes, bbs, img_id, image_sizes, img_size, or_size
            )

            info_text = f"[Attention-based] Image: {os.path.basename(path)}\n"
            if img_id in image_sizes:
                orig_w, orig_h = image_sizes[img_id]
                info_text += f"Original size: {orig_w}x{orig_h} | Resize: {img_size}x{img_size}\n"
            else:
                info_text += f"Resize: {img_size}x{img_size}\n"
            info_text += f"GT boxes: {len(gt_bboxes)} | Pred boxes: {len(bbs)}\n"
            info_text += f"Average IoU: {avg_iou:.4f} | {threshold_info}"

            plt.suptitle(info_text, fontsize=10, y=0.98)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"bbox_debug_{saved_count:03d}_{os.path.basename(path)}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Create separate figure with all 12 layers
            if save_all_layers and multiscale_maps is not None and len([k for k in multiscale_maps.keys() if k.startswith('layer_')]) >= 12:
                try:
                    fig_layers, axes_layers = plt.subplots(3, 4, figsize=(20, 15))
                    axes_layers = axes_layers.flatten()

                    for layer_idx in range(12):
                        layer_key = f'layer_{layer_idx}'
                        if layer_key in multiscale_maps:
                            layer_color = self.colorize_heatmap(multiscale_maps[layer_key])
                            axes_layers[layer_idx].imshow(layer_color)
                            axes_layers[layer_idx].set_title(f'Layer {layer_idx}', fontsize=12, weight='bold')
                            axes_layers[layer_idx].axis('off')
                        else:
                            axes_layers[layer_idx].imshow(img_np)
                            axes_layers[layer_idx].set_title(f'Layer {layer_idx} unavailable', fontsize=10, style='italic', color='gray')
                            axes_layers[layer_idx].axis('off')

                    plt.suptitle(f"All 12 Transformer Layers | {os.path.basename(path)}", fontsize=14, weight='bold')
                    plt.tight_layout()

                    layers_save_path = os.path.join(output_dir, f"all_layers_{saved_count:03d}_{os.path.basename(path)}.png")
                    plt.savefig(layers_save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    log_info(f"⚠ Error saving all layers visualization: {e}")

            # Ensure file is fully written to disk before continuing
            # This prevents "libpng error: Read Error" when GUI tries to read partially-written files
            try:
                import time
                import os as os_sync
                time.sleep(0.05)  # Small delay to ensure disk write completes
                if os.path.exists(save_path):
                    # Force sync to disk
                    os_sync.sync() if hasattr(os_sync, 'sync') else None
            except Exception:
                pass

            # Extract embedding from backbone and save to list (for Fire class)
            with torch.no_grad():
                # Extract backbone embedding
                if hasattr(model, 'backbone'):
                    # For DinoClassifier and similar models
                    features_dict = model.backbone.forward_features(feature_unsqueeze)
                    if isinstance(features_dict, dict):
                        cls_embedding = features_dict.get("x_norm_clstoken", features_dict.get("x_norm_patchtokens"))
                    else:
                        cls_embedding = features_dict

                    # Convert to numpy
                    embedding_np = cls_embedding.cpu().numpy().flatten().tolist()

                    # Store data
                    embeddings_data.append({
                        "image_name": os.path.basename(path),
                        "true_class": true_label,
                        "embedding": embedding_np
                    })

            saved_count += 1
            fire_count += 1
            if (fire_count + no_fire_count) % 5 == 0:
                log_info(f"Processed {fire_count} Fire + {no_fire_count} No Fire = {fire_count + no_fire_count} total (visualizations: {saved_count})")

        log_info(f"\nTotal of {saved_count} visualizations saved to: {output_dir}")
        log_info(f"Embeddings collected: {fire_count} Fire + {no_fire_count} No Fire = {fire_count + no_fire_count} total")

        # Save embeddings to JSON file
        if len(embeddings_data) > 0:
            json_path = os.path.join(output_dir, "embeddings.json")
            with open(json_path, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            log_info(f"Embeddings saved to: {json_path} ({len(embeddings_data)} samples)")

        log_info(f"{'='*60}\n")

def save_bbox_visualizations(model, model_name, device, num_workers, output_dir='bbox_debug',
                             num_samples=20, mask_threshold=0.5, score_threshold=0.3,
                             save_cropped_predictions=True, remove_nested_bbs=True, split="Test",
                             save_all_layers=False):
    """Wrapper function for backward compatibility with legacy code.

    Args:
        model: Classification/detection model to evaluate.
        model_name: Name identifier of the model.
        device: PyTorch device to run inference on.
        num_workers: Number of data loader workers.
        output_dir: Directory to save visualization images (default: 'bbox_debug').
        num_samples: Maximum number of samples to visualize (default: 20).
        mask_threshold: Threshold for heatmap binarization (default: 0.5).
        score_threshold: Confidence threshold for OICR detections (default: 0.3).
        save_cropped_predictions: Whether to classify cropped regions (default: True).
        remove_nested_bbs: Whether to remove nested bounding boxes (default: True).
        split: Dataset split to use (default: 'Test').
        save_all_layers: Whether to save all 12 transformer layer visualizations (default: False).
    """
    visualizer = BBoxVisualizer(ROOT_PATH)
    visualizer.save_bbox_visualizations(
        model, model_name, device, num_workers, output_dir,
        num_samples, mask_threshold, score_threshold,
        save_cropped_predictions, remove_nested_bbs, split, save_all_layers
    )


def process_multiple_checkpoints(get_model_fn, model_name, num_classes, device,
                                 num_workers, num_samples, models_dir="models",
                                 add_attn_maps_fn=None, split="Test", save_all_layers=False):
    """Process multiple epoch checkpoints and best model for bbox visualization.

    Automatically finds all epoch checkpoints (ep_1, ep_2, ...), sorts them numerically,
    and processes them along with the best model.

    Args:
        get_model_fn: Function to create a new model instance (e.g., get_model).
        model_name: Name of the model to process.
        num_classes: Number of classes for the model.
        device: PyTorch device to use.
        num_workers: Number of workers for data loading.
        num_samples: Number of samples to visualize per model.
        models_dir: Directory containing model checkpoints (default: 'models').
        add_attn_maps_fn: Optional function to add attention maps (for non-Dino models).
        split: Dataset split to use (default: 'Test').
        save_all_layers: Whether to save all 12 transformer layer visualizations (default: False).

    Returns:
        int: Number of models processed successfully.
    """

    epoch_models = glob.glob(os.path.join(models_dir, f"{model_name}_ep_*.pth"))

    # Sort numerically by epoch number (ep_1, ep_2, ..., ep_10, ep_11)
    epoch_models = sorted(epoch_models, key=lambda x: int(x.split('_ep_')[-1].replace('.pth', '')))
    best_model = os.path.join(models_dir, f"{model_name}.pth")

    models_to_process = []

    # Extract epoch number from filename (e.g., "DinoV2RS_Small_ep_1.pth" -> "ep_1")
    for ep_model_path in epoch_models:
        ep_name = os.path.basename(ep_model_path).replace(f"{model_name}_", "").replace(".pth", "")
        models_to_process.append((ep_model_path, ep_name))

    # Add best model if exists
    if os.path.exists(best_model):
        models_to_process.append((best_model, "best"))

    if len(models_to_process) == 0:
        log_info(f"No models found for {model_name}. Looking for:")
        return 0

    for model_path, label in models_to_process:
        log_info(f"\n{'='*60}")
        log_info(f"Processing model: {label}")
        log_info(f"{'='*60}\n")

        model = get_model_fn(model_name, num_classes=num_classes, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        if add_attn_maps_fn is not None and "Dino" not in model_name:
            model = add_attn_maps_fn(model)

        output_dir = os.path.join("bbox_debug", label)
        save_bbox_visualizations(
            model=model,
            model_name=model_name,
            device=device,
            num_workers=num_workers,
            num_samples=num_samples,
            output_dir=output_dir,
            split=split,
            save_all_layers=save_all_layers
        )

    log_info(f"\n{'='*60}")
    log_info(f"All models processed successfully!")
    log_info(f"{'='*60}\n")

    return len(models_to_process)