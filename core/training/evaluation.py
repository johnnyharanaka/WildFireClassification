"""Evaluation functions for model validation and testing.

Provides functions for evaluating models during training (validation epochs),
final testing with metrics, and single batch evaluation.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from ..config.config import get_active_dataset_config, get_config, log_info


def eval_fn(model, features, tgt):
    """Evaluate model on a batch of features.

    Args:
        model: Model to evaluate
        features: Input features
        tgt: Target labels

    Returns:
        tuple: (outputs, predictions, accuracy)
    """
    outputs = model(features)
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == tgt).float().mean()
    return outputs, predictions, accuracy


def val_epoch(model, val_iter, device, criterion, epoch, total_epoch, contrastive_losses=None, use_classification_loss=True):
    """Run validation for one epoch.

    Args:
        model: Model to validate
        val_iter: Validation data loader
        device: Device (cuda/mps/cpu)
        criterion: Loss function for classification
        epoch: Current epoch number
        total_epoch: Total number of epochs
        contrastive_losses: Dict with contrastive loss configs, or None if disabled

    Returns:
        tuple: (mean_loss, mean_accuracy)
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Track individual loss components for detailed logging
    total_cls_loss = 0.0
    total_supcon_loss = 0.0
    total_triplet_loss = 0.0
    total_center_loss = 0.0

    # Check if classification loss is disabled (parameter from train_model)
    disable_cls_loss = not use_classification_loss

    # Check which contrastive losses are enabled
    use_supcon = contrastive_losses is not None and contrastive_losses.get('supcon') is not None
    use_triplet = contrastive_losses is not None and contrastive_losses.get('triplet') is not None
    use_center = contrastive_losses is not None and contrastive_losses.get('center') is not None

    log_info("Starting Evaluation...")
    model.eval()
    with torch.no_grad():
        loop = tqdm(val_iter, desc=f"Epoch {epoch + 1}/{total_epoch}")
        for batch_counter, (features, labels, _) in enumerate(loop):
            features = features.to(device)
            labels = labels.to(device)

            # Initialize loss
            loss = torch.tensor(0.0, device=device)

            # Determine what we need to extract during validation
            need_embedding = use_supcon or use_triplet or use_center

            # Forward pass - extract what's needed
            if need_embedding:
                outputs, embeddings = model(features, return_embedding=True)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            else:
                outputs = model(features)

            # Classification loss (only if not disabled)
            if not disable_cls_loss:
                has_contrastive = use_supcon or use_triplet or use_center
                cls_weight = float(get_config("contrastive", "classification_weight") or 1.0) if has_contrastive else 1.0
                cls_loss = cls_weight * criterion(outputs, labels)
                loss = cls_loss
                total_cls_loss += cls_loss.item()

            # Add contrastive losses (independent of classification loss)
            # This ensures val_loss matches the training objective
            if contrastive_losses is not None:
                # SupCon loss
                if use_supcon:
                    supcon_config = contrastive_losses['supcon']
                    supcon_loss_func = supcon_config['loss_func']
                    supcon_weight = supcon_config['weight']
                    supcon_miner = supcon_config['miner']

                    if supcon_miner is not None:
                        hard_pairs = supcon_miner(embeddings, labels)
                        supcon_loss = supcon_loss_func(embeddings, labels, hard_pairs)
                    else:
                        supcon_loss = supcon_loss_func(embeddings, labels)

                    weighted_supcon = supcon_weight * supcon_loss
                    loss = loss + weighted_supcon
                    total_supcon_loss += weighted_supcon.item()

                # Triplet loss
                if use_triplet:
                    triplet_config = contrastive_losses['triplet']
                    triplet_loss_func = triplet_config['loss_func']
                    triplet_weight = triplet_config['weight']
                    triplet_miner = triplet_config['miner']

                    if triplet_miner is not None:
                        hard_pairs = triplet_miner(embeddings, labels)
                        triplet_loss = triplet_loss_func(embeddings, labels, hard_pairs)
                    else:
                        triplet_loss = triplet_loss_func(embeddings, labels)

                    weighted_triplet = triplet_weight * triplet_loss
                    loss = loss + weighted_triplet
                    total_triplet_loss += weighted_triplet.item()

                # Center loss
                if use_center:
                    center_config = contrastive_losses['center']
                    center_loss_func = center_config['loss_func']
                    center_weight = center_config['weight']

                    center_loss = center_loss_func(embeddings, labels)
                    weighted_center = center_weight * center_loss
                    loss = loss + weighted_center
                    total_center_loss += weighted_center.item()

            predictions = outputs.argmax(dim=1)
            correct = (predictions == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

        mean_loss = total_loss / len(val_iter)
        mean_acc = total_correct / total_samples

        # Compute mean for each loss component
        num_batches = len(val_iter)
        mean_cls_loss = total_cls_loss / num_batches if not disable_cls_loss else 0.0
        mean_supcon_loss = total_supcon_loss / num_batches if use_supcon else 0.0
        mean_triplet_loss = total_triplet_loss / num_batches if use_triplet else 0.0
        mean_center_loss = total_center_loss / num_batches if use_center else 0.0

        # Log overall metrics
        log_info(f"Epoch {epoch+1}/{total_epoch}: Val Loss = {mean_loss:.4f}, Val Acc = {mean_acc:.4f}")

        # Log loss breakdown only when multiple losses are active
        active_losses = []
        if not disable_cls_loss:
            active_losses.append(f"Classification: {mean_cls_loss:.4f}")
        if use_supcon:
            active_losses.append(f"SupCon: {mean_supcon_loss:.4f}")
        if use_triplet:
            active_losses.append(f"Triplet: {mean_triplet_loss:.4f}")
        if use_center:
            active_losses.append(f"Center: {mean_center_loss:.4f}")
        if len(active_losses) > 1:
            log_info("  Loss Breakdown: " + " | ".join(active_losses))
        log_info("")

    model.train()
    return mean_loss, mean_acc


def test_epoch(model, test_iter, device, criterion, compute_map_metric: bool = False):
    """Evaluate model on the test set.

    Args:
        model: Model to evaluate
        test_iter: Test data loader
        device: Device (cuda/mps/cpu)
        criterion: Loss function
        compute_map_metric: If True, compute mAP in addition to F1

    Returns:
        float: F1 score or mAP (depending on configuration)
    """
    accs = []
    losses = []
    all_preds = []
    all_labels = []
    all_paths = []
    all_probs = []

    log_info("Starting Final Evaluation...")
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_iter)

        for batch_counter, batch_data in enumerate(loop):
            if len(batch_data) == 3:
                features, labels, paths = batch_data
            elif len(batch_data) == 4:
                features, labels, paths, _ = batch_data
            else:
                features, labels, paths = batch_data[0], batch_data[1], batch_data[2]

            features = features.to(device)
            labels = labels.to(device)

            outputs, preds, acc = eval_fn(model, features, labels)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]

            accs.append(acc.item())
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
            all_probs.extend(max_probs.cpu().numpy())

        mean_acc = torch.mean(torch.tensor(accs))
        mean_loss = torch.mean(torch.tensor(losses))

        eval_metric = get_active_dataset_config('eval_metric') or 'f1'
        f1_average = get_active_dataset_config('f1_average') or 'binary'

        f1 = f1_score(all_labels, all_preds, average=f1_average)
        log_info(f"F1 Score: {f1:.4f}")
        log_info(f"Test Accuracy: {mean_acc:.4f}")

        if compute_map_metric and eval_metric == 'mAP':
            try:
                from ..metrics import classification_predictions_to_detections, compute_map, format_map_results
                from ..datasets.nwpu_dataset import CLASS_MAPPING

                log_info("\nCalculating mAP metric...")

                num_classes = get_active_dataset_config('num_classes') or 10

                images_tensor = torch.zeros((len(all_preds), 3, 224, 224))
                preds_tensor = torch.tensor(all_preds, dtype=torch.long)
                labels_tensor = torch.tensor(all_labels, dtype=torch.long)
                probs_tensor = torch.tensor(all_probs, dtype=torch.float32)

                predictions_list, ground_truths_list = classification_predictions_to_detections(
                    images=images_tensor,
                    predictions=preds_tensor,
                    labels=labels_tensor,
                    image_paths=all_paths,
                    confidences=probs_tensor
                )

                map_results = compute_map(
                    predictions=predictions_list,
                    ground_truths=ground_truths_list,
                    num_classes=num_classes,
                    iou_threshold=0.5,
                    class_names=CLASS_MAPPING
                )

                log_info(format_map_results(map_results, verbose=True))

                return map_results['mAP']

            except Exception as e:
                log_info(f"Error calculating mAP: {e}")
                log_info("Returning F1 score instead")

    model.train()

    return f1


def evaluate_classification_metrics(model, device, batch_size=16, num_workers=0):
    """Evaluate classification metrics (Accuracy, F1-score) on test set.

    Args:
        model: Trained model
        device: Device (cuda/mps/cpu)
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        dict: {'accuracy': float, 'f1': float}
    """
    from sklearn.metrics import accuracy_score
    from ..datasets import get_data_loaders

    root_path = get_active_dataset_config('data_path') or get_active_dataset_config('root_path') or 'data/'
    data_loaders, _ = get_data_loaders(
        root_path=root_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=256
    )
    test_loader = data_loaders.get('Test')

    if not test_loader:
        log_info("No test loader found")
        return {'accuracy': 0.0, 'f1': 0.0}

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) >= 2:
                images, labels = batch_data[0], batch_data[1]
            else:
                continue

            images = images.to(device)
            outputs = model(images)

            if isinstance(outputs, dict) and 'logits' in outputs:
                preds = torch.argmax(outputs['logits'], dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_average = get_active_dataset_config('f1_average') or 'weighted'
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    log_info(f"\n{'='*60}")
    log_info("Classification Metrics (Test Set)")
    log_info(f"{'='*60}")
    log_info(f"Accuracy: {accuracy:.4f}")
    log_info(f"F1-score ({f1_average}): {f1:.4f}")
    log_info(f"{'='*60}\n")

    return {'accuracy': accuracy, 'f1': f1}
