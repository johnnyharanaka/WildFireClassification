import copy
import glob as glob_module
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config.config import get_active_dataset_config, get_config, get_global_seed, get_root_path, log_info
from ..datasets import get_data_loaders
from .evaluation import val_epoch
from ..losses import get_loss, setup_contrastive_learning
from .lr_scheduler import EarlyStopping, get_optimizer, get_scheduler, get_optimizer_for_params, get_optimizer_with_lr
from .contrastive_engine import compute_contrastive_metrics
from ..metrics import evaluate_wsod
from ..models import add_attn_maps

MODELS_PATH = "models/"


def get_device() -> torch.device:
  """
  Get the best available device for computation.

  Returns:
      torch.device: CUDA if available, otherwise MPS if available, otherwise CPU
  """
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  return device

def train_one_epoch(train_iter, device, optimizer, scheduler, model, criterion, epoch, total_epoch, contrastive_losses, use_classification_loss=True):
  """
  Train model for one epoch with classifier and optional contrastive losses.

  Args:
      train_iter: Training data loader
      device: Device (cuda/mps/cpu)
      optimizer: Optimizer
      scheduler: Learning rate scheduler
      model: Model to train
      criterion: Loss function for classification
      epoch: Current epoch number
      total_epoch: Total number of epochs
      contrastive_losses: Dict with contrastive loss configs, or None if disabled
          Format: {'supcon': {...}, 'triplet': {...}}
          Each entry contains: {'loss_func': Loss, 'weight': float, 'miner': Miner or None}

  Returns:
      tuple: (mean_loss, mean_accuracy)
  """
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  # For tracking last batch embeddings to compute contrastive metrics
  last_embeddings = None
  last_labels = None

  # Check which contrastive losses are enabled
  use_supcon = contrastive_losses is not None and contrastive_losses.get('supcon') is not None
  use_triplet = contrastive_losses is not None and contrastive_losses.get('triplet') is not None
  use_center = contrastive_losses is not None and contrastive_losses.get('center') is not None

  model.train()
  loop = tqdm(train_iter, desc=f"Epoch {epoch + 1}/{total_epoch}")
  for idx, (features, labels, _) in enumerate(loop):
    features, labels = features.to(device), labels.to(device)

    optimizer.zero_grad()

    # Initialize outputs for accuracy computation
    outputs = None

    # Check if classification loss is disabled (parameter from train_model)
    disable_cls_loss = not use_classification_loss

    # Initialize total loss
    loss = None

    # =================================================================
    # FORWARD PASS - Extract all necessary features in ONE pass
    # =================================================================

    # Determine what we need to extract
    need_embedding = use_supcon or use_triplet or use_center

    # Single forward pass extracting everything needed
    if need_embedding:
      # SupCon/Triplet: Get outputs + embeddings
      outputs, embeddings = model(features, return_embedding=True)
      embeddings = F.normalize(embeddings, p=2, dim=1)
      last_embeddings = embeddings.detach()
      last_labels = labels
    else:
      # No contrastive learning
      outputs = model(features)

    # =================================================================
    # CLASSIFICATION LOSS
    # =================================================================
    if not disable_cls_loss:
      # Only apply classification_weight when contrastive losses are also active
      has_contrastive = use_supcon or use_triplet or use_center
      cls_weight = float(get_config("contrastive", "classification_weight") or 1.0) if has_contrastive else 1.0
      loss = cls_weight * criterion(outputs, labels)
    else:
      loss = torch.tensor(0.0, device=device, requires_grad=True)

    # =================================================================
    # SUPERVISED CONTRASTIVE LOSS (SupCon) - Image-level
    # =================================================================
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

      loss = loss + supcon_weight * supcon_loss

    # =================================================================
    # TRIPLET LOSS - Metric learning
    # =================================================================
    if use_triplet:
      triplet_config = contrastive_losses['triplet']
      triplet_loss_func = triplet_config['loss_func']
      triplet_weight = triplet_config['weight']
      triplet_miner = triplet_config['miner']

      # Compute Triplet loss
      if triplet_miner is not None:
        hard_pairs = triplet_miner(embeddings, labels)
        triplet_loss = triplet_loss_func(embeddings, labels, hard_pairs)
      else:
        triplet_loss = triplet_loss_func(embeddings, labels)

      loss = loss + triplet_weight * triplet_loss

    # =================================================================
    # CENTER LOSS - Intra-class compactness
    # =================================================================
    if use_center:
      center_config = contrastive_losses['center']
      center_loss_func = center_config['loss_func']
      center_weight = center_config['weight']

      # Compute Center loss (no miner needed)
      center_loss = center_loss_func(embeddings, labels)
      loss = loss + center_weight * center_loss

    loss.backward()
    optimizer.step()

    # Compute accuracy
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)

    total_loss += loss.item()
    total_correct += correct
    total_samples += labels.size(0)

    loop.set_postfix(loss=loss.item(), acc=accuracy)

  mean_loss = total_loss / len(train_iter)
  mean_accuracy = total_correct / total_samples

  if (use_supcon or use_triplet or use_center) and last_embeddings is not None:
    metrics = compute_contrastive_metrics(last_embeddings, last_labels)
    log_info(f"Epoch {epoch+1}/{total_epoch}: Train Loss = {mean_loss:.4f}, Train Acc = {mean_accuracy:.4f}")
    log_info(f"  Contrastive Metrics:")
    log_info(f"    Intra-class sim: {metrics['intra_class_sim']:.4f}")
    log_info(f"    Inter-class sim: {metrics['inter_class_sim']:.4f}")
    log_info(f"    Alignment: {metrics['alignment']:.4f}")
    log_info(f"    Uniformity: {metrics['uniformity']:.4f}")
  else:
    log_info(f"Epoch {epoch+1}/{total_epoch}: Train Loss = {mean_loss:.4f}, Train Acc = {mean_accuracy:.4f}")

  return mean_loss, mean_accuracy


def get_training_stages(epochs_stage1, epochs_stage2):
  """
  Determine training stages based on config.

  In two-stage mode:
    - Stage 1 (Contrastive): trains for epochs_stage1 epochs, tracks best WSOD metric
    - Stage 2 (Classification): loads best Stage 1 checkpoint, trains for epochs_stage2 epochs

  Returns:
      dict: {
          'mode': 'single' or 'two-stage',
          'epochs_stage1': int,
          'epochs_stage2': int,
          'contrastive_type': str or None
      }
  """
  selected = get_config("contrastive", "selected")

  if selected is None:
    # Single-stage: classification only
    return {
      'mode': 'single',
      'epochs_stage1': 0,
      'epochs_stage2': epochs_stage2,
      'contrastive_type': None
    }
  else:
    # Two-stage: contrastive then classification
    return {
      'mode': 'two-stage',
      'epochs_stage1': epochs_stage1,
      'epochs_stage2': epochs_stage2,
      'contrastive_type': selected
    }


def multi_start_stage1(model, data_loaders, device, criterion, contrastive_losses, epochs_stage1, model_name, eval_metric, wsod_metric_key, save_all_epochs, use_classification_loss=False, stage1_metric="mAP"):
  """
  Multi-start strategy for Stage 1: tries multiple learning rates,
  resets if no improvement within patience, keeps the best model overall.

  Args:
      model: Model to train
      data_loaders: Dictionary with 'Train' and 'Val' data loaders
      device: Device to train on
      criterion: Loss function
      contrastive_losses: Dict with contrastive loss configs
      epochs_stage1: Max epochs per trial
      model_name: Name of the model for checkpointing
      eval_metric: Name of the metric (for logging)
      wsod_metric_key: Key to extract from WSOD results ('map' or 'corloc')
      save_all_epochs: Whether to save all epoch checkpoints

  Returns:
      dict: Best model state_dict across all trials
  """
  multi_start_cfg = {
    'eval_after_epochs': int(get_config("multi_start", "eval_after_epochs") or 2),
    'min_threshold': float(get_config("multi_start", "min_threshold") or 0.05),
    'trials': get_config("multi_start", "trials"),
    'learning_rates': get_config("multi_start", "learning_rates"),
  }

  patience_epochs = multi_start_cfg['eval_after_epochs']
  min_threshold = multi_start_cfg['min_threshold']

  # Support both formats: new "trials" (list of {lr, weight_decay}) or legacy "learning_rates"
  if multi_start_cfg['trials']:
    trials = [{'lr': float(t['lr']), 'weight_decay': float(t['weight_decay'])} for t in multi_start_cfg['trials']]
  elif multi_start_cfg['learning_rates']:
    trials = [{'lr': float(lr), 'weight_decay': None} for lr in multi_start_cfg['learning_rates']]
  else:
    trials = [{'lr': 1e-5, 'weight_decay': None}, {'lr': 5e-5, 'weight_decay': None}, {'lr': 1e-4, 'weight_decay': None}]

  log_info(f"{'='*60}")
  log_info("MULTI-START STAGE 1: CONTRASTIVE LEARNING")
  log_info(f"Trials: {len(trials)}")
  for i, t in enumerate(trials):
    wd_str = f", WD={t['weight_decay']}" if t['weight_decay'] is not None else ""
    log_info(f"  [{i+1}] LR={t['lr']}{wd_str}")
  log_info(f"Patience: {patience_epochs} epochs without improvement")
  log_info(f"Min threshold: {min_threshold}")
  log_info(f"Max epochs per trial: {epochs_stage1}")
  log_info(f"Stage 1 selection metric: {stage1_metric}")
  log_info(f"{'='*60}")

  # Save initial weights to reset between trials
  initial_weights = copy.deepcopy(model.state_dict())

  global_best_metric = 0.0
  global_best_val_loss = float("inf")
  global_best_weights = copy.deepcopy(model.state_dict())
  global_best_trial = -1
  global_best_epoch = -1

  for trial_idx, trial in enumerate(trials):
    lr = trial['lr']
    wd = trial['weight_decay']
    wd_str = f", WD={wd}" if wd is not None else ""
    log_info(f"{'─'*50}")
    log_info(f"Trial {trial_idx + 1}/{len(trials)} — LR={lr}{wd_str}")
    log_info(f"{'─'*50}")

    # Reset model to initial weights
    model.load_state_dict(copy.deepcopy(initial_weights))

    # Reset ALL RNG state to the same base seed so each trial sees
    # identical data order/augmentations — the only variable is LR/WD
    base_seed = get_global_seed()
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(base_seed)
    for loader in data_loaders.values():
      if hasattr(loader, 'generator') and loader.generator is not None:
        loader.generator.manual_seed(base_seed)

    # Create fresh optimizer and scheduler with this LR (and WD if provided)
    optimizer = get_optimizer_with_lr(model, lr, weight_decay=wd)
    scheduler = get_scheduler(optimizer)

    trial_best_metric = 0.0
    trial_best_val_loss = float("inf")
    trial_best_weights = None
    trial_best_epoch = -1
    epochs_without_improvement = 0

    stage1_contrastive = contrastive_losses

    try:
      for epoch in range(epochs_stage1):
        train_loss, train_acc = train_one_epoch(
            data_loaders["Train"], device, optimizer, scheduler, model, criterion,
            epoch, epochs_stage1, stage1_contrastive, use_classification_loss=use_classification_loss
        )
        val_loss, val_acc = val_epoch(model, data_loaders["Val"], device, criterion, epoch, epochs_stage1, stage1_contrastive, use_classification_loss=use_classification_loss)

        # Evaluate WSOD metrics on Val set
        wsod_metric_value = 0.0
        if model_name:
          try:
            eval_model = model
            if "Dino" not in model_name:
              eval_model = add_attn_maps(model)

            wsod_results = evaluate_wsod(
                model=eval_model,
                model_name=model_name,
                device=device,
                split="Val",
                verbose=False,
            )
            wsod_metric_value = wsod_results.get(wsod_metric_key, 0.0)
          except Exception as e:
            log_info(f"WSOD evaluation skipped: {e}")

        scheduler.step()

        # Save epoch checkpoint if enabled
        if save_all_epochs and model_name:
          if not os.path.isdir(MODELS_PATH):
            os.makedirs(MODELS_PATH)
          epoch_checkpoint_path = os.path.join(MODELS_PATH, f"{model_name}_stage1_trial{trial_idx+1}_ep_{epoch + 1}.pth")
          save_model(model, path=epoch_checkpoint_path)

        # Check if metric improved
        if trial_best_weights is None:
          # First valid result — always accept
          is_better = True
        elif stage1_metric == "val_loss":
          is_better = val_loss < trial_best_val_loss
        else:
          is_better = wsod_metric_value > trial_best_metric

        if is_better:
          trial_best_metric = wsod_metric_value
          trial_best_val_loss = val_loss
          trial_best_weights = copy.deepcopy(model.state_dict())
          trial_best_epoch = epoch
          epochs_without_improvement = 0
          if stage1_metric == "val_loss":
            log_info(f"  ★ Trial {trial_idx+1} new best val_loss: {trial_best_val_loss:.4f}")
          else:
            log_info(f"  ★ Trial {trial_idx+1} new best {eval_metric}: {trial_best_metric:.4f}")
        else:
          epochs_without_improvement += 1

        # Early reset: no improvement for patience epochs
        if epochs_without_improvement >= patience_epochs:
          log_info(f"No improvement for {patience_epochs} epochs, stopping trial {trial_idx+1}")
          break

        # Absolute minimum threshold check after eval_after_epochs
        if epoch + 1 >= patience_epochs:
          if stage1_metric == "val_loss":
            below_threshold = trial_best_val_loss > (1.0 - min_threshold)
          else:
            below_threshold = trial_best_metric < min_threshold
          if below_threshold:
            log_info(f"Trial {trial_idx+1} below min_threshold ({min_threshold}) after {epoch+1} epochs, skipping")
            break

    except KeyboardInterrupt:
      log_info(f"Trial {trial_idx+1} interrupted by user")

    # Compare trial best with global best (first trial always accepted)
    if global_best_trial == -1:
      is_global_better = True
    elif stage1_metric == "val_loss":
      is_global_better = trial_best_val_loss < global_best_val_loss
    else:
      is_global_better = trial_best_metric > global_best_metric

    if is_global_better:
      global_best_metric = trial_best_metric
      global_best_val_loss = trial_best_val_loss
      global_best_weights = trial_best_weights if trial_best_weights is not None else copy.deepcopy(model.state_dict())
      global_best_trial = trial_idx
      global_best_epoch = trial_best_epoch
      if stage1_metric == "val_loss":
        log_info(f"  ★★ New global best! Trial {trial_idx+1}, val_loss: {global_best_val_loss:.4f}")
      else:
        log_info(f"  ★★ New global best! Trial {trial_idx+1}, {eval_metric}: {global_best_metric:.4f}")
    else:
      if stage1_metric == "val_loss":
        log_info(f"  Trial {trial_idx+1} best val_loss: {trial_best_val_loss:.4f} (global best: {global_best_val_loss:.4f})")
      else:
        log_info(f"  Trial {trial_idx+1} best: {trial_best_metric:.4f} (global best: {global_best_metric:.4f})")

  best_trial = trials[global_best_trial]
  best_wd_str = f", WD={best_trial['weight_decay']}" if best_trial['weight_decay'] is not None else ""
  log_info(f"{'='*60}")
  log_info(f"MULTI-START STAGE 1 COMPLETE")
  log_info(f"Best trial: {global_best_trial + 1}/{len(trials)} (LR={best_trial['lr']}{best_wd_str})")
  if stage1_metric == "val_loss":
    log_info(f"Best val_loss: {global_best_val_loss:.4f} at epoch {global_best_epoch + 1}")
  else:
    log_info(f"Best {eval_metric}: {global_best_metric:.4f} at epoch {global_best_epoch + 1}")
  log_info(f"{'='*60}")

  return global_best_weights


def train_model(data_loaders, device, optimizer, scheduler, model, criterion, epochs_stage1=1, epochs_stage2=1, contrastive_losses=None, patience=3, early_stop_delta=0.0001, model_name=None, start_epoch=0):
  """
  Train the model with early stopping and model checkpointing.

  In two-stage mode:
    - Stage 1 (Contrastive): trains for epochs_stage1 epochs, selects best model by WSOD metric (CorLoc/mAP)
    - Stage 2 (Classification): loads best Stage 1 checkpoint, trains for epochs_stage2 epochs

  Args:
      data_loaders: Dictionary with 'Train' and 'Val' data loaders
      device: Device to train on (cuda/mps/cpu)
      optimizer: Optimizer for training
      scheduler: Learning rate scheduler
      model: Model to train
      criterion: Loss function
      epochs_stage1: Number of epochs for Stage 1 (contrastive)
      epochs_stage2: Number of epochs for Stage 2 (classification)
      contrastive_losses: Dict with contrastive loss configs, or None if disabled
      patience: Number of epochs to wait for improvement before early stopping
      early_stop_delta: Minimum change in validation loss to qualify as improvement
      model_name: Name of the model for saving epoch checkpoints (optional)
      start_epoch: Epoch to start training from (for resuming)

  Returns:
      dict: Dictionary containing train_acc, val_acc, best_epoch, and best_val_loss
  """
  required_keys = ["Train", "Val"]
  if not all(key in data_loaders for key in required_keys):
    raise ValueError(f"data_loaders must contain keys: {required_keys}. Found: {list(data_loaders.keys())}")

  # Get training stages configuration
  training_config = get_training_stages(epochs_stage1, epochs_stage2)

  log_info(f"Device: {device}")
  optimizer_name = type(optimizer).__name__
  optimizer_lr = optimizer.defaults.get('lr', 'N/A')
  batch_size = get_config("defaults", "batch_size") or "N/A"
  log_info(f"Batch size: {batch_size}")
  log_info(f"Optimizer: {optimizer_name} (lr={optimizer_lr})")
  log_info(f"Scheduler: {type(scheduler).__name__}")
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  log_info(f"Model: {type(model).__name__}")
  log_info(f"Total params: {total_params:,}")
  log_info(f"Trainable params: {trainable_params:,}")
  log_info(f"Loss Function: {type(criterion).__name__}")

  # Check if backbone should be frozen in Stage 2
  freeze_backbone_stage2 = get_config("defaults", "freeze_backbone_stage2")
  if freeze_backbone_stage2 is None:
    freeze_backbone_stage2 = True  # Default to True for backward compatibility

  # Check if Stage 1 should be skipped
  skip_stage_one = get_config("defaults", "skip_stage_one") or False

  # Get metric for Stage 1 best model selection (mAP, CorLoc, or val_loss)
  stage1_metric = get_config("defaults", "stage1_metric") or "mAP"

  # Derive WSOD key from stage1_metric
  if stage1_metric == "CorLoc":
    eval_metric = "CorLoc"
    wsod_metric_key = "corloc"
  elif stage1_metric == "mAP":
    eval_metric = "mAP"
    wsod_metric_key = "map"
  else:
    eval_metric = get_active_dataset_config('eval_metric') or 'CorLoc'
    wsod_metric_key = 'map' if eval_metric == 'mAP' else 'corloc'

  # Stage 2 always selects best model by val_loss

  # Get Stage 2 method (cls = linear classifier, knn = k-nearest neighbors)
  stage2_method = get_config("defaults", "stage2_method") or "cls"

  # Log training plan
  log_info(f"{'='*60}")
  log_info("TRAINING CONFIGURATION")
  log_info(f"{'='*60}")
  log_info(f"Mode: {training_config['mode'].upper()}")

  # Check if CE loss should be used in Stage 1 (alongside contrastive)
  stage1_use_cls_loss = get_config("contrastive", "use_classification_loss") or False

  if training_config['mode'] == 'two-stage':
    if skip_stage_one:
      log_info(f"Stage 1: SKIPPED")
    else:
      multi_start_enabled = get_config("multi_start", "enabled") or False
      if stage1_use_cls_loss and contrastive_losses:
        cls_w = float(get_config("contrastive", "classification_weight") or 1.0)
        con_w = float(get_config("contrastive", "contrastive_weight") or 1.0)
        log_info(f"Stage 1 (CE + Contrastive): {epochs_stage1} epochs")
        log_info(f"  - Loss: CE (w={cls_w}) + {training_config['contrastive_type'].upper()} (w={con_w})")
        log_info(f"  - Target: Train backbone + classification head")
      elif stage1_use_cls_loss:
        log_info(f"Stage 1 (Classification): {epochs_stage1} epochs")
        log_info(f"  - Loss: CE")
        log_info(f"  - Target: Train backbone + classification head")
      else:
        log_info(f"Stage 1 (Contrastive): {epochs_stage1} epochs")
        log_info(f"  - Loss: {training_config['contrastive_type'].upper()}")
        log_info(f"  - Target: Train backbone embeddings")
      log_info(f"  - Best model selection: {stage1_metric} ({eval_metric} on Val)" if stage1_metric in ("mAP", "CorLoc") else f"  - Best model selection: {stage1_metric}")
      if multi_start_enabled:
        ms_trials = get_config("multi_start", "trials") or get_config("multi_start", "learning_rates") or []
        log_info(f"  - Multi-Start: ENABLED ({len(ms_trials)} trials)")
      else:
        log_info(f"  - Multi-Start: DISABLED")
      log_info(f"  - Early Stopping: DISABLED (runs all epochs)")
    if stage2_method == "knn":
      knn_k = get_config("knn", "k") or 20
      knn_temp = get_config("knn", "temperature") or 0.07
      log_info(f"Stage 2 (KNN): k={knn_k}, T={knn_temp}")
      log_info(f"  - Method: Extract features → KNN fit")
    else:
      log_info(f"Stage 2 (Classification): {epochs_stage2} epochs")
      log_info(f"  - Loss: Classification")
      log_info(f"  - Backbone: {'FROZEN' if freeze_backbone_stage2 else 'TRAINABLE'}")
      log_info(f"  - Best model selection: val_loss")
    if not skip_stage_one:
      log_info(f"  - Starts from: Best Stage 1 checkpoint (by {eval_metric})")
    _es_enabled = get_config("early_stopping", "enabled")
    if _es_enabled is None or _es_enabled:
      log_info(f"  - Early Stopping: ENABLED (Patience: {patience}, Delta: {early_stop_delta})")
    else:
      log_info(f"  - Early Stopping: DISABLED")
  else:
    log_info(f"Single-stage: Classification only")
    log_info(f"  - Loss: Classification")
    log_info(f"  - Epochs: 1-{epochs_stage2}")
    _es_enabled = get_config("early_stopping", "enabled")
    if _es_enabled is None or _es_enabled:
      log_info(f"  - Early Stopping: ENABLED (Patience: {patience}, Delta: {early_stop_delta})")
    else:
      log_info(f"  - Early Stopping: DISABLED")
  log_info(f"{'='*60}")

  # Check if we should save all epoch checkpoints
  save_all_epochs = get_config("checkpoint", "save_all_epochs") or False

  # ============================================================================
  # TWO-STAGE TRAINING
  # ============================================================================
  if training_config['mode'] == 'two-stage':

    # ------ STAGE 1: Contrastive Learning ------
    if skip_stage_one:
      log_info(f"{'='*60}")
      log_info("STAGE 1: SKIPPED (skip_stage_one=true)")
      log_info(f"{'='*60}")
    else:
      # Check if multi-start is enabled
      multi_start_enabled = get_config("multi_start", "enabled") or False

      if multi_start_enabled:
        # Multi-start: try multiple LRs, pick best
        best_stage1_weights = multi_start_stage1(
            model=model,
            data_loaders=data_loaders,
            device=device,
            criterion=criterion,
            contrastive_losses=contrastive_losses,
            epochs_stage1=epochs_stage1,
            model_name=model_name,
            eval_metric=eval_metric,
            wsod_metric_key=wsod_metric_key,
            save_all_epochs=save_all_epochs,
            use_classification_loss=stage1_use_cls_loss,
            stage1_metric=stage1_metric,
        )
        model.load_state_dict(best_stage1_weights)
      else:
        # Original single-run Stage 1
        log_info(f"{'='*60}")
        log_info("STAGE 1: CONTRASTIVE LEARNING")
        log_info(f"{'='*60}")

        best_wsod_metric = 0.0
        best_stage1_val_loss = float("inf")
        best_wsod_epoch = 0
        best_stage1_weights = copy.deepcopy(model.state_dict())

        stage1_contrastive = contrastive_losses

        # Stage 1: No early stopping - train all epochs and select best by WSOD metric
        try:
          for epoch in range(epochs_stage1):
            train_loss, train_acc = train_one_epoch(
                data_loaders["Train"], device, optimizer, scheduler, model, criterion,
                epoch, epochs_stage1, stage1_contrastive, use_classification_loss=stage1_use_cls_loss
            )
            val_loss, val_acc = val_epoch(model, data_loaders["Val"], device, criterion, epoch, epochs_stage1, stage1_contrastive, use_classification_loss=stage1_use_cls_loss)

            # Evaluate WSOD metrics (CorLoc, mAP) on Val set
            wsod_metric_value = 0.0
            if model_name:
              try:
                eval_model = model
                if "Dino" not in model_name:
                  eval_model = add_attn_maps(model)

                wsod_results = evaluate_wsod(
                    model=eval_model,
                    model_name=model_name,
                    device=device,
                    split="Val",
                    verbose=False,
                    )
                wsod_metric_value = wsod_results.get(wsod_metric_key, 0.0)
              except Exception as e:
                log_info(f"WSOD evaluation skipped: {e}")

            scheduler.step()

            # Save epoch checkpoint if enabled
            if save_all_epochs and model_name:
              if not os.path.isdir(MODELS_PATH):
                os.makedirs(MODELS_PATH)
              epoch_checkpoint_path = os.path.join(MODELS_PATH, f"{model_name}_stage1_ep_{epoch + 1}.pth")
              save_model(model, path=epoch_checkpoint_path)

            # Track best model by stage1_metric
            if stage1_metric == "val_loss":
              is_better = val_loss < best_stage1_val_loss
            else:
              is_better = wsod_metric_value > best_wsod_metric

            if is_better:
              best_wsod_metric = wsod_metric_value
              best_stage1_val_loss = val_loss
              best_wsod_epoch = epoch
              best_stage1_weights = copy.deepcopy(model.state_dict())
              if stage1_metric == "val_loss":
                log_info(f"  ★ New best val_loss: {best_stage1_val_loss:.4f}")
              else:
                log_info(f"  ★ New best {eval_metric}: {best_wsod_metric:.4f}")

        except KeyboardInterrupt:
          log_info("Stage 1 interrupted by user")

        log_info(f"{'='*60}")
        log_info(f"STAGE 1 COMPLETE")
        if stage1_metric == "val_loss":
          log_info(f"Best val_loss: {best_stage1_val_loss:.4f} at epoch {best_wsod_epoch + 1}")
        else:
          log_info(f"Best {eval_metric}: {best_wsod_metric:.4f} at epoch {best_wsod_epoch + 1}")
        log_info(f"{'='*60}")

        # Load best Stage 1 model for Stage 2
        model.load_state_dict(best_stage1_weights)

    # ------ STAGE 2 ------
    if stage2_method == "knn":
      # ====== STAGE 2: KNN ======
      log_info(f"{'='*60}")
      log_info("STAGE 2: KNN")
      if not skip_stage_one:
        log_info(f"Starting from best Stage 1 checkpoint")
      log_info(f"{'='*60}")

      knn_k = int(get_config("knn", "k") or 20)
      knn_temp = float(get_config("knn", "temperature") or 0.07)

      # Freeze backbone and extract features
      model.eval()
      for param in model.backbone.parameters():
        param.requires_grad = False

      log_info("Extracting train features...")
      all_features = []
      all_labels = []
      with torch.no_grad():
        for images, labels, *_ in tqdm(data_loaders["Train"], desc="KNN fit"):
          images = images.to(device)
          features = model.extract_cls_features(images)
          all_features.append(features.cpu())
          all_labels.append(labels)

      train_features = torch.cat(all_features, dim=0)
      train_labels = torch.cat(all_labels, dim=0)
      log_info(f"Feature bank: {train_features.shape[0]} samples, dim={train_features.shape[1]}")

      model.fit_knn(train_features.to(device), train_labels.to(device), k=knn_k, temperature=knn_temp)
      log_info(f"KNN fitted (k={knn_k}, T={knn_temp})")

      # Evaluate on Val
      val_loss, val_acc = val_epoch(model, data_loaders["Val"], device, criterion, 0, 1, contrastive_losses=None, use_classification_loss=True)

      # Evaluate WSOD
      if model_name:
        try:
          eval_model = model
          if "Dino" not in model_name:
            eval_model = add_attn_maps(model)

          wsod_results = evaluate_wsod(
              model=eval_model,
              model_name=model_name,
              device=device,
              split="Val",
              verbose=False,
          )
          wsod_value = wsod_results.get(wsod_metric_key, 0.0)
          log_info(f"[Val] {eval_metric}: {wsod_value:.4f} | Acc: {val_acc:.4f}")
        except Exception as e:
          log_info(f"WSOD evaluation skipped: {e}")

      best_train_acc = 0.0
      best_val_acc = val_acc
      best_val_loss = val_loss
      best_epoch = 0
      best_model_weights = copy.deepcopy(model.state_dict())

    else:
      # ====== STAGE 2: Classification ======
      log_info(f"{'='*60}")
      log_info("STAGE 2: CLASSIFICATION")
      if not skip_stage_one:
        log_info(f"Starting from best Stage 1 checkpoint")
      log_info(f"{'='*60}")

      # Freeze backbone if configured
      if freeze_backbone_stage2:
        log_info("Freezing backbone...")
        for param in model.backbone.parameters():
          param.requires_grad = False
        model.backbone.eval()

        # Recreate optimizer with only head parameters
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optimizer_for_params(head_params)
        scheduler = get_scheduler(optimizer, constant=True)

        log_info(f"Trainable parameters: {sum(p.numel() for p in head_params)}")
      else:
        log_info("Keeping backbone trainable (full model fine-tuning)...")
        # Reset optimizer and scheduler for stage 2
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, constant=True)

      # Reset early stopping for stage 2 (always enabled — early stopping is strictly Stage 2 only)
      _es_enabled = get_config("early_stopping", "enabled")
      if _es_enabled is None:
          _es_enabled = True
      early_stopping = EarlyStopping(patience=patience, delta=early_stop_delta, verbose=True) if _es_enabled else None

      best_val_loss = float("inf")
      best_model_weights = copy.deepcopy(model.state_dict())
      best_train_acc = 0.0
      best_val_acc = 0.0
      best_epoch = 0

      try:
        for epoch in range(epochs_stage2):
          # Keep backbone in eval() mode during stage 2 (only if frozen)
          if freeze_backbone_stage2:
            model.backbone.eval()

          train_loss, train_acc = train_one_epoch(
              data_loaders["Train"], device, optimizer, scheduler, model, criterion,
              epoch, epochs_stage2, contrastive_losses=None, use_classification_loss=True
          )
          val_loss, val_acc = val_epoch(model, data_loaders["Val"], device, criterion, epoch, epochs_stage2, contrastive_losses=None, use_classification_loss=True)

          scheduler.step()

          # Save epoch checkpoint if enabled
          if save_all_epochs and model_name:
            if not os.path.isdir(MODELS_PATH):
              os.makedirs(MODELS_PATH)
            epoch_checkpoint_path = os.path.join(MODELS_PATH, f"{model_name}_stage2_ep_{epoch + 1}.pth")
            save_model(model, path=epoch_checkpoint_path)

          if early_stopping is not None:
            early_stopping.check_early_stop(val_loss)

          # Track best model by val_loss
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = epoch

          if early_stopping is not None and early_stopping.stop_training:
            log_info(f"Early stopping at epoch {epoch + 1}")
            break

      except KeyboardInterrupt:
        log_info("Stage 2 interrupted by user")

      log_info(f"Best Stage 2 model from epoch {best_epoch + 1} with val loss: {best_val_loss:.4f}")
      model.load_state_dict(best_model_weights)

  # ============================================================================
  # SINGLE-STAGE TRAINING (Classification only)
  # ============================================================================
  else:
    log_info("Starting training...")

    _es_enabled = get_config("early_stopping", "enabled")
    if _es_enabled is None:
        _es_enabled = True
    early_stopping = EarlyStopping(patience=patience, delta=early_stop_delta, verbose=True) if _es_enabled else None

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_epoch = start_epoch

    try:
      for epoch in range(start_epoch, epochs_stage2):
        train_loss, train_acc = train_one_epoch(
            data_loaders["Train"], device, optimizer, scheduler, model, criterion,
            epoch, epochs_stage2, contrastive_losses, use_classification_loss=True
        )
        val_loss, val_acc = val_epoch(model, data_loaders["Val"], device, criterion, epoch, epochs_stage2, contrastive_losses, use_classification_loss=True)

        # Evaluate WSOD metrics on Val set
        if model_name:
          try:
            eval_model = model
            if "Dino" not in model_name:
              eval_model = add_attn_maps(model)

            wsod_results = evaluate_wsod(
                model=eval_model,
                model_name=model_name,
                device=device,
                split="Val",
                verbose=False,
            )
          except Exception as e:
            log_info(f"WSOD evaluation skipped: {e}")

        scheduler.step()

        # Save epoch checkpoint if enabled
        if save_all_epochs and model_name:
          if not os.path.isdir(MODELS_PATH):
            os.makedirs(MODELS_PATH)
          epoch_checkpoint_path = os.path.join(MODELS_PATH, f"{model_name}_ep_{epoch + 1}.pth")
          save_model(model, path=epoch_checkpoint_path)

        if early_stopping is not None:
          early_stopping.check_early_stop(val_loss)

        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model_weights = copy.deepcopy(model.state_dict())
          best_train_acc = train_acc
          best_val_acc = val_acc
          best_epoch = epoch

        if early_stopping is not None and early_stopping.stop_training:
          log_info(f"Early stopping at epoch {epoch + 1}")
          break

    except KeyboardInterrupt:
      log_info("Training interrupted by user")

    log_info(f"Best model from epoch {best_epoch + 1} with validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_weights)

  return {
    'train_acc': best_train_acc,
    'val_acc': best_val_acc,
    'best_epoch': best_epoch + 1,
    'best_val_loss': best_val_loss
  }


def save_model(model, path="models/model.pth"):
  """
  Save model weights to disk.

  Args:
      model: Model to save
      path: Path where to save the model weights
  """
  torch.save(model.state_dict(), os.path.abspath(path))
  log_info(f"Model saved in: {path}")


def find_latest_checkpoint(model_name: str) -> int:
  """
  Find the latest epoch checkpoint for a given model.

  Args:
      model_name: Name of the model

  Returns:
      int: Latest epoch number, or None if no checkpoint found
  """
  pattern = os.path.join(MODELS_PATH, f"{model_name}_ep_*.pth")
  checkpoints = glob_module.glob(pattern)

  if not checkpoints:
    return None

  # Extract epoch numbers and find the maximum
  max_epoch = 0
  for ckpt in checkpoints:
    match = re.search(rf"{model_name}_ep_(\d+)\.pth", ckpt)
    if match:
      epoch = int(match.group(1))
      max_epoch = max(max_epoch, epoch)

  return max_epoch if max_epoch > 0 else None


def train(
    batch_size: int,
    num_workers: int,
    model,
    device,
    model_name: str,
    num_classes: int,
    resume_epoch: bool = False
):
  """
  Train the model with specified parameters.

  Args:
      batch_size: Batch size for training
      num_workers: Number of workers for data loading
      model: Model to train
      device: Device (cuda/mps/cpu)
      model_name: Name for saving the model
      num_classes: Number of classes in dataset
      resume_epoch: If True, auto-detect and resume from latest checkpoint

  Returns:
      tuple: (train_accuracy, val_accuracy)
  """
  model_cfg = get_config("models", model_name) or {}
  train_size = model_cfg.get("train_size", 224)
  eval_size = model_cfg.get("img_size", train_size)

  # Auto-detect latest checkpoint if resuming
  start_epoch = 0
  if resume_epoch:
    latest_epoch = find_latest_checkpoint(model_name)
    if latest_epoch is not None:
      checkpoint_path = os.path.join(MODELS_PATH, f"{model_name}_ep_{latest_epoch}.pth")
      log_info(f"Found latest checkpoint: {checkpoint_path}")
      model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
      log_info(f"Loaded weights from epoch {latest_epoch}")
      start_epoch = latest_epoch
    else:
      log_info(f"No checkpoint found for {model_name}")
      log_info(f"Starting training from scratch")

  root_path = get_root_path()
  log_info(f"Using data path: {root_path}")

  data_loaders, datasets = get_data_loaders(
      root_path,
      batch_size=batch_size,
      num_workers=num_workers,
      img_size=train_size,
      eval_size=eval_size,
      splits=["Train", "Val"]
  )

  criterion = get_loss()
  optimizer = get_optimizer(model)
  scheduler = get_scheduler(optimizer)

  contrastive_losses = setup_contrastive_learning()

  patience = int(get_config("early_stopping", "patience") or 8)
  delta = float(get_config("early_stopping", "delta") or 0.0001)

  epochs_stage1 = int(get_config("defaults", "epochs_stage1") or 6)
  epochs_stage2 = int(get_config("defaults", "epochs_stage2") or 6)

  results = train_model(
      data_loaders,
      device,
      optimizer,
      scheduler,
      model,
      criterion,
      epochs_stage1=epochs_stage1,
      epochs_stage2=epochs_stage2,
      contrastive_losses=contrastive_losses,
      patience=patience,
      early_stop_delta=delta,
      model_name=model_name,
      start_epoch=start_epoch
  )

  if not os.path.isdir(MODELS_PATH):
    os.makedirs(MODELS_PATH)

  model_path = os.path.join(MODELS_PATH, f"{model_name}.pth")
  save_model(model, path=model_path)

  log_info(f"Training completed with results:")
  log_info(f"  Train Accuracy: {results['train_acc']:.4f}")
  log_info(f"  Val Accuracy: {results['val_acc']:.4f}")
  log_info(f"  Best Epoch: {results['best_epoch']}")
  log_info(f"  Best Val Loss: {results['best_val_loss']:.4f}")

  return results['train_acc'], results['val_acc']
