"""DinoV2 Detector for Weakly Supervised Object Detection (WSOD).

Uses attention maps from the DINOv2 backbone to generate bounding boxes
without requiring instance-level annotations during training.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple

from simpleml.registries import MODELS

from .dino_classifier import DinoClassifier, DinoConfigFactory, DinoModelSize, PretrainType
from .detection_utils import BBox, generate_bounding_box, get_last_layer_attention
from .multi_start import run_multi_start


@MODELS.register
class DinoDetector(nn.Module):
    """DINOv2-based weakly supervised object detector.

    Uses the CLS token's attention over spatial patches as a localization
    signal (WSOD). During training, a classification head provides image-level
    supervision. At inference, call detect() to obtain bounding boxes.

    Attributes:
        backbone: DINOv2 vision transformer backbone.
        classifier: Linear layer for image-level classification.
        patch_size: Patch size of the backbone, used for attention reshaping.
    """

    MODEL_MAPPING = {
        "DinoV2_Small": (DinoModelSize.SMALL, PretrainType.IMAGENET),
        "DinoV2_Base": (DinoModelSize.BASE, PretrainType.IMAGENET),
        "DinoV2_Large": (DinoModelSize.LARGE, PretrainType.IMAGENET),
        "DinoV2RS_Small": (DinoModelSize.SMALL, PretrainType.REMOTE_SENSING),
        "DinoV2RS_Base": (DinoModelSize.BASE, PretrainType.REMOTE_SENSING),
        "DinoV2RS_Large": (DinoModelSize.LARGE, PretrainType.REMOTE_SENSING),
    }

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        proj_dim: int = 128,
        greedy_merge_enabled: bool = True,
        greedy_iou_threshold: float = 0.1,
        greedy_distance_threshold: float = 50.0,
    ):
        """Initialize DinoDetector.

        Args:
            model_name: Backbone identifier (e.g., "DinoV2RS_Base").
            num_classes: Number of image-level classes for the classification head.
            pretrained: Whether to load pretrained backbone weights.
            proj_dim: Output dimension of the contrastive projection head (MLP).
                Used only in Stage 1. Frozen and unused in Stage 2.
            greedy_merge_enabled: Whether to merge nearby boxes after detection.
            greedy_iou_threshold: Minimum IoU to trigger a merge.
            greedy_distance_threshold: Maximum edge-to-edge distance in pixels
                to trigger a merge.

        Raises:
            ValueError: If model_name is not recognized.
        """
        super().__init__()

        mapping = self.MODEL_MAPPING.get(model_name)
        if mapping is None:
            raise ValueError(
                f"Unknown model name: {model_name}. Valid names: {list(self.MODEL_MAPPING)}"
            )

        model_size, pretrain_type = mapping
        self.backbone = DinoClassifier._create_backbone(model_size, pretrain_type, pretrained)

        if model_size != DinoModelSize.LARGE:
            config = DinoConfigFactory.get_config(model_size, pretrain_type)
            self.patch_size: int = config.patch_size
        else:
            self.patch_size = 14

        feature_dim = self.backbone.norm.normalized_shape[0]
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, proj_dim),
        )

        self.greedy_merge_enabled = greedy_merge_enabled
        self.greedy_iou_threshold = greedy_iou_threshold
        self.greedy_distance_threshold = greedy_distance_threshold

        # 1 = contrastive (Stage 1), 2 = classification (Stage 2)
        self.stage: int = 2

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, projected_embeddings)`` for contrastive training.

        Runs the backbone once and branches into two parallel heads:
        the classifier (for CE loss) and the projection head (for SupCon).

        Args:
            x: Input tensor ``[B, C, H, W]``.

        Returns:
            Tuple of classification logits ``[B, num_classes]`` and projection
            embeddings ``[B, proj_dim]`` (not yet L2-normalized).
        """
        features_dict = self.backbone.forward_features(x)
        cls_features = features_dict["x_norm_clstoken"]
        return self.classifier(cls_features), self.projection_head(cls_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning image-level classification logits.

        Used during training with image-level supervision (WSOD).

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Classification logits [B, num_classes].
        """
        features_dict = self.backbone.forward_features(x)
        cls_features = features_dict["x_norm_clstoken"]
        return self.classifier(cls_features)

    def training_step(
        self,
        batch: list,
        loss_fns: list[nn.Module],
    ) -> torch.Tensor:
        """Compute training loss, dispatching by ``self.stage``.

        - **Stage 1** (contrastive): Extracts L2-normalized CLS embeddings and
          applies every loss in ``loss_fns[1:]``. Falls back to ``loss_fns[0]``
          when no contrastive losses are provided.
        - **Stage 2** (classification): Standard forward pass with ``loss_fns[0]``.

        Args:
            batch: ``[inputs [B, C, H, W], list[target_dict]]`` where each
                target dict contains an ``image_label`` key.
            loss_fns: Loss functions from the Trainer.  ``loss_fns[0]`` must be
                the classification loss; ``loss_fns[1:]`` are contrastive losses.

        Returns:
            Scalar loss tensor.
        """
        device = next(self.parameters()).device
        inputs = batch[0].to(device)
        raw = batch[1]
        if isinstance(raw, torch.Tensor):
            targets = raw.to(dtype=torch.long, device=device)
        else:
            targets = torch.tensor(
                [t['image_label'] for t in raw], dtype=torch.long, device=device
            )

        if self.stage == 1:
            logits, embeddings = self.forward_with_embedding(inputs)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            contrastive_fns = loss_fns[1:]
            if not contrastive_fns:
                return loss_fns[0](logits, targets)
            loss = contrastive_fns[0](embeddings, targets)
            for fn in contrastive_fns[1:]:
                loss = loss + fn(embeddings, targets)
            return loss

        return loss_fns[0](self(inputs), targets)

    def fit_loop(self, trainer: Any) -> dict[str, Any]:
        """Two-stage training loop: contrastive pre-training then classification.

        The Trainer delegates here when this method is present, so all
        orchestration lives in the model and the Trainer stays generic.

        Stage 1 trains with ``self.stage = 1`` (contrastive losses). The best
        checkpoint is selected by ``training.best_metric`` on the validation set.
        Stage 2 trains with ``self.stage = 2`` (classification), optionally
        freezing the backbone first.

        Expected training config keys (all optional):
            stage1_epochs (int): Epochs for Stage 1. Default 5.
            stage2_epochs (int): Epochs for Stage 2. Default 10.
            freeze_backbone_stage2 (bool): Freeze backbone in Stage 2. Default True.

        Args:
            trainer: The Trainer instance (provides loaders, optimizer, etc.).

        Returns:
            Dict compatible with ``Trainer.fit`` return value.
        """
        from simpleml.logger import log_info

        cfg = trainer._cfg
        stage1_epochs = cfg.get("stage1_epochs", 5)
        stage2_epochs = cfg.get("stage2_epochs", 10)
        freeze_backbone = cfg.get("freeze_backbone_stage2", True)
        best_metric_key = cfg.get("stage1_best_metric", cfg.get("best_metric"))

        # ------------------------------------------------------------------ #
        # STAGE 1: Contrastive pre-training                                   #
        # ------------------------------------------------------------------ #
        log_info(f"{'='*60}")
        log_info(f"STAGE 1 (Contrastive): {stage1_epochs} epochs")
        log_info(f"{'='*60}")

        self.stage = 1
        best_stage1_weights = copy.deepcopy(self.state_dict())
        best_stage1_metric = -math.inf

        ms_result = run_multi_start(self, trainer, stage1_epochs, best_metric_key)
        if ms_result is not None:
            best_stage1_weights = ms_result
        else:
            for epoch in range(stage1_epochs):
                log_info(f"Epoch {epoch + 1}/{stage1_epochs}")
                train_loss = trainer._train_one_epoch(epoch)
                trainer._log_metrics({"loss": train_loss}, epoch, prefix="stage1/train")

                if trainer.scheduler and cfg["scheduler_step_on"] == "epoch":
                    trainer.scheduler.step()

                if trainer.val_loader is not None:
                    val_result = trainer._validate_one_epoch(epoch)
                    val_loss = val_result["loss"]
                    val_metrics = val_result.get("metrics", {})
                    trainer._log_metrics({"loss": val_loss}, epoch, prefix="stage1/val")
                    trainer._log_metrics(val_metrics, epoch, prefix="stage1/val")

                    summary = {
                        "train_loss": f"{train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                    }
                    summary.update({k: f"{v:.4f}" for k, v in val_metrics.items()})
                    log_info("  ".join(f"{k}: {v}" for k, v in summary.items()))

                    if best_metric_key and best_metric_key in val_metrics:
                        value = val_metrics[best_metric_key]
                        if value > best_stage1_metric:
                            best_stage1_metric = value
                            best_stage1_weights = copy.deepcopy(self.state_dict())
                            log_info(f"  ★ New best {best_metric_key}: {best_stage1_metric:.4f}")
                else:
                    log_info(f"train_loss: {train_loss:.4f}")

        log_info(f"Stage 1 complete.")
        if best_metric_key:
            log_info(f"Best {best_metric_key}: {best_stage1_metric:.4f}")
        self.load_state_dict(best_stage1_weights)

        # ------------------------------------------------------------------ #
        # STAGE 2: Classification fine-tuning                                 #
        # ------------------------------------------------------------------ #
        log_info(f"{'='*60}")
        log_info(f"STAGE 2 (Classification): {stage2_epochs} epochs")
        if freeze_backbone:
            log_info("  Backbone: FROZEN")
        log_info(f"{'='*60}")

        self.stage = 2

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.projection_head.parameters():
                param.requires_grad = False
            trainable = [p for p in self.parameters() if p.requires_grad]
            import inspect
            opt_cls = type(trainer.optimizer)
            valid_keys = set(inspect.signature(opt_cls.__init__).parameters) - {"self", "params"}
            opt_kwargs = {k: v for k, v in trainer.optimizer.defaults.items() if k in valid_keys}
            trainer.optimizer = opt_cls(trainable, **opt_kwargs)
            log_info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

        # Stage 2 best-model metric: val_loss when stage2_best_metric="val_loss" or absent
        stage2_metric = cfg.get("stage2_best_metric", "val_loss")
        trainer._cfg["best_metric"] = None if stage2_metric == "val_loss" else stage2_metric
        trainer.best_val_loss = math.inf
        trainer.best_metric_value = (
            -math.inf if cfg.get("best_metric_mode", "max") == "max" else math.inf
        )

        last_train_loss = math.nan
        last_val_loss = None
        last_metrics: dict[str, float] = {}

        for epoch in range(stage2_epochs):
            log_info(f"Epoch {epoch + 1}/{stage2_epochs}")
            last_train_loss = trainer._train_one_epoch(epoch)
            trainer._log_metrics({"loss": last_train_loss}, epoch, prefix="stage2/train")
            trainer._log_metrics(
                {"lr": trainer.optimizer.param_groups[0]["lr"]},
                epoch,
                prefix="stage2/train",
            )

            if trainer.scheduler and cfg["scheduler_step_on"] == "epoch":
                trainer.scheduler.step()

            should_validate = (
                trainer.val_loader is not None
                and (epoch + 1) % cfg["val_every"] == 0
            )
            if should_validate:
                val_result = trainer._validate_one_epoch(epoch)
                last_val_loss = val_result["loss"]
                last_metrics = val_result.get("metrics", {})
                trainer._log_metrics({"loss": last_val_loss}, epoch, prefix="stage2/val")
                trainer._log_metrics(last_metrics, epoch, prefix="stage2/val")
                trainer._maybe_save_checkpoint(epoch, last_val_loss, last_metrics)
            else:
                last_metrics = {}
                trainer._maybe_save_checkpoint(epoch, None, {})

            summary: dict[str, Any] = {"train_loss": f"{last_train_loss:.4f}"}
            if last_val_loss is not None:
                summary["val_loss"] = f"{last_val_loss:.4f}"
            summary.update({k: f"{v:.4f}" for k, v in last_metrics.items()})
            log_info("  ".join(f"{k}: {v}" for k, v in summary.items()))

        return {
            "last_train_loss": last_train_loss,
            "last_val_loss": last_val_loss,
            "best_val_loss": (
                trainer.best_val_loss if trainer.best_val_loss < math.inf else None
            ),
            "last_metrics": last_metrics,
            "epochs_trained": stage1_epochs + stage2_epochs,
        }

    def detect(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        min_area: int = 200,
        greedy_merge: Optional[bool] = None,
        iou_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        dynamic_threshold: bool = True,
        threshold_method: str = "otsu",
    ) -> List[List[BBox]]:
        """Run WSOD detection on a batch using attention maps.

        Extracts the last-layer CLS attention for each image in the batch,
        then generates bounding boxes via connected components and optional
        greedy merging.

        Args:
            x: Input tensor [B, C, H, W].
            threshold: Binarization threshold. None = computed dynamically.
            min_area: Minimum box area in pixels.
            greedy_merge: Whether to merge nearby boxes.
            iou_threshold: IoU threshold for greedy merge.
            distance_threshold: Edge-distance threshold in pixels for greedy merge.
            dynamic_threshold: Use dynamic threshold when threshold is None.
            threshold_method: Dynamic threshold method
                ("otsu", "percentile", "adaptive", "kmeans").

        Returns:
            List of length B; each element is a list of
            (x1, y1, x2, y2, confidence) tuples.
        """
        _greedy_merge = self.greedy_merge_enabled if greedy_merge is None else greedy_merge
        _iou = self.greedy_iou_threshold if iou_threshold is None else iou_threshold
        _dist = self.greedy_distance_threshold if distance_threshold is None else distance_threshold

        results: List[List[BBox]] = []
        for i in range(x.shape[0]):
            single = x[i].unsqueeze(0)
            attn_map = get_last_layer_attention(self.backbone, single, self.patch_size)
            bboxes = generate_bounding_box(
                attn_map,
                threshold=threshold,
                min_area=min_area,
                greedy_merge=_greedy_merge,
                iou_threshold=_iou,
                distance_threshold=_dist,
                dynamic_threshold=dynamic_threshold,
                threshold_method=threshold_method,
            )
            results.append(bboxes)
        return results
