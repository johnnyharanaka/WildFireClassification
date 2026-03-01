"""
DinoV2 Classifier implementation with different backbone configurations.
Supports both ImageNet and Remote Sensing pretrained models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from simpleml.registries import MODELS

from ..models.vision_transformer import vit_base, vit_large, vit_small
from .LoRA import LoRALayer, apply_lora_to_vit, init_lora_weights

_MODELS_DIR = Path(__file__).parent / "pretrained"

class DinoModelSize(Enum):
    """Enum for DinoV2 model sizes."""

    SMALL = "DinoV2_Small"
    BASE = "DinoV2_Base"
    LARGE = "DinoV2_Large"


class PretrainType(Enum):
    """Enum for pretrain types."""

    IMAGENET = "IMAGENET"
    REMOTE_SENSING = "RS"


@dataclass(frozen=True)
class DinoConfig:
    """Configuration for DinoV2 backbone.

    Attributes:
        patch_size: Size of image patches.
        img_size: Input image size.
        block_chunks: Number of block chunks for processing.
        init_values: Initial values for layer scaling.
        num_register_tokens: Number of register tokens.
        interpolate_antialias: Whether to use antialiasing during interpolation.
        interpolate_offset: Offset for interpolation.
        checkpoint_path: Path to pretrained weights.
    """

    patch_size: int
    img_size: int
    block_chunks: int
    init_values: Optional[float]
    num_register_tokens: int
    interpolate_antialias: bool
    interpolate_offset: float
    checkpoint_path: str


class DinoConfigFactory:
    """Factory for creating DinoV2 configurations."""

    CONFIGS = {
        (DinoModelSize.SMALL, PretrainType.REMOTE_SENSING): DinoConfig(
            patch_size=16,
            img_size=224,
            block_chunks=0,
            init_values=None,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            checkpoint_path=str(_MODELS_DIR / "dinov2_vits14_reg4_pretrain_rs.pth"),
        ),
        (DinoModelSize.SMALL, PretrainType.IMAGENET): DinoConfig(
            patch_size=14,
            img_size=518,
            block_chunks=0,
            init_values=1.0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            checkpoint_path=str(_MODELS_DIR / "dinov2_vits14_reg4_pretrain.pth"),
        ),
        (DinoModelSize.BASE, PretrainType.REMOTE_SENSING): DinoConfig(
            patch_size=16,
            img_size=224,
            block_chunks=0,
            init_values=None,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            checkpoint_path=str(_MODELS_DIR / "dinov2_vitb14_reg4_pretrain_rs.pth"),
        ),
        (DinoModelSize.BASE, PretrainType.IMAGENET): DinoConfig(
            patch_size=14,
            img_size=518,
            block_chunks=0,
            init_values=1.0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            checkpoint_path=str(_MODELS_DIR / "dinov2_vitb14_reg4_pretrain.pth"),
        ),
    }

    @classmethod
    def get_config(
        cls, model_size: DinoModelSize, pretrain_type: PretrainType
    ) -> DinoConfig:
        """Get configuration for a specific model size and pretrain type.

        Args:
            model_size: Size of the DinoV2 model.
            pretrain_type: Type of pretraining (ImageNet or Remote Sensing).

        Returns:
            DinoConfig instance for the specified configuration.

        Raises:
            ValueError: If configuration not found for the given parameters.
        """
        config = cls.CONFIGS.get((model_size, pretrain_type))
        if config is None:
            raise ValueError(
                f"No configuration found for {model_size.value} with {pretrain_type.value} pretraining"
            )
        return config


@MODELS.register
class DinoClassifier(nn.Module):
    """DinoV2-based classifier with support for different model sizes and pretrain types.

    This classifier uses a pretrained DinoV2 backbone with a linear classification head
    and a projection head for generating embeddings.

    Attributes:
        backbone: The DinoV2 vision transformer backbone.
        classifier: Linear layer for classification.
        projection_head: MLP for generating embeddings.
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
        num_classes: int = 1000,
        pretrained: bool = True,
        proj_dim: int = 128,
    ):
        """Initialize DinoClassifier.

        Args:
            model_name: Name of the model, encodes size and pretrain type
                (e.g., "DinoV2_Small" for ImageNet, "DinoV2RS_Small" for Remote Sensing).
            num_classes: Number of output classes for classification.
            pretrained: Whether to load pretrained weights.
            proj_dim: Dimension of projection head output.

        Raises:
            ValueError: If model_name is invalid.
        """
        super().__init__()

        mapping = self.MODEL_MAPPING.get(model_name)
        if mapping is None:
            raise ValueError(f"Unknown model name: {model_name}. Valid names: {list(self.MODEL_MAPPING)}")

        model_size, pretrain_type = mapping
        self.backbone = self._create_backbone(model_size, pretrain_type, pretrained)

        feature_dim = self.backbone.norm.normalized_shape[0]

        self.classifier = nn.Linear(feature_dim, num_classes)

        # Projection head for contrastive learning (SimCLR/SupCon style)
        # Reduces dimensionality: feature_dim -> feature_dim//2 -> proj_dim
        # Use fixed seed for consistent initialization across runs
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, proj_dim),
        )
        torch.set_rng_state(rng_state)

    @staticmethod
    def _create_backbone(
        model_size: DinoModelSize, pretrain_type: PretrainType, pretrained: bool
    ) -> nn.Module:
        """Create DinoV2 backbone and optionally load pretrained weights.

        Args:
            model_size: Size of the model (SMALL, BASE, or LARGE).
            pretrain_type: Type of pretraining, used to select the checkpoint.
            pretrained: Whether to load pretrained weights.

        Returns:
            Backbone model, with pretrained weights if pretrained=True.

        Raises:
            ValueError: If model size is unsupported.
        """
        if model_size == DinoModelSize.LARGE:
            backbone = vit_large(num_register_tokens=4)
            if pretrained:
                checkpoint_path = str(_MODELS_DIR / "dinov2_vitl14_reg4_pretrain_rs.pth")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                backbone.load_state_dict(checkpoint)
            return backbone

        config = DinoConfigFactory.get_config(model_size, pretrain_type)

        if model_size == DinoModelSize.SMALL:
            backbone = vit_small(
                patch_size=config.patch_size,
                img_size=config.img_size,
                block_chunks=config.block_chunks,
                init_values=config.init_values,
                num_register_tokens=config.num_register_tokens,
                interpolate_antialias=config.interpolate_antialias,
                interpolate_offset=config.interpolate_offset,
            )
        elif model_size == DinoModelSize.BASE:
            backbone = vit_base(
                patch_size=config.patch_size,
                img_size=config.img_size,
                block_chunks=config.block_chunks,
                init_values=config.init_values,
                num_register_tokens=config.num_register_tokens,
                interpolate_antialias=config.interpolate_antialias,
                interpolate_offset=config.interpolate_offset,
            )
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

        if pretrained:
            checkpoint = torch.load(
                config.checkpoint_path, map_location="cpu", weights_only=False
            )
            backbone.load_state_dict(checkpoint, strict=False)

        return backbone

    def get_last_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Get last self-attention from backbone.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Self-attention tensor from the last transformer layer.
        """
        return self.backbone.get_last_self_attention(x)

    def get_all_self_attentions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get self-attention from all transformer layers.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of attention tensors, one per layer.
            Each tensor has shape (B, num_heads, seq_len, seq_len).
        """
        return self.backbone.get_all_self_attentions(x)

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the classifier.

        Args:
            x: Input tensor of shape (B, C, H, W).
            return_embedding: Whether to return projection embeddings.

        Returns:
            Always returns logits as first element.
            If return_embedding is True, returns embeddings as second element.

            Possible return formats:
            - logits only: (logits,) or just logits
            - logits + embedding: (logits, embeddings)
        """
        features_dict = self.backbone.forward_features(x)
        cls_features = features_dict["x_norm_clstoken"]

        logits = self.classifier(cls_features)

        # Build return tuple based on requested outputs
        results = [logits]

        # Add embeddings if requested
        if return_embedding:
            projection = self.projection_head(cls_features)
            results.append(projection)

        # Return single value or tuple
        return tuple(results) if len(results) > 1 else results[0]


@MODELS.register
class LoRADinoClassifier(DinoClassifier):
    """DinoV2 classifier with LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    Extends DinoClassifier by applying LoRA to the attention layers of the backbone.
    Only LoRA parameters, classifier, and projection head are trainable.

    Attributes:
        lora_r: Rank of LoRA decomposition.
        lora_alpha: Scaling factor for LoRA.
        w_As: List of LoRA down projection layers.
        w_Bs: List of LoRA up projection layers.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        proj_dim: int = 128,
        lora_r: int = 4,
        lora_alpha: int = 4,
        lora_layers: Optional[List[int]] = None,
    ):
        """Initialize LoRADinoClassifier.

        Args:
            model_name: Name of the model, encodes size and pretrain type
                (e.g., "DinoV2RS_Small" for Remote Sensing small backbone).
            num_classes: Number of output classes for classification.
            pretrained: Whether to load pretrained weights.
            proj_dim: Dimension of projection head output.
            lora_r: Rank of the LoRA decomposition.
            lora_alpha: Scaling factor for LoRA updates.
            lora_layers: List of layer indices to apply LoRA. None applies to all layers.
        """
        super().__init__(model_name, num_classes, pretrained, proj_dim)

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # Apply LoRA to backbone
        self.w_As, self.w_Bs = apply_lora_to_vit(
            self.backbone, lora_r, lora_alpha, lora_layers
        )

        # Initialize LoRA weights
        init_lora_weights(self.w_As, self.w_Bs)

        # Re-initialize classifier
        nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        print(f"LoRA applied with r={lora_r}, alpha={lora_alpha}")
        print(f"Trainable parameters: {self.get_num_trainable_params():,}")

    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters.

        Returns:
            Number of parameters with requires_grad=True.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_lora_weights(self, filepath: str) -> None:
        """Save LoRA and head weights to file.

        Args:
            filepath: Path to save the weights.
        """
        state = {
            "w_As": [w.state_dict() for w in self.w_As],
            "w_Bs": [w.state_dict() for w in self.w_Bs],
            "classifier": self.classifier.state_dict(),
            "projection_head": self.projection_head.state_dict(),
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
        }
        torch.save(state, filepath)

    def load_lora_weights(self, filepath: str) -> None:
        """Load LoRA and head weights from file.

        Args:
            filepath: Path to load the weights from.
        """
        state = torch.load(filepath, map_location="cpu")
        for w, w_state in zip(self.w_As, state["w_As"]):
            w.load_state_dict(w_state)
        for w, w_state in zip(self.w_Bs, state["w_Bs"]):
            w.load_state_dict(w_state)
        self.classifier.load_state_dict(state["classifier"])
        self.projection_head.load_state_dict(state["projection_head"])
