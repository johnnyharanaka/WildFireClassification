"""Module for model creation and management.

Provides a unified interface for creating different model architectures.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from timm.models import create_model as timm_create_model
from core.config.config import log_info
from .dino_classifier import DinoClassifier, LoRADinoClassifier
from .custom_models import FNClassifier
from third_party.TSCAM.lib.config.default import cfg_from_file, config

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ModelParams:
    """Parameters for model creation.

    Attributes:
        base_model: Base model name or identifier.
        params: Model-specific parameters dictionary.
        model_class: Model class identifier for factory creation.
        needs_attention_maps: Whether model requires attention map extraction.
        target_layer: Callable to get target layer for visualization.
    """

    base_model: str
    params: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"pretrained": True, "num_classes": 2}
    )
    model_class: Optional[str] = "timm"
    needs_attention_maps: Optional[bool] = False
    target_layer: Optional[Callable] = None


class Models(Enum):
    """Enum of available models with their configurations."""

    DinoV2_Small = ModelParams(
        model_class="DinoClassifier",
        base_model="DinoV2_Small",
        params={"num_classes": 2, "type": "IMAGENET"},
        target_layer=lambda model: model.backbone.blocks[-1],
    )
    DinoV2_Base = ModelParams(
        model_class="DinoClassifier",
        base_model="DinoV2_Base",
        params={"num_classes": 2, "type": "IMAGENET"},
        target_layer=lambda model: model.backbone.blocks[-1],
    )

    DinoV2RS_Small = ModelParams(
        model_class="DinoClassifier",
        base_model="DinoV2_Small",
        params={"num_classes": 2, "type": "RS"},
        target_layer=lambda model: model.backbone.blocks[-1],
    )
    DinoV2RS_Base = ModelParams(
        model_class="DinoClassifier",
        base_model="DinoV2_Base",
        params={"num_classes": 2, "type": "RS"},
        target_layer=lambda model: model.backbone.blocks[-1],
    )

    DinoV2RSL_Small = ModelParams(
        model_class="LoRADinoClassifier",
        base_model="DinoV2_Small",
        params={"num_classes": 2, "type": "RS", "lora_r": 4, "lora_alpha": 4},
        target_layer=lambda model: model.backbone.blocks[-1],
    )
    DinoV2RSL_Base = ModelParams(
        model_class="LoRADinoClassifier",
        base_model="DinoV2_Base",
        params={"num_classes": 2, "type": "RS", "lora_r": 4, "lora_alpha": 4},
        target_layer=lambda model: model.backbone.blocks[-1],
    )

    DinoV3_VitL16_SAT = ModelParams(
        model_class="DinoV3Classifier",
        base_model="dinov3_vitl16",
        params={
            "num_classes": 2,
            "checkpoint_path": "core/original_models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        },
        target_layer=lambda model: model.backbone.blocks[-1],
    )

    DeiT_Tiny_TSCAM = ModelParams(
        model_class="TSCAM",
        base_model="deit_tscam_small_patch16_224",
        params={
            "config_file": "core/TSCAM/configs/CUB/deit_tscam_small_patch16_224.yaml",
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "drop_block_rate": None,
        },
    )

    DeiT_Tiny = ModelParams(base_model="deit_tiny_patch16_224.fb_in1k")
    DeiT_Base = ModelParams(base_model="deit_base_patch16_224.fb_in1k")
    ViT_Tiny = ModelParams(base_model="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    ViT_Base = ModelParams(base_model="vit_base_patch16_224.augreg_in21k_ft_in1k")

    @property
    def model_family(self) -> str:
        """Returns the model family/class.

        Returns:
            Model class identifier string.
        """
        return self.value.model_class

    @property
    def needs_attention_maps(self) -> bool:
        """Check if model needs attention maps.

        Returns:
            True if model requires attention map extraction.
        """
        return "Dino" not in self.name


class TimmEmbeddings(nn.Module):
    """Wrapper for TIMM models to support embeddings and classification.

    This wrapper adds a projection head for embeddings and a classifier head
    on top of TIMM base models.
    """

    def __init__(self, base_model: nn.Module, num_classes: int, proj_dim: int = 128):
        """Initialize TimmEmbeddings wrapper.

        Args:
            base_model: Base TIMM model instance.
            num_classes: Number of output classes.
            proj_dim: Projection dimension for embeddings.
        """
        super().__init__()
        self.model = base_model
        self.feature_dim = self.model.num_features

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, proj_dim),
        )

        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if hasattr(self.model, "blocks"):
            self.blocks = self.model.blocks

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """Forward pass with optional embedding return.

        Args:
            x: Input tensor.
            return_embedding: Whether to return embeddings along with logits.

        Returns:
            Logits tensor or tuple of (logits, projection).
        """
        emb = self.model.forward_features(x)

        if emb.ndim == 4:
            cls_emb = emb.flatten(2).mean(dim=-1)
        elif emb.ndim == 3:
            cls_emb = emb[:, 0]
        else:
            cls_emb = emb

        logits = self.classifier(cls_emb)
        projection = self.projection_head(cls_emb)

        if return_embedding:
            return logits, projection
        return logits


class ModelFactory:
    """Factory class for creating different model architectures."""

    @staticmethod
    def create_model(
        model_type: Models,
        num_classes: Optional[int] = None,
        yaml_config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Create a model instance based on the specified type.

        Args:
            model_type: The model type from Models enum.
            num_classes: Number of output classes (overrides default).
            yaml_config: Optional config from YAML file (overrides enum defaults).

        Returns:
            Instantiated model.
        """
        model_params = model_type.value
        params = model_params.params.copy()

        # Override with YAML config values if provided
        if yaml_config:
            for key in ["lora_r", "lora_alpha"]:
                if key in yaml_config:
                    params[key] = yaml_config[key]

        if num_classes is not None:
            params["num_classes"] = num_classes

        return ModelFactory._dispatch_creation(model_params, params)

    @staticmethod
    def _dispatch_creation(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Dispatch to appropriate creation method based on model class.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated model.
        """
        creators = {
            "FNClassifier": ModelFactory._create_fn_classifier,
            "DinoClassifier": ModelFactory._create_dino_classifier,
            "LoRADinoClassifier": ModelFactory._create_lora_dino_classifier,
            "DinoV3Classifier": ModelFactory._create_dinov3_classifier,
            "TSCAM": ModelFactory._create_tscam,
            "timm": ModelFactory._create_timm,
        }

        creator = creators.get(model_params.model_class)
        if creator is None:
            raise ValueError(f"Unsupported model class: {model_params.model_class}")

        return creator(model_params, params)

    @staticmethod
    def _create_fn_classifier(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Create FNClassifier model.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated FNClassifier.
        """

        return FNClassifier(
            model_name=model_params.base_model,
            num_classes=params["num_classes"],
            h_dimm=params["h_dimm"],
            f_dimm=params["f_dimm"],
            fm2d=params["fm2d"],
        )

    @staticmethod
    def _create_dino_classifier(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Create DinoClassifier model.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated DinoClassifier.
        """

        return DinoClassifier(
            model_params.base_model,
            params["num_classes"],
            params["type"],
        )

    @staticmethod
    def _create_lora_dino_classifier(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Create LoRADinoClassifier model.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated LoRADinoClassifier.
        """
        return LoRADinoClassifier(
            model_name=model_params.base_model,
            num_classes=params["num_classes"],
            pretrain_type=params["type"],
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 4),
        )

    @staticmethod
    def _create_dinov3_classifier(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Create DinoV3Classifier model.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated DinoV3Classifier.
        """
        from .dinov3_classifier import DinoV3Classifier

        return DinoV3Classifier(
            model_name=model_params.base_model,
            num_classes=params["num_classes"],
            checkpoint_path=params.get("checkpoint_path"),
        )

    @staticmethod
    def _create_tscam(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Create TSCAM model.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated TSCAM model.
        """

        cfg_from_file(params["config_file"])
        config.BASIC.ROOT_DIR = "core/TSCAM/"

        return timm_create_model(
            config.MODEL.ARCH,
            pretrained=True,
            num_classes=params["num_classes"],
            drop_rate=params["drop_rate"],
            drop_path_rate=params["drop_path_rate"],
            drop_block_rate=params["drop_block_rate"],
        )

    @staticmethod
    def _create_timm(model_params: ModelParams, params: Dict[str, Any]) -> nn.Module:
        """Create TIMM model with embedding support.

        Args:
            model_params: Model parameter configuration.
            params: Runtime parameters.

        Returns:
            Instantiated TIMM model with embeddings.
        """
        base_model = timm_create_model(
            model_params.base_model,
            pretrained=True,
            num_classes=0,
        )
        return TimmEmbeddings(base_model, params["num_classes"])


def list_all_models() -> List[str]:
    """List all available model names.

    Returns:
        List of model name strings.
    """
    model_names = [model.name for model in Models]
    log_info(f"Available Models:\n\n{chr(10).join(model_names)}")
    return model_names


def load_model(model: nn.Module, path: str = "DinoV2_Small") -> None:
    """Load model weights from file.

    Args:
        model: Model instance to load weights into.
        path: Model name (used to construct file path).
    """
    file_path = f"models/{path}.pth"
    if not os.path.exists(file_path):
        log_info(f"Model file not found: {file_path}")
        log_info(f"Using randomly initialized model (training from scratch)")
        model.eval()
        return
    model.load_state_dict(torch.load(file_path, map_location="cpu"), strict=False)
    model.eval()
    log_info(f"Model loaded from: {file_path}")


def get_model(
    model_name: str, num_classes: Optional[int] = None, device: torch.device = torch.device("cpu")
) -> nn.Module:
    """Create and initialize a model.

    Args:
        model_name: Name of the model from Models enum.
        num_classes: Number of output classes.
        device: Device to place the model on.

    Returns:
        Initialized model on specified device.
    """
    # Get model config from YAML if available
    from core.config.config import get_config
    model_config = get_config("models", model_name) or {}

    model_instance = ModelFactory.create_model(Models[model_name], num_classes, model_config)
    model_instance.train()
    model_instance.to(device)
    return model_instance


def add_attn_maps(model: nn.Module) -> nn.Module:
    """Add attention map extraction capabilities to a transformer model.

    Args:
        model: Transformer model with attention blocks.

    Returns:
        Modified model with attention extraction methods.
    """

    def get_last_self_attention(input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract and compute attention rollout from model.

        Args:
            input_tensor: Input tensor for forward pass.

        Returns:
            Attention rollout tensor.
        """
        model(input_tensor)
        attention_maps = []

        for blk in model.blocks:
            attn_map = blk.attn.attn_map.max(dim=1).values.squeeze(0).detach().cpu()
            attention_maps.append(attn_map)

        rollout = []
        identity = torch.eye(attention_maps[0].shape[-1])
        product = identity

        for attn_map in attention_maps:
            product = product @ (attn_map + identity)
            product = product / product.sum(dim=-1, keepdim=True)
            rollout.append(product)

        return torch.stack(rollout).unsqueeze(0)

    def create_forward_wrapper(attn_module):
        """Create a forward wrapper that captures attention maps.

        Args:
            attn_module: Attention module to wrap.

        Returns:
            Forward function with attention capture.
        """

        def forward_with_attn(inp: torch.Tensor) -> torch.Tensor:
            """Forward pass with attention map capture.

            Args:
                inp: Input tensor.

            Returns:
                Output tensor.
            """
            batch_size, seq_len, channels = inp.shape
            qkv = (
                attn_module.qkv(inp)
                .reshape(batch_size, seq_len, 3, attn_module.num_heads, attn_module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )

            queries, keys, values = qkv.unbind(0)
            queries, keys = attn_module.q_norm(queries), attn_module.k_norm(keys)

            queries = queries * attn_module.scale
            attention = queries @ keys.transpose(-2, -1)
            attention = attention.softmax(dim=-1)
            attention = attn_module.attn_drop(attention)

            attn_module.attn_map = attention
            attn_module.cls_attn_map = attention[:, :, 0, 2:]

            output = attention @ values
            output = output.transpose(1, 2).reshape(batch_size, seq_len, channels)
            output = attn_module.proj(output)
            output = attn_module.proj_drop(output)

            return output

        return forward_with_attn

    for blk in model.blocks:
        blk.attn.forward = create_forward_wrapper(blk.attn)

    model.get_last_self_attention = get_last_self_attention

    return model


if __name__ == "__main__":
    test_model = get_model("ViT_Tiny", num_classes=2)
    test_model = add_attn_maps(test_model)

    test_input = torch.randn(1, 3, 224, 224)
    _ = test_model(test_input)

    attention_rollout = test_model.get_last_self_attention(test_input)
    log_info(f"Attention rollout shape: {attention_rollout.shape}")
