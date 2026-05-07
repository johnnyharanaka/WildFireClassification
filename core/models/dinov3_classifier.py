"""
DinoV3 Classifier implementation with ViT-L/16 backbone.
Supports SAT-493M satellite imagery pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple
import sys

class DinoV3Classifier(nn.Module):
    """DinoV3-based classifier with ViT-L/16 backbone.

    Architecture:
    - Patch size: 16
    - Embedding dim: 1024
    - Attention heads: 16
    - Storage tokens: 4
    - Parameters: ~300M
    - Pretrained on: SAT-493M (satellite imagery, 493M images)

    Attributes:
        backbone: The DinoV3 vision transformer backbone.
        classifier: Linear layer for classification.
        projection_head: MLP for generating embeddings.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        num_classes: int = 2,
        checkpoint_path: str = None,
        proj_dim: int = 128,
    ):
        """Initialize DinoV3Classifier.

        Args:
            model_name: Name of the model (currently only "dinov3_vitl16" supported).
            num_classes: Number of output classes for classification.
            checkpoint_path: Path to pretrained checkpoint. If None, loads from hub.
            proj_dim: Dimension of projection head output.
        """
        super().__init__()

        if model_name != "dinov3_vitl16":
            raise ValueError(f"Only dinov3_vitl16 is currently supported, got {model_name}")

        # Import DinoV3 backbone builder
        dinov3_path = str(Path(__file__).parent.parent.parent / "backbones" / "dinov3")
        if dinov3_path not in sys.path:
            sys.path.insert(0, dinov3_path)

        try:
            from dinov3.hub.backbones import dinov3_vitl16
        except ImportError:
            raise ImportError(
                "Could not import dinov3. Make sure the dinov3 directory exists at core/dinov3/"
            )

        # Load backbone with checkpoint
        if checkpoint_path is not None:
            # Load from local checkpoint
            self.backbone = dinov3_vitl16(
                pretrained=True,
                weights=checkpoint_path,
                check_hash=False
            )
        else:
            # Load from hub with default weights
            self.backbone = dinov3_vitl16(pretrained=True)

        # Get feature dimension (for ViT-L, embed_dim=1024)
        feature_dim = self.backbone.embed_dim

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Projection head for contrastive learning
        # Reduces dimensionality: feature_dim -> feature_dim//2 -> proj_dim
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, proj_dim),
        )

        # KNN bank (populated by fit_knn, persisted in state_dict)
        self.register_buffer("knn_features", None)
        self.register_buffer("knn_labels", None)
        self.knn_k = 20
        self.knn_temperature = 0.07

    def fit_knn(self, features: torch.Tensor, labels: torch.Tensor, k: int = 20, temperature: float = 0.07):
        """Store L2-normalized feature bank for KNN inference."""
        self.knn_features = F.normalize(features, dim=1)
        self.knn_labels = labels
        self.knn_k = k
        self.knn_temperature = temperature

    @property
    def has_knn(self) -> bool:
        return self.knn_features is not None

    def _knn_predict(self, cls_features: torch.Tensor) -> torch.Tensor:
        """KNN classification via weighted voting."""
        query = F.normalize(cls_features, dim=1)
        sim = torch.mm(query, self.knn_features.T)
        topk_sims, topk_idx = sim.topk(self.knn_k, dim=1)
        topk_labels = self.knn_labels[topk_idx]

        weights = F.softmax(topk_sims / self.knn_temperature, dim=1)
        num_classes = self.classifier.out_features
        one_hot = F.one_hot(topk_labels, num_classes).float()
        logits = (weights.unsqueeze(-1) * one_hot).sum(dim=1)
        return logits

    def extract_cls_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features without classification head."""
        features_dict = self.backbone.forward_features(x)
        return features_dict["x_norm_clstoken"]

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

        if self.has_knn and not self.training:
            logits = self._knn_predict(cls_features)
        else:
            logits = self.classifier(cls_features)

        # Build return tuple based on requested outputs
        results = [logits]

        # Add embeddings if requested
        if return_embedding:
            projection = self.projection_head(cls_features)
            results.append(projection)

        # Return single value or tuple
        return tuple(results) if len(results) > 1 else results[0]
