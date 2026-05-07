"""Custom neural network architectures for fire classification.

This module provides custom model architectures that combine feature extraction
backbones with multi-scale feature fusion for improved classification performance.
"""

import torch
import torch.nn as nn
from timm.models import create_model


class FNClassifier(nn.Module):
    """Feature Network Classifier with multi-scale feature fusion.

    Extracts multi-scale features from a backbone model and fuses them through
    projection layers before classification. Supports GradCAM through gradient
    and activation hooks.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        h_dimm: int = 256,
        f_dimm: int = 512,
        fm2d: int = 7
    ):
        """Initialize FNClassifier with specified architecture parameters.

        Args:
            model_name: Name of the backbone model from timm library.
            num_classes: Number of output classes for classification.
            pretrained: Whether to use pretrained backbone weights.
            h_dimm: Hidden dimension for projection layers.
            f_dimm: Feature dimension before final classification.
            fm2d: Target spatial dimension for feature map pooling.
        """
        super(FNClassifier, self).__init__()

        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )

        feature_dims = self.backbone.feature_info.channels()

        self.proj = nn.ModuleList()

        for i in range(len(feature_dims)):
            layers = [
                nn.Conv2d(feature_dims[i], h_dimm, kernel_size=1),
                nn.BatchNorm2d(h_dimm),
                nn.ReLU(inplace=True)
            ]
            if i < len(feature_dims)-1:
                layers.append(nn.AdaptiveAvgPool2d((fm2d, fm2d)))

            self.proj.append(nn.Sequential(*layers))

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(len(feature_dims)*h_dimm),
            nn.ReLU(inplace=True),
            nn.Conv2d(len(feature_dims)*h_dimm, f_dimm, kernel_size=1),
            nn.BatchNorm2d(f_dimm),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(f_dimm, num_classes)
        )

        self.gradients = []
        self.activations = []
        self.hooks = []

    def save_gradient(self, idx: int):
        """Create a hook function to save gradients at specified index.

        Args:
            idx: Index to store the gradient in saved_gradients list.

        Returns:
            Hook function that saves gradients.
        """
        def hook(grad):
            self.saved_gradients[idx] = grad
        return hook

    def forward(self, x: torch.Tensor, save_grads: bool = False) -> torch.Tensor:
        """Forward pass through the network with optional gradient saving.

        Args:
            x: Input tensor of shape (B, C, H, W).
            save_grads: Whether to register hooks for gradient saving.

        Returns:
            Classification logits of shape (B, num_classes).
        """
        feature = self.backbone(x)
        proj_outs = [layer(feat) for layer, feat in zip(self.proj, feature)]

        if save_grads:
            def save_gradient(grad):
                self.gradients.append(grad)

            def save_activation(activation):
                self.activations.append(activation)

            for i, p_out in enumerate(proj_outs):
                hook = p_out.register_hook(save_gradient)
                hook2 = p_out.register_hook(save_activation)
                self.hooks.append(hook)
                self.hooks.append(hook2)

        classifier_out = self.classifier(torch.cat(proj_outs, dim=1))
        return classifier_out
