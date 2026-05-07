"""GradCAM visualization for CNN and transformer models.

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for both
convolutional neural networks and vision transformers.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Dict

from core.config.config import log_info
from .crf_refiner import refine_heatmap
from .models import Models


def compute_gradcam(model, model_name, img_tensor, class_index=None):
    """Compute standard Grad-CAM for CNN models.

    Implements the original Grad-CAM algorithm for convolutional neural
    networks. Uses gradient-weighted class activation mapping to produce
    visual explanations for CNN predictions.

    Args:
        model: CNN model with convolutional layers.
        model_name: Name of the model for target layer lookup.
        img_tensor: Input image tensor [B, C, H, W].
        class_index: Target class index. If None, uses predicted class.

    Returns:
        Heatmap [H, W] normalized to [0, 1] as numpy array.
    """
    activations = None
    grads = None

    if Models[model_name].value.target_layer is not None:
        conv_layer = Models[model_name].value.target_layer(model)
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        def backward_hook(module, grad_in, grad_out):
            nonlocal grads
            grads = grad_out[0]

        hook = conv_layer.register_forward_hook(forward_hook)
        bwd_hook = conv_layer.register_full_backward_hook(backward_hook)

    img_tensor = img_tensor.detach()
    img_tensor.requires_grad_(True)
    if img_tensor.grad is not None:
        img_tensor.grad.zero_()

    if Models[model_name].value.target_layer is None:
        model.gradients = []
        model.activations = []
        preds = model(img_tensor, save_grads=True)
    else:
        preds = model(img_tensor)

    if class_index is None:
        class_index = preds.argmax(dim=1)

    loss = preds[:, class_index]
    model.zero_grad()
    loss.backward()

    if Models[model_name].value.target_layer is None:
        grads = torch.cat(model.gradients)
        activations = torch.cat(model.activations)

    pooled_grads = torch.mean(grads, axis=(0, 2, 3))
    pooled_grads = pooled_grads.detach().cpu().numpy()

    if Models[model_name].value.target_layer is None:
        for hook in model.hooks:
            hook.remove()
    else:
        hook.remove()
        bwd_hook.remove()

    activations = activations.detach().cpu().numpy()[0]

    for i in range(pooled_grads.shape[0]):
        activations[i, ...] *= pooled_grads[i]

    heatmap = np.mean(activations, axis=0)
    heatmap = np.maximum(heatmap, 0)
    max_value = np.max(heatmap)
    if max_value:
        heatmap /= max_value

    return heatmap


def refine_heatmap_with_crf(
    heatmap: np.ndarray,
    image: Optional[np.ndarray] = None,
    use_crf: bool = True,
    crf_config: Optional[Dict] = None
) -> np.ndarray:
    """Apply Conditional Random Field (CRF) refinement to a heatmap.

    CRF refinement uses the original image's color information to align
    heatmap boundaries with actual object edges, producing sharper and
    more accurate visualizations.

    Args:
        heatmap: Input heatmap [H, W] normalized to [0, 1].
        image: Original image [H, W, 3] in BGR or RGB format. Used for
            color-based edge alignment. If None, CRF cannot be applied.
        use_crf: Whether to apply CRF refinement (default=True).
        crf_config: Dictionary with CRF parameters. Supported keys:
            - method: 'crf' (default)
            - iterations: Number of CRF iterations (default=10)
            - theta_xy: Spatial standard deviation (default=80.0)
            - w1: Weight for spatial kernel (default=10.0)
            - w2: Weight for bilateral kernel (default=3.0)
            - sigma_x: Spatial sigma X (default=3.0)
            - sigma_y: Spatial sigma Y (default=3.0)
            - sigma_rgb: RGB similarity sigma (default=13.0)

    Returns:
        Refined heatmap [H, W] normalized to [0, 1]. Returns original
        heatmap if use_crf=False or refinement fails.
    """
    if not use_crf:
        return heatmap

    if crf_config is None:
        crf_config = {}

    try:
        refined = refine_heatmap(
            heatmap,
            image=image,
            method=crf_config.get('method', 'crf'),
            iterations=crf_config.get('iterations', 10),
            theta_xy=crf_config.get('theta_xy', 80.0),
            w1=crf_config.get('w1', 10.0),
            w2=crf_config.get('w2', 3.0),
            sigma_x=crf_config.get('sigma_x', 3.0),
            sigma_y=crf_config.get('sigma_y', 3.0),
            sigma_rgb=crf_config.get('sigma_rgb', 13.0)
        )
        return refined
    except Exception as e:
        log_info(f"  ⚠ CRF refinement failed: {e}, returning original heatmap")
        return heatmap


def overlay_heatmap(img_path, heatmap, alpha=0.6, img_only=False):
    """Overlay a heatmap on the original image.

    Creates a visualization by superimposing a color-mapped heatmap onto
    the original image. Uses the JET colormap for heatmap visualization.

    Args:
        img_path: Path to the original image file.
        heatmap: Heatmap array [H, W] with values in [0, 1].
        alpha: Blending factor for overlay (default=0.6). Higher values
            show more of the original image.
        img_only: If True, returns only the heatmap without blending
            with the original image (default=False).

    Returns:
        Tuple of (superimposed_img, original_img):
            - superimposed_img: RGB array with heatmap overlay or heatmap only
            - original_img: Original RGB image array
    """
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    heatmap = cv2.resize(
        heatmap, (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap
    if not img_only:
        superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    return superimposed_img, img
