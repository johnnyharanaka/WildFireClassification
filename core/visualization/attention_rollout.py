"""Attention Rollout visualization for transformer models.

Implements Attention Rollout (Abnar & Zuidema, ACL 2020) for interpreting
transformer models, with optional extensions for improved visualization.

References:
    - Abnar & Zuidema. "Quantifying Attention Flow in Transformers." ACL 2020.
    - Chefer et al. "Transformer Interpretability Beyond Attention Visualization." CVPR 2021.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

from core.config.config import get_config, log_info

def aggregate_heads_by_entropy(attn_tensor):
    """Aggregate attention heads weighted by inverse entropy.

    Heads with lower entropy are more focused and get higher weights.
    This preserves discriminative patterns in heads with sharp attention.

    Args:
        attn_tensor: Tensor of shape (L, B, H, T, T) where H is num heads

    Returns:
        Aggregated attention of shape (L, B, T, T)
    """
    L, B, H, T, _ = attn_tensor.shape

    # Compute entropy for each head (lower entropy = more focused)
    # entropy shape: (L, B, H)
    entropy = -(attn_tensor * torch.log(attn_tensor + 1e-8)).sum(dim=-1).mean(dim=-1)

    # Weight heads by inverse entropy (lower entropy gets higher weight)
    # weights shape: (L, B, H, 1, 1)
    weights = torch.softmax(-entropy, dim=2).unsqueeze(-1).unsqueeze(-1)

    # Weighted average across heads
    aggregated = (attn_tensor * weights).sum(dim=2)

    return aggregated


def compute_gradient_weights(attention_tensor, gradient_tensor=None, scale=1.0):
    """Compute gradient-based weights to emphasize attention boundaries.

    Strategy: Detects edges and transitions using gradient magnitude,
    emphasizing regions with abrupt changes. This helps highlight important
    boundaries in the attention map.

    Args:
        attention_tensor: Original attention map [H, W].
        gradient_tensor: Pre-computed gradients (optional). If None, uses
            finite differences to approximate gradients.
        scale: Scaling factor to emphasize weights (default=1.0).
            Recommended range: 1.0-3.0. Higher values produce stronger effects.

    Returns:
        Weight tensor [H, W] normalized to [0, 1].
    """
    att = attention_tensor.clone().float()

    if gradient_tensor is not None:
        grad = gradient_tensor.clone().float()
        grad_norm = torch.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
        grad_magnitude = torch.abs(grad)
    else:
        if att.ndim == 2:
            dy = torch.abs(att[1:, :] - att[:-1, :])
            dx = torch.abs(att[:, 1:] - att[:, :-1])

            dy = F.pad(dy, (0, 0, 1, 0), value=0)
            dx = F.pad(dx, (1, 0, 0, 0), value=0)

            grad_magnitude = torch.sqrt(dy**2 + dx**2 + 1e-8)
        else:
            grad_magnitude = torch.zeros_like(att)

    grad_norm = grad_magnitude.max()
    if grad_norm > 1e-8:
        grad_magnitude = grad_magnitude / grad_norm

    grad_magnitude = grad_magnitude ** (1.0 / max(scale, 0.5))
    weights = att * (1.0 + scale * grad_magnitude)

    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = torch.zeros_like(weights)

    return weights


def compute_multiscale_attention_maps(
    model,
    model_name,
    img_tensor,
    use_mean_patches=False,
):
    """Compute all attention maps separately.

    Args:
        model: Transformer model with get_all_self_attentions() method.
        model_name: Name of the model for configuration lookup.
        img_tensor: Input image tensor [B, C, H, W].
        use_mean_patches: Use mean of patches instead of CLS token (default=False).

    Returns:
        a heatmap [H, W] normalized to [0, 1] as numpy array.
    """
    x = img_tensor.detach()
    device = x.device

    patch_size = 16
    cfg_ps = get_config("models", model_name, "patch_size")
    if cfg_ps is not None:
        patch_size = int(cfg_ps)

    _, _, H, W = x.shape
    w_featmap = H // patch_size
    h_featmap = W // patch_size
    n_patches = w_featmap * h_featmap

    with torch.no_grad():
        all_attns = None
        if hasattr(model, "get_all_self_attentions"):
            try:
                all_attns = model.get_all_self_attentions(x)
            except Exception:
                raise ValueError("Unexpected attention format")            

        layers = []
        if isinstance(all_attns, (list, tuple)):
            for a in all_attns:
                if not torch.is_tensor(a):
                    raise TypeError("Each attention must be a tensor")
                if a.dim() == 4:
                    layers.append(a)
                elif a.dim() == 5:
                    for i in range(a.shape[0]):
                        layers.append(a[i])
                else:
                    raise ValueError(f"Unexpected attention format: {a.shape}")

        num_layers = len(layers)
        result = {}

        for layer_idx in range(num_layers):
            attn = layers[layer_idx]

            # Remove register tokens if present
            B, heads, tokens, _ = attn.shape
            if "Dino" in model_name and tokens > 5:
                # Keep CLS and patches, remove register tokens
                num_reg = 4
                keep = torch.cat([
                    torch.tensor([0], device=device),
                    torch.arange(1 + num_reg, tokens, device=device)
                ])
                attn = attn[..., keep, :]
                attn = attn[..., keep]
                tokens = attn.shape[-1]

            # Aggregate heads by entropy
            attn_aggregated = aggregate_heads_by_entropy(attn.unsqueeze(0))[0]

            # Extract CLS attention or mean patches
            if use_mean_patches == True:
                attn_vec = attn_aggregated[0, 1:, 1:].mean(dim=0)
                cls_attention = attn_aggregated[0, 0, 1:]
                attn_vec = 0.6 * attn_vec + 0.4 * cls_attention
            elif use_mean_patches == False:
                cls_map = attn_aggregated[:, 0, 1:]
                attn_vec = cls_map[0]
            else:
                attn_patches = attn_aggregated[0, 1:, 1:].mean(dim=0)
                cls_attention = attn_aggregated[0, 0, 1:]
                cls_weight = float(use_mean_patches)
                patch_weight = 1.0 - cls_weight
                attn_vec = patch_weight * attn_patches + cls_weight * cls_attention

            # Adjust to n_patches
            cur = attn_vec.shape[-1]
            if cur != n_patches:
                if cur > n_patches:
                    attn_vec = attn_vec[:n_patches]
                else:
                    pad = torch.zeros(n_patches - cur, device=device)
                    attn_vec = torch.cat([attn_vec, pad], dim=0)

            # Reshape and interpolate
            attn_map = attn_vec.reshape(w_featmap, h_featmap).unsqueeze(0)
            up = F.interpolate(attn_map.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
            attn_up = up[0]

            # Normalize
            result_np = attn_up.detach().cpu().numpy()
            result_np = (result_np - result_np.min()) / (result_np.max() - result_np.min() + 1e-8)
            result[f'layer_{layer_idx}'] = result_np

        return result


def _extract_attention_map(attn, w_featmap, h_featmap, n_patches, patch_size, use_mean_patches, device):
    """Helper function to extract attention map from a single layer."""
    B, heads, tokens, _ = attn.shape

    # Aggregate heads
    attn_mean = attn.mean(dim=1)

    # Extract CLS or patches
    if use_mean_patches == False:
        cls_map = attn_mean[:, 0, 1:]
        attn_vec = cls_map[0]
    else:
        attn_vec = attn_mean[0, 1:, 1:].mean(dim=0)

    # Adjust size
    cur = attn_vec.shape[-1]
    if cur != n_patches:
        if cur > n_patches:
            attn_vec = attn_vec[:n_patches]
        else:
            pad = torch.zeros(n_patches - cur, device=device)
            attn_vec = torch.cat([attn_vec, pad], dim=0)

    # Reshape and upsample
    attn_map = attn_vec.reshape(w_featmap, h_featmap).unsqueeze(0)
    up = F.interpolate(attn_map.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
    attn_up = up[0]

    result_np = attn_up.detach().cpu().numpy()
    result_np = (result_np - result_np.min()) / (result_np.max() - result_np.min() + 1e-8)
    return result_np


def compute_attention_rollout(
    model,
    model_name,
    img_tensor,
    use_gradient_weighting=False,
    gradient_weight_scale=0.2,
    use_mean_patches=False,
    use_gaussian_blur=False,
    gaussian_sigma=3,
):
    """Compute attention rollout following Abnar & Zuidema (ACL 2020).

    Implementation of Attention Rollout from "Quantifying Attention Flow
    in Transformers" with optional extensions:

    Core method (from paper):
    - Sequential multiplication of attention matrices across layers
    - Residual connections via 0.5*W_att + 0.5*I
    - Tracks how input token identities propagate through layers

    Extensions (not in original paper):
    - Entropy-weighted head aggregation (inspired by Chefer et al. 2021)
    - Mean patch aggregation option (in addition to CLS token)
    - Gradient weighting for edge emphasis
    - Gaussian smoothing to reduce artifacts

    Reference:
        Samira Abnar and Willem Zuidema. "Quantifying Attention Flow in
        Transformers." ACL 2020.

    Args:
        model: Transformer model with get_all_self_attentions() method.
        model_name: Name of the model for configuration lookup.
        img_tensor: Input image tensor [B, C, H, W].
        use_gradient_weighting: Apply gradient-based weighting (default=False).
        gradient_weight_scale: Gradient weight scale factor (default=0.2).
        use_mean_patches: Use mean of patches instead of CLS token (default=False).
            When False: uses CLS token attention (as in original paper)
            When True: uses mean of patch tokens (extension)
            When float: weighted combination of CLS and patches
        use_gaussian_blur: Apply Gaussian smoothing (default=False).
        gaussian_sigma: Standard deviation for Gaussian filter (default=3).

    Returns:
        Heatmap [H, W] normalized to [0, 1] as numpy array.
    """
    x = img_tensor.detach()
    device = x.device

    patch_size = 16
    cfg_ps = get_config("models", model_name, "patch_size")
    if cfg_ps is not None:
        patch_size = int(cfg_ps)

    _, _, H, W = x.shape
    w_featmap = H // patch_size
    h_featmap = W // patch_size
    n_patches = w_featmap * h_featmap

    with torch.no_grad():
        all_attns = None
        if hasattr(model, "get_all_self_attentions"):
            try:
                all_attns = model.get_all_self_attentions(x)
            except Exception:
                all_attns = None

        if all_attns is None:
            last = model.get_last_self_attention(x)
            all_attns = [last]

        layers = []
        if isinstance(all_attns, (list, tuple)):
            for a in all_attns:
                if not torch.is_tensor(a):
                    raise TypeError("Each attention must be a tensor")
                if a.dim() == 4:
                    layers.append(a)
                elif a.dim() == 5:
                    for i in range(a.shape[0]):
                        layers.append(a[i])
                else:
                    raise ValueError(f"Unexpected attention format: {a.shape}")
            attn_tensor = torch.stack(layers, dim=0)
        else:
            a = all_attns
            if a.dim() == 4:
                attn_tensor = a.unsqueeze(0)
            elif a.dim() == 5:
                attn_tensor = a
            else:
                raise ValueError(f"Unexpected attention format: {a.shape}")

        L, B, heads, tokens, _ = attn_tensor.shape

        if "Dino" in model_name and tokens > 5:
            # Token order: [CLS(0), reg1..reg4(1-4), patches(5+)]
            # Keep CLS and patches, remove only register tokens
            num_reg = 4
            keep = torch.cat([
                torch.tensor([0], device=device),
                torch.arange(1 + num_reg, tokens, device=device)
            ])
            attn_tensor = attn_tensor[..., keep, :]
            attn_tensor = attn_tensor[..., keep]
            tokens = attn_tensor.shape[-1]

        # Use entropy-weighted aggregation to preserve discriminative patterns
        attn_mean = aggregate_heads_by_entropy(attn_tensor)

        I = torch.eye(tokens, device=device).unsqueeze(0).unsqueeze(0)
        aug_attn = 0.5 * attn_mean + 0.5 * I
        aug_attn = aug_attn / aug_attn.sum(dim=-1, keepdim=True)

        # Standard sequential rollout (Attention Rollout from Abnar & Zuidema, ACL 2020)
        rollout = aug_attn[0]
        for i in range(1, L):
            rollout = torch.bmm(aug_attn[i], rollout)
            rollout_sum = rollout.sum(dim=-1, keepdim=True)
            rollout = rollout / (rollout_sum + 1e-8)

        if use_mean_patches:
            attn_vec = rollout[0, 1:, 1:].mean(dim=0)
            cls_attention = rollout[0, 0, 1:]
            attn_vec = 0.6 * attn_vec + 0.4 * cls_attention
        elif not use_mean_patches:
            cls_map = rollout[:, 0, 1:]
            attn_vec = cls_map[0]
        else:
            attn_patches = rollout[0, 1:, 1:].mean(dim=0)
            cls_attention = rollout[0, 0, 1:]
            cls_weight = float(use_mean_patches)
            patch_weight = 1.0 - cls_weight
            attn_vec = patch_weight * attn_patches + cls_weight * cls_attention

        cur = attn_vec.shape[-1]
        if cur != n_patches:
            if cur > n_patches:
                attn_vec = attn_vec[:n_patches]
            else:
                pad = torch.zeros(n_patches - cur, device=device)
                attn_vec = torch.cat([attn_vec, pad], dim=0)

        attn_map = attn_vec.reshape(w_featmap, h_featmap).unsqueeze(0)
        up = F.interpolate(attn_map.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        attn_up = up[0]

        if use_gradient_weighting:
            attn_up = compute_gradient_weights(attn_up, gradient_tensor=None, scale=gradient_weight_scale)
            result_np = attn_up.detach().cpu().numpy()

            if use_gaussian_blur:
                try:
                    result_np = gaussian_filter(result_np, sigma=gaussian_sigma)
                    result_np = np.clip(result_np, 0, 1)
                except Exception as e:
                    log_info(f"  ⚠ Gaussian blur failed: {e}, returning original heatmap")

            return result_np

        attn_up = attn_up.float()
        mn, mx = attn_up.min(), attn_up.max()
        if mx > mn:
            attn_up = (attn_up - mn) / (mx - mn)
        else:
            attn_up = torch.zeros_like(attn_up)

        result_np = attn_up.detach().cpu().numpy()

        if use_gaussian_blur:
            try:
                result_np = gaussian_filter(result_np, sigma=gaussian_sigma)
                result_np = np.clip(result_np, 0, 1)
            except Exception as e:
                log_info(f"  ⚠ Gaussian blur failed: {e}, returning original heatmap")

        return result_np

