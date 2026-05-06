"""
Attention map extraction module for Vision Transformers.

Provides functions to extract attention maps from transformer models,
including last-layer attention and attention rollout methods.
"""
import torch
import torch.nn.functional as F
import numpy as np

from core.config.config import get_config


def get_last_layer_attention(model, model_name, img_tensor):
    """
    Extract attention map from the last transformer layer.

    This method extracts the attention weights from the final layer,
    focusing on how the CLS token attends to spatial patches.

    Args:
        model: Transformer model with get_last_self_attention() method
        model_name: Model name for config lookup (patch_size)
        img_tensor: Input image tensor [B, C, H, W]

    Returns:
        np.ndarray: Attention heatmap [H, W] normalized to [0, 1]
    """
    with torch.no_grad():
        last_attn = model.get_last_self_attention(img_tensor)

        # Token order: [CLS(0), reg1(1)..reg4(4), patches(5+)]
        B, heads, tokens, _ = last_attn.shape
        num_reg = 4 if tokens > 5 else 0  # DINOv2 has 4 register tokens

        # Aggregate heads by mean
        attn_aggregated = last_attn.mean(dim=1)
        cls_attn = attn_aggregated[0, 0, 1 + num_reg:]  # CLS -> patches

        # Get patch dimensions
        patch_size = 14
        cfg_ps = get_config("models", model_name, "patch_size")
        if cfg_ps is not None:
            patch_size = int(cfg_ps)

        _, _, H, W = img_tensor.shape
        w_featmap = H // patch_size
        h_featmap = W // patch_size
        n_patches = w_featmap * h_featmap

        # Adjust size if needed
        cur = cls_attn.shape[0]
        if cur != n_patches:
            if cur > n_patches:
                cls_attn = cls_attn[:n_patches]
            else:
                pad = torch.zeros(n_patches - cur, device=cls_attn.device)
                cls_attn = torch.cat([cls_attn, pad], dim=0)

        # Reshape to spatial dimensions
        attn_map = cls_attn.reshape(w_featmap, h_featmap)

        # Upsample to image size
        attn_up = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        # Normalize to [0, 1]
        attn_np = attn_up.detach().cpu().numpy()
        attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

        return attn_np
