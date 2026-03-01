"""
LoRA (Low-Rank Adaptation) implementation for Vision Transformers.
Applies low-rank decomposition to attention layers for efficient fine-tuning.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
  """LoRA layer that wraps QKV projection with low-rank adaptations for Q and V.

  Applies LoRA to the query and value projections while keeping the key projection unchanged.
  The adaptation is scaled by alpha/r factor.

  Attributes:
      w: Original QKV projection layer.
      w_a_q: Low-rank down projection for query.
      w_b_q: Low-rank up projection for query.
      w_a_v: Low-rank down projection for value.
      w_b_v: Low-rank up projection for value.
      r: Rank of the low-rank decomposition.
      alpha: Scaling factor for LoRA.
  """

  def __init__(
      self,
      w: nn.Module,
      w_a_q: nn.Module,
      w_b_q: nn.Module,
      w_a_v: nn.Module,
      w_b_v: nn.Module,
      r: int,
      alpha: int,
  ):
    super().__init__()
    self.w = w
    self.w_a_q = w_a_q
    self.w_b_q = w_b_q
    self.w_a_v = w_a_v
    self.w_b_v = w_b_v
    self.r = r
    self.alpha = alpha

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with LoRA adaptation.

    Args:
        x: Input tensor.

    Returns:
        QKV tensor with LoRA adaptations applied to Q and V.
    """
    q, k, v = self.w(x).chunk(3, dim=-1)

    x_q = q + (self.alpha // self.r) * self.w_b_q(self.w_a_q(x))
    x_v = v + (self.alpha // self.r) * self.w_b_v(self.w_a_v(x))

    return torch.cat([x_q, k, x_v], dim=-1)


def apply_lora_to_vit(
    backbone: nn.Module,
    r: int,
    alpha: int,
    lora_layers: Optional[List[int]] = None,
) -> tuple[List[nn.Linear], List[nn.Linear]]:
  """Apply LoRA to a Vision Transformer backbone.

  Freezes all backbone parameters and applies LoRA to specified attention layers.

  Args:
      backbone: The ViT backbone module with transformer blocks.
      r: Rank of the low-rank decomposition.
      alpha: Scaling factor for LoRA.
      lora_layers: List of layer indices to apply LoRA to. If None, applies to all layers.

  Returns:
      Tuple of (w_As, w_Bs) containing all LoRA projection layers for initialization.
  """
  assert r > 0, "LoRA rank must be positive"
  assert alpha > 0, "LoRA alpha must be positive"

  dim = backbone.blocks[0].attn.proj.in_features

  if lora_layers is None:
    lora_layers = list(range(len(backbone.blocks)))

  w_As: List[nn.Linear] = []
  w_Bs: List[nn.Linear] = []

  # Freeze backbone parameters
  for param in backbone.parameters():
    param.requires_grad = False

  # Apply LoRA to specified layers
  for layer_idx, blk in enumerate(backbone.blocks):
    if layer_idx not in lora_layers:
      continue
    if isinstance(blk.attn.qkv, LoRALayer):
      continue

    w_a_linear_q = nn.Linear(dim, r, bias=False)
    w_b_linear_q = nn.Linear(r, dim, bias=False)
    w_a_linear_v = nn.Linear(dim, r, bias=False)
    w_b_linear_v = nn.Linear(r, dim, bias=False)

    w_As.extend([w_a_linear_q, w_a_linear_v])
    w_Bs.extend([w_b_linear_q, w_b_linear_v])

    blk.attn.qkv = LoRALayer(
        blk.attn.qkv,
        w_a_linear_q,
        w_b_linear_q,
        w_a_linear_v,
        w_b_linear_v,
        r,
        alpha,
    )

  return w_As, w_Bs


def init_lora_weights(w_As: List[nn.Linear], w_Bs: List[nn.Linear]) -> None:
  """Initialize LoRA weights.

  Uses Kaiming uniform initialization for A matrices and zeros for B matrices,
  following the original LoRA paper.

  Args:
      w_As: List of down projection layers.
      w_Bs: List of up projection layers.
  """
  for w_A in w_As:
    nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
  for w_B in w_Bs:
    nn.init.zeros_(w_B.weight)