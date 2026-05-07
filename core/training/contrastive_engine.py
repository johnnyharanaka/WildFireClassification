"""
Contrastive Learning Engine Module

This module handles all contrastive learning logic including:
- Contrastive loss computation with hard mining
- Metrics tracking (alignment, uniformity, intra/inter-class similarity)
"""

import warnings
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def compute_contrastive_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    contrastive_loss_func: torch.nn.Module,
    miner: Optional[torch.nn.Module] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute contrastive loss from model embeddings.

    Args:
        model: Model with return_embedding support
        features: Input images [batch_size, channels, height, width]
        labels: Ground truth labels [batch_size]
        contrastive_loss_func: Contrastive loss function
        miner: Optional hard negative miner

    Returns:
        Tuple of (loss, normalized_embeddings)
    """
    # Forward pass to get embeddings
    _, embeddings = model(features, return_embedding=True)

    # Normalize embeddings (critical for contrastive learning)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute contrastive loss with optional hard mining
    if miner is not None:
        hard_pairs = miner(embeddings, labels)
        loss = contrastive_loss_func(embeddings, labels, hard_pairs)
    else:
        loss = contrastive_loss_func(embeddings, labels)

    return loss, embeddings


def compute_contrastive_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute quality metrics for contrastive learning.

    Metrics:
    - intra_class_sim: Average cosine similarity within same class
    - inter_class_sim: Average cosine similarity between different classes
    - alignment: How well same-class samples align (lower is better)
    - uniformity: How uniformly distributed embeddings are (lower is better)

    Args:
        embeddings: Normalized embeddings [batch_size, embedding_dim]
        labels: Ground truth labels [batch_size]

    Returns:
        Dictionary with metric names and values
    """
    with torch.no_grad():
        sim_matrix = embeddings @ embeddings.T  # [batch, batch]

        labels_expanded = labels.unsqueeze(1)  # [batch, 1]
        same_class_mask = (labels_expanded == labels_expanded.T)  # [batch, batch]

        batch_size = embeddings.shape[0]
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        same_class_mask = same_class_mask & eye_mask
        diff_class_mask = (~same_class_mask) & eye_mask

        intra_class_sim = 0.0
        if same_class_mask.sum() > 0:
            intra_class_sim = sim_matrix[same_class_mask].mean().item()

        inter_class_sim = 0.0
        if diff_class_mask.sum() > 0:
            inter_class_sim = sim_matrix[diff_class_mask].mean().item()

        alignment = 0.0
        if same_class_mask.sum() > 0:
            distances = 2 - 2 * sim_matrix[same_class_mask]  # Convert similarity to distance
            alignment = distances.mean().item()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*aten::_pdist_forward.*")
            uniformity = torch.pdist(embeddings, p=2).pow(2).mul(-2).exp().mean().log().item()

    return {
        'intra_class_sim': intra_class_sim,
        'inter_class_sim': inter_class_sim,
        'alignment': alignment,
        'uniformity': uniformity
    }


