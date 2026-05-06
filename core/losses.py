"""Loss functions for training classification models.

Provides implementations of various loss functions including Focal Loss for
handling class imbalance in fire classification tasks, as well as contrastive
learning setup and utility functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners

from core.config.config import get_config, log_info


class CenterLoss(nn.Module):
    """Center Loss for deep feature learning.

    Learns a center (centroid) for each class and minimizes the distance
    between samples and their corresponding class centers. This encourages
    intra-class compactness in the embedding space.

    Paper: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    (Wen et al., 2016) https://ydwen.github.io/papers/WenECCV16.pdf

    Formula: L_center = (1/2) * sum_i ||x_i - c_{y_i}||^2
    where c_{y_i} is the center of class y_i
    """

    def __init__(self, num_classes: int, embedding_size: int = None):
        """Initialize Center Loss.

        Args:
            num_classes: Number of classes in the dataset
            embedding_size: Dimension of the embedding vectors (optional, auto-detected if None)
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.centers = None  # Will be initialized on first forward pass if embedding_size is None

        # Initialize centers if embedding_size is provided
        if embedding_size is not None:
            self.centers = nn.Parameter(
                torch.randn(num_classes, embedding_size)
            )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Center Loss with normalized embeddings.

        Args:
            embeddings: Feature embeddings of shape (batch_size, embedding_size)
                       Expected to be L2-normalized (done in engine.py)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Center loss value (scalar)
        """
        batch_size = embeddings.size(0)

        # Embeddings are already normalized in engine.py before being passed here
        # (same normalization used by SupCon/Triplet for consistency)

        # Get centers for each sample based on their labels
        # Ensure centers are on the same device as labels
        centers_batch = self.centers.to(labels.device)[labels]  # (batch_size, embedding_size)

        # Normalize centers to unit sphere (compatible with normalized embeddings)
        centers_norm = F.normalize(centers_batch, p=2, dim=1)

        # Compute squared L2 distance (in range [0, 4] since both normalized)
        loss = ((embeddings - centers_norm) ** 2).sum() / (2.0 * batch_size)

        return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in classification tasks.

    Focal Loss applies a modulating term to the cross entropy loss to focus
    learning on hard misclassified examples. Useful for imbalanced datasets.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Formula: FL(p_t) = -α(1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss with specified parameters.

        Args:
            alpha: Weighting factor in [0, 1] to balance positive vs negative examples,
                   or a list of weights for each class.
            gamma: Exponent of the modulating factor (1 - p_t)^gamma.
            reduction: Specifies the reduction to apply ('none', 'mean', or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss for given inputs and targets.

        Args:
            inputs: Model logits of shape (N, C) where N is batch size and C is number of classes.
            targets: Ground truth labels of shape (N,) containing class indices.

        Returns:
            Computed loss value (scalar if reduction is 'mean' or 'sum', otherwise shape (N,)).
        """
        p = F.softmax(inputs, dim=1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.gather(p, 1, targets.view(-1, 1)).squeeze(1)

        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss():
    """Load loss function based on configuration settings.

    Reads the loss configuration and returns the appropriate loss function
    with parameters specified in the config file.

    Config options:
        loss.selected: "ce" (Cross-Entropy) or "focalLoss" (Focal Loss)
        loss.focal_loss.alpha: Weighting factor (default: 0.25)
        loss.focal_loss.gamma: Focusing parameter (default: 2.0)

    Returns:
        torch.nn.Module: Configured loss function ready for training.

    Raises:
        ValueError: If unknown loss function is specified in config.
    """
    loss_selected = get_config("loss", "selected")

    if loss_selected == "focalLoss":
        focal_params = get_config("loss", "focalLoss") or {}
        alpha = float(focal_params.get("alpha", 0.25))
        gamma = float(focal_params.get("gamma", 2.0))

        log_info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_selected == "ce":
        ce_params = get_config("loss", "ce") or {}
        label_smoothing = float(ce_params.get("label_smoothing", 0.0))
        if label_smoothing > 0:
            log_info(f"Using Cross-Entropy Loss (label_smoothing={label_smoothing})")
        else:
            log_info("Using Cross-Entropy Loss")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    else:
        raise ValueError(f"Unknown loss function: {loss_selected}. Choose 'ce' or 'focalLoss'")


def setup_contrastive_learning():
    """Setup contrastive learning components based on config.

    Supports two modes:
    1. Legacy mode (contrastive.selected): Single loss function
    2. Multi-loss mode (contrastive.losses.*): Multiple losses combined

    Legacy mode (deprecated):
        contrastive.selected: "supcon", "triplet", or null

    Multi-loss mode (recommended):
        contrastive.losses.supcon.enabled: true/false
        contrastive.losses.triplet.enabled: true/false
        contrastive.losses.center.enabled: true/false

    Returns:
        dict: Dictionary with structure:
            {
                'densecl': None,
                'supcon': {'loss_func': SupConLoss, 'weight': 0.4, 'miner': Miner} or None,
                'triplet': {'loss_func': TripletLoss, 'weight': 0.3, 'miner': Miner} or None,
                'center': {'loss_func': CenterLoss, 'weight': 0.3, 'miner': None} or None
            }
        Returns None if no contrastive learning configured

    Raises:
        ValueError: If unknown contrastive mode is specified
    """
    contrastive_config = get_config("contrastive")

    if not contrastive_config:
        log_info("No contrastive learning configured")
        return None

    # Check if using legacy single-loss mode
    selected = contrastive_config.get("selected") if isinstance(contrastive_config, dict) else contrastive_config

    if selected is not None:
        # Two-stage training mode
        log_info(f"Two-stage training enabled: {selected.upper()}")
        log_info("  Stage 1: Contrastive learning (backbone)")
        log_info("  Stage 2: Classification (frozen backbone + head)")

        miner, loss_func = _setup_single_loss(selected)

        # Check if valid loss function
        if loss_func is None:
            log_warning(f"Invalid contrastive mode: {selected}")
            return None

        loss_type = selected.lower()
        contrastive_weight = contrastive_config.get("contrastive_weight", 1.0)
        weight = float(contrastive_weight)

        return {
            'densecl': None,
            'supcon': {'loss_func': loss_func, 'weight': weight, 'miner': miner} if loss_type == 'supcon' else None,
            'triplet': {'loss_func': loss_func, 'weight': weight, 'miner': miner} if loss_type == 'triplet' else None,
            'center': {'loss_func': loss_func, 'weight': weight, 'miner': miner} if loss_type == 'center' else None
        }

    # Single-stage mode (selected == null)
    log_info("Single-stage training: Classification only")
    return None


def _setup_single_loss(contrastive_mode):
    """Setup single contrastive loss based on selected mode.

    Reads parameters from contrastive.losses.<mode> config section.

    Returns:
        tuple: (miner, contrastive_loss_func) or (None, None)
    """
    if contrastive_mode is None:
        log_info("Contrastive learning disabled (selected=None)")
        return None, None

    distance = distances.CosineSimilarity()
    mode_lower = contrastive_mode.lower()

    try:
        if mode_lower == "supcon":
            # Read from contrastive.losses.supcon
            supcon_config = get_config("contrastive", "losses", "supcon") or {}
            temperature = supcon_config.get("temperature", 0.07)
            miner_type = supcon_config.get("miner", "none")

            log_info("Supervised Contrastive Loss (SupConLoss)")
            log_info(f"- Temperature: {temperature}")
            log_info(f"- Miner type: {miner_type}")

            if miner_type != "none":
                miner = miners.TripletMarginMiner(
                    margin=0.2,
                    distance=distance,
                    type_of_triplets=miner_type
                )
            else:
                miner = None

            contrastive_loss_func = losses.SupConLoss(temperature=temperature)
            return miner, contrastive_loss_func

        elif mode_lower == "triplet":
            # Read from contrastive.losses.triplet
            triplet_config = get_config("contrastive", "losses", "triplet") or {}
            margin = triplet_config.get("margin", 0.2)
            miner_type = triplet_config.get("miner", "semi-hard")

            log_info("Triplet Loss with Hard Triplet Miner")
            log_info(f"- Margin: {margin}")
            log_info(f"- Miner type: {miner_type}")

            if miner_type != "none":
                miner = miners.TripletMarginMiner(
                    margin=margin,
                    distance=distance,
                    type_of_triplets=miner_type
                )
            else:
                miner = None

            contrastive_loss_func = losses.TripletMarginLoss(margin=margin, distance=distance)
            return miner, contrastive_loss_func

        elif mode_lower == "center":
            # Read from contrastive.losses.center
            center_config = get_config("contrastive", "losses", "center") or {}
            num_classes = center_config.get("num_classes", 2)
            embedding_size = center_config.get("embedding_size", 128)

            log_info("Center Loss (intra-class compactness)")
            log_info(f"- Num classes: {num_classes}")
            log_info(f"- Embedding size: {embedding_size}")

            contrastive_loss_func = CenterLoss(
                num_classes=num_classes,
                embedding_size=embedding_size
            )
            return None, contrastive_loss_func  # Center loss doesn't need miner

        else:
            raise ValueError(
                f"Unknown contrastive mode: '{contrastive_mode}'. "
                f"Supported modes: 'supcon', 'triplet', 'center'"
            )

    except Exception as e:
        log_info(f"Error setting up contrastive learning: {e}")
        raise
