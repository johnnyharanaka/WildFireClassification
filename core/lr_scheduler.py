"""Learning rate schedulers and training utilities.

Provides optimizers, learning rate schedulers, and early stopping mechanisms
for training deep learning models.
"""
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from core.config.config import get_config, log_info


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """Create optimizer based on configuration settings.

    Supports AdamW and SGD optimizers with parameters loaded from config.

    Args:
        model: PyTorch model whose parameters will be optimized.

    Returns:
        Configured optimizer instance (AdamW or SGD).

    Config options:
        optimizers.selected: "adamw" or "sgd"
        optimizers.adamw.lr: Learning rate for AdamW
        optimizers.adamw.betas: Beta coefficients for AdamW
        optimizers.adamw.eps: Epsilon for numerical stability
        optimizers.adamw.weight_decay: Weight decay coefficient
        optimizers.sgd.lr: Learning rate for SGD
        optimizers.sgd.momentum: Momentum factor
        optimizers.sgd.weight_decay: Weight decay coefficient
        optimizers.sgd.nesterov: Whether to use Nesterov momentum
    """
    optimizer_selected = get_config("optimizers", "selected")

    if optimizer_selected == "adamw":
        opt_params = get_config("optimizers", "adamw")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(opt_params["lr"]),
            betas=(float(opt_params["betas"][0]), float(opt_params["betas"][1])),
            eps=float(opt_params["eps"]),
            weight_decay=float(opt_params["weight_decay"]),
        )
        return optimizer
    else:
        opt_params = get_config("optimizers", "sgd")
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(opt_params["lr"]),
            momentum=float(opt_params["momentum"]),
            weight_decay=float(opt_params["weight_decay"]),
            nesterov=bool(opt_params["nesterov"]),
        )
        return optimizer


def get_optimizer_for_params(parameters):
    """
    Create optimizer for specific parameter list (used in stage 2).

    Args:
        parameters: List of parameters to optimize

    Returns:
        Optimizer configured from config.yaml
    """
    optimizer_selected = get_config("optimizers", "selected")

    if optimizer_selected == "adamw":
        opt_params = get_config("optimizers", "adamw")
        return optim.AdamW(
            parameters,
            lr=float(opt_params["lr"]),
            betas=(float(opt_params["betas"][0]), float(opt_params["betas"][1])),
            eps=float(opt_params["eps"]),
            weight_decay=float(opt_params["weight_decay"]),
        )
    else:  # SGD
        opt_params = get_config("optimizers", "sgd")
        return optim.SGD(
            parameters,
            lr=float(opt_params["lr"]),
            momentum=float(opt_params["momentum"]),
            weight_decay=float(opt_params["weight_decay"]),
            nesterov=bool(opt_params["nesterov"]),
        )


def get_scheduler(optimizer: optim.Optimizer, constant: bool = False) -> lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on configuration settings.

    Args:
        optimizer: PyTorch optimizer to schedule learning rate for.
        constant: If True, returns a no-op scheduler (constant LR). Used for Stage 2.

    Config options:
        scheduler.enabled: Whether to use CosineAnnealingLR (Stage 1 only)
        scheduler.T_max: Number of epochs for cosine decay
        scheduler.eta_min: Minimum learning rate
    """
    if constant:
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    scheduler_enabled = get_config("scheduler", "enabled")
    if scheduler_enabled is False:
        log_info("Scheduler disabled — using constant LR")
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    selected = get_config("scheduler", "selected") or "cosine_annealing"

    if selected == "warm_restarts":
        cfg = get_config("scheduler", "warm_restarts") or {}
        T_0 = int(cfg.get("T_0", 5))
        T_mult = int(cfg.get("T_mult", 1))
        eta_min = float(cfg.get("eta_min", 5e-7))
        log_info(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    else:
        cfg = get_config("scheduler", "cosine_annealing") or {}
        T_max = int(cfg.get("T_max", 8))
        eta_min = float(cfg.get("eta_min", 5e-7))
        log_info(f"Scheduler: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


def get_optimizer_with_lr(model: nn.Module, lr: float, weight_decay: float = None) -> optim.Optimizer:
    """Create optimizer from config but override the learning rate and optionally weight decay.

    Uses the same optimizer type and hyperparameters from config,
    but substitutes the learning rate (and weight decay if provided) with the given values.

    Args:
        model: PyTorch model whose parameters will be optimized.
        lr: Learning rate to use instead of the config value.
        weight_decay: Weight decay to use instead of the config value. If None, uses config value.

    Returns:
        Configured optimizer with custom learning rate.
    """
    optimizer_selected = get_config("optimizers", "selected")

    if optimizer_selected == "adamw":
        opt_params = get_config("optimizers", "adamw")
        wd = weight_decay if weight_decay is not None else float(opt_params["weight_decay"])
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(float(opt_params["betas"][0]), float(opt_params["betas"][1])),
            eps=float(opt_params["eps"]),
            weight_decay=wd,
        )
    else:
        opt_params = get_config("optimizers", "sgd")
        wd = weight_decay if weight_decay is not None else float(opt_params["weight_decay"])
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(opt_params["momentum"]),
            weight_decay=wd,
            nesterov=bool(opt_params["nesterov"]),
        )


def get_optimizer_for_params_with_lr(parameters, lr: float, weight_decay: float = None) -> optim.Optimizer:
    """Create optimizer for specific parameters with custom learning rate and optional weight decay.

    Uses the same optimizer type and hyperparameters from config,
    but applies them to the given parameter list with the specified LR/WD.

    Args:
        parameters: List of parameters to optimize.
        lr: Learning rate to use instead of the config value.
        weight_decay: Weight decay to use instead of the config value. If None, uses config value.

    Returns:
        Configured optimizer with custom learning rate for the given parameters.
    """
    optimizer_selected = get_config("optimizers", "selected")

    if optimizer_selected == "adamw":
        opt_params = get_config("optimizers", "adamw")
        wd = weight_decay if weight_decay is not None else float(opt_params["weight_decay"])
        return optim.AdamW(
            parameters,
            lr=lr,
            betas=(float(opt_params["betas"][0]), float(opt_params["betas"][1])),
            eps=float(opt_params["eps"]),
            weight_decay=wd,
        )
    else:
        opt_params = get_config("optimizers", "sgd")
        wd = weight_decay if weight_decay is not None else float(opt_params["weight_decay"])
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=float(opt_params["momentum"]),
            weight_decay=wd,
            nesterov=bool(opt_params["nesterov"]),
        )


class EarlyStopping:
    """Early stopping mechanism to prevent overfitting during training.

    Monitors validation loss and stops training when no improvement is observed
    for a specified number of epochs (patience).
    """

    def __init__(self, patience: int = 5, delta: float = 0, verbose: bool = False):
        """Initialize EarlyStopping with specified parameters.

        Args:
            patience: Number of epochs with no improvement before stopping.
            delta: Minimum change in monitored value to qualify as improvement.
            verbose: Whether to print messages when early stopping occurs.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss: float) -> None:
        """Check if training should stop based on validation loss.

        Updates internal state and sets stop_training flag if patience is exceeded.

        Args:
            val_loss: Current epoch's validation loss value.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    log_info("Stopping early as no improvement has been observed.")
