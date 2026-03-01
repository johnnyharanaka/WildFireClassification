"""Generic multi-start Stage 1 utility for SimpleML models.

Any model with a ``fit_loop`` that wants to run multiple LR/WD trials can
import :func:`run_multi_start` and delegate Stage 1 orchestration to it.
"""

import copy
import inspect
import math
from typing import Any, Optional


def run_multi_start(
    model: Any,
    trainer: Any,
    epochs: int,
    best_metric_key: Optional[str],
) -> Optional[dict]:
    """Run multi-start Stage 1: try multiple LR/WD trials, return best weights.

    Reads ``trainer._cfg["multi_start"]`` for configuration. Returns ``None``
    when ``enabled`` is not ``True`` so the caller can fall back to its normal
    single-trial loop (backward compatible).

    TensorBoard prefixes follow the pattern ``stage1/trial{N}/train`` and
    ``stage1/trial{N}/val``.

    Args:
        model: The nn.Module being trained. Must already be in stage 1 mode.
        trainer: The Trainer instance (provides loaders, optimizer, scheduler,
            ``_train_one_epoch``, ``_validate_one_epoch``, ``_log_metrics``).
        epochs: Number of epochs to run per trial.
        best_metric_key: Metric name used to select the best-in-trial weights.
            When ``None`` or absent from val_metrics the last epoch's weights
            are kept as the trial result.

    Returns:
        State-dict of the best trial's model weights, or ``None`` if
        ``multi_start.enabled`` is not ``True``.
    """
    from simpleml.logger import log_info

    cfg = trainer._cfg
    ms_cfg = cfg.get("multi_start", {})
    if ms_cfg.get("enabled") is not True:
        return None

    patience: int = ms_cfg.get("patience", 4)
    min_threshold: float = ms_cfg.get("min_threshold", 0.05)

    # Build list of (lr, weight_decay) trials
    if "trials" in ms_cfg:
        trials = [(t["lr"], t.get("weight_decay")) for t in ms_cfg["trials"]]
    elif "learning_rates" in ms_cfg:
        trials = [(lr, None) for lr in ms_cfg["learning_rates"]]
    else:
        trials = [(1e-5, None), (1e-4, None), (1e-3, None)]

    # Header
    log_info(f"MULTI-START Stage 1: {len(trials)} trial(s), {epochs} epoch(s) each")
    log_info(f"  patience={patience}, min_threshold={min_threshold}")
    for i, (lr, wd) in enumerate(trials, 1):
        wd_str = f", wd={wd}" if wd is not None else ""
        log_info(f"  Trial {i}: lr={lr}{wd_str}")
    log_info("=" * 60)

    # Save initial weights to reset the model before each trial
    initial_weights = copy.deepcopy(model.state_dict())

    # Extract base optimizer class and instantiation kwargs
    opt_cls = type(trainer.optimizer)
    valid_opt_keys = (
        set(inspect.signature(opt_cls.__init__).parameters) - {"self", "params"}
    )
    base_opt_kwargs = {
        k: v for k, v in trainer.optimizer.defaults.items() if k in valid_opt_keys
    }

    # Extract scheduler class and kwargs (if a scheduler is active)
    orig_scheduler = trainer.scheduler
    sched_cls = None
    base_sched_kwargs: dict = {}
    if orig_scheduler is not None:
        sched_cls = type(orig_scheduler)
        skip = {"self", "optimizer", "last_epoch", "verbose"}
        base_sched_kwargs = {
            name: getattr(orig_scheduler, name)
            for name in inspect.signature(sched_cls.__init__).parameters
            if name not in skip and hasattr(orig_scheduler, name)
        }

    # Stash originals so we can restore them after all trials
    orig_optimizer = trainer.optimizer

    # Multi-start loop
    global_best_metric = -math.inf
    global_best_weights: Optional[dict] = None

    for trial_idx, (lr, wd) in enumerate(trials):
        trial_num = trial_idx + 1
        wd_str = f", wd={wd}" if wd is not None else ""
        log_info(f"--- Trial {trial_num}/{len(trials)} | lr={lr}{wd_str} ---")

        model.load_state_dict(copy.deepcopy(initial_weights))

        # Build trial optimizer (override lr and optionally weight_decay)
        trial_opt_kwargs = dict(base_opt_kwargs)
        trial_opt_kwargs["lr"] = lr
        if wd is not None:
            trial_opt_kwargs["weight_decay"] = wd
        trial_optimizer = opt_cls(model.parameters(), **trial_opt_kwargs)
        trainer.optimizer = trial_optimizer

        # Build trial scheduler linked to the trial optimizer
        if sched_cls is not None:
            trial_scheduler = sched_cls(trial_optimizer, **base_sched_kwargs)
            trainer.scheduler = trial_scheduler
        else:
            trainer.scheduler = None

        trial_best_metric = -math.inf
        trial_best_weights = copy.deepcopy(model.state_dict())
        no_improve_count = 0

        prefix_train = f"stage1/trial{trial_num}/train"
        prefix_val = f"stage1/trial{trial_num}/val"

        for epoch in range(epochs):
            train_loss = trainer._train_one_epoch(epoch)
            trainer._log_metrics({"loss": train_loss}, epoch, prefix=prefix_train)

            if trainer.scheduler is not None:
                trainer.scheduler.step()

            val_metric_value: Optional[float] = None
            if trainer.val_loader is not None:
                val_result = trainer._validate_one_epoch(epoch)
                val_loss = val_result["loss"]
                val_metrics = val_result.get("metrics", {})
                trainer._log_metrics({"loss": val_loss}, epoch, prefix=prefix_val)
                trainer._log_metrics(val_metrics, epoch, prefix=prefix_val)

                summary = {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                }
                summary.update({k: f"{v:.4f}" for k, v in val_metrics.items()})
                log_info(
                    f"  ep{epoch + 1}  "
                    + "  ".join(f"{k}: {v}" for k, v in summary.items())
                )

                if best_metric_key and best_metric_key in val_metrics:
                    val_metric_value = val_metrics[best_metric_key]
            else:
                log_info(f"  ep{epoch + 1}  train_loss: {train_loss:.4f}")

            # Track best within this trial
            if val_metric_value is not None:
                improved = val_metric_value > trial_best_metric + min_threshold
                if improved:
                    trial_best_metric = val_metric_value
                    trial_best_weights = copy.deepcopy(model.state_dict())
                    no_improve_count = 0
                    log_info(
                        f"  ★ Trial best {best_metric_key}: {trial_best_metric:.4f}"
                    )
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        log_info(f"  Early stop (patience={patience})")
                        break
            else:
                # No metric available: keep the latest epoch's weights
                trial_best_weights = copy.deepcopy(model.state_dict())

        log_info(
            f"  Trial {trial_num} best"
            + (f" {best_metric_key}: {trial_best_metric:.4f}" if best_metric_key else "")
        )

        # Update global best
        if global_best_weights is None or trial_best_metric >= global_best_metric:
            global_best_metric = trial_best_metric
            global_best_weights = copy.deepcopy(trial_best_weights)
            if trial_best_metric > -math.inf:
                log_info(
                    f"  *** New global best: {global_best_metric:.4f} (Trial {trial_num})"
                )

    # Restore original optimizer and scheduler
    trainer.optimizer = orig_optimizer
    trainer.scheduler = orig_scheduler

    # Summary
    log_info("=" * 60)
    if best_metric_key and global_best_metric > -math.inf:
        log_info(
            f"Multi-start complete. Global best {best_metric_key}: {global_best_metric:.4f}"
        )
    else:
        log_info("Multi-start complete.")
    log_info("=" * 60)

    return global_best_weights
