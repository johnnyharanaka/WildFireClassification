#!/usr/bin/env python3
import os
import sys
import torch
import argparse
from typing import Optional, Dict, Any
from pathlib import Path
from main import environment_setup, set_seed
from core import (
    train, vis, list_all_models, load_model, test_model,
    embedding_vis, check_images, get_device, get_model, add_attn_maps
)
from core.debug import save_bbox_visualizations
from core.config.config import (
    write_results, init_results, get_config,
    get_active_dataset_config, set_root_path, log_info
)
from core.dino_classifier import DinoClassifier


class FireClassificationInterface:
    """
    A Python interface for FireClassification workflow.

    Usage:
        interface = FireClassificationInterface()
        interface.train_model(
            model_name="DinoV2_Small",
            epochs=12,
            batch_size=16
        )
    """

    def __init__(self, seed: int = 42, device: Optional[str] = None, root_path: Optional[str] = None):
        """Initialize the interface with environment setup."""
        self.seed = seed
        self.device_str = device

        # Setup environment
        environment_setup(seed)

        # Set device
        if device:
            self.device = torch.device(device)
            log_info(f"Using device: {self.device}")
        else:
            self.device = get_device()

        # Set root path if provided
        if root_path:
            set_root_path(root_path)
            log_info(f"Dataset path: {root_path}")

    def get_active_config(self) -> Dict[str, Any]:
        """Get active configuration from config.yaml."""
        return {
            "num_classes": get_active_dataset_config('num_classes'),
            "dataset_name": get_active_dataset_config('name'),
            "eval_metric": get_active_dataset_config('eval_metric'),
            "batch_size": get_config("defaults", "batch_size"),
            "epochs": get_config("defaults", "epochs"),
            "run_models": get_config("defaults", "run_models"),
        }

    def list_models(self) -> None:
        """List all available models."""
        log_info("\n=== Available Models ===")
        list_all_models()

    def train_model(
        self,
        model_name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        num_classes: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train a single model.

        Args:
            model_name: Name of model to train (uses config default if None)
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
            num_workers: Number of data loader workers
            num_classes: Number of classes (uses active dataset config if None)

        Returns:
            Dictionary with metrics: {"f1": float, "train_acc": float, "val_acc": float}
        """
        # Use defaults if not provided
        if num_classes is None:
            num_classes = get_active_dataset_config('num_classes')

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        if epochs is None:
            config_epochs = get_config("defaults", "epochs")
            epochs = config_epochs if config_epochs is not None else 12

        if batch_size is None:
            config_batch_size = get_config("defaults", "batch_size")
            batch_size = config_batch_size if config_batch_size is not None else 16

        log_info(f"\n=== Training {model_name} ===")
        log_info(f"Epochs: {epochs}, Batch Size: {batch_size}, Classes: {num_classes}")

        model = get_model(model_name, num_classes=num_classes, device=self.device)

        train_acc, val_acc = train(
            batch_size=batch_size,
            num_workers=num_workers,
            model=model,
            device=self.device,
            model_name=model_name,
            num_classes=num_classes,
        )

        return {"train_acc": train_acc, "val_acc": val_acc}

    def test_model(
        self,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        num_workers: int = 0,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Test a trained model.

        Args:
            model_name: Name of model to test
            num_classes: Number of classes
            num_workers: Number of data loader workers
            threshold: Detection threshold

        Returns:
            Dictionary with metrics: {"max_box_acc": float, "max_threshold": float}
        """
        if num_classes is None:
            num_classes = get_active_dataset_config('num_classes')

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        log_info(f"\n=== Testing {model_name} ===")

        model = get_model(model_name, num_classes=num_classes, device=self.device)
        load_model(model, model_name)

        if "Dino" not in model_name:
            model = add_attn_maps(model)

        eval_metric = get_active_dataset_config("eval_metric")
        max_box_acc, max_threshold = test_model(
            model=model,
            model_name=model_name,
            device=self.device,
            num_workers=num_workers,
            threshold=threshold,
            metric=eval_metric
        )

        return {"max_box_acc": max_box_acc, "max_threshold": max_threshold}

    def visualize_model(
        self,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        save_file: bool = False,
    ) -> None:
        """
        Visualize model predictions on test set.

        Args:
            model_name: Name of model to visualize
            num_classes: Number of classes
            save_file: Whether to save visualization
        """
        if num_classes is None:
            num_classes = get_active_dataset_config('num_classes')

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        log_info(f"\n=== Visualizing {model_name} ===")

        model = get_model(model_name, num_classes=num_classes, device=self.device)
        load_model(model, model_name)

        if "Dino" not in model_name:
            model = add_attn_maps(model)

        vis(model=model, model_name=model_name, device=self.device, save_file=save_file)

    def visualize_embeddings(
        self,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        return_for_notebook: bool = False,
    ):
        """
        Visualize model embeddings (t-SNE/UMAP).

        Args:
            model_name: Name of model
            num_classes: Number of classes
            return_for_notebook: If True, returns figures and data for notebook display

        Returns:
            If return_for_notebook=True, returns dict with train and test results containing figures and data
        """
        if num_classes is None:
            num_classes = get_active_dataset_config('num_classes')

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        log_info(f"\n=== Visualizing embeddings for {model_name} ===")

        model = get_model(model_name, num_classes=num_classes, device=self.device)
        load_model(model, model_name)

        return embedding_vis(model=model, model_name=model_name, device=self.device, return_for_notebook=return_for_notebook)

    def visualize_embeddings_from_json(
        self,
        model_name: Optional[str] = None,
        checkpoint: str = "best",
        return_for_notebook: bool = False,
    ):
        """
        Visualize embeddings from saved JSON files (generated by --debug-bbox).

        Args:
            model_name: Name of model (used to find the correct folder structure)
            checkpoint: Which checkpoint to visualize ('ep_1', 'ep_2', 'best', etc.)
            return_for_notebook: If True, returns figures for notebook display

        Returns:
            If return_for_notebook=True, returns dict with Plotly figures
        """
        from core.vis import visualize_embeddings_from_json

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        json_path = f"bbox_debug/{checkpoint}/embeddings.json"

        log_info(f"\n=== Visualizing embeddings from {json_path} ===")

        return visualize_embeddings_from_json(json_path=json_path, return_for_notebook=return_for_notebook)

    def visualize_all_checkpoints(
        self,
        model_name: Optional[str] = None,
        return_for_notebook: bool = False,
    ):
        """
        Visualize embeddings from all available checkpoints (ep_1, ep_2, ..., best).

        Args:
            model_name: Name of model
            return_for_notebook: If True, returns dict with all figures

        Returns:
            If return_for_notebook=True, returns dict mapping checkpoint names to figures
        """
        import glob
        from pathlib import Path
        from core.vis import visualize_embeddings_from_json

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        # Find all checkpoint folders
        bbox_debug_path = Path("bbox_debug")
        if not bbox_debug_path.exists():
            log_info("❌ bbox_debug folder not found. Run --debug-bbox first.")
            return None

        # Get all subdirectories (ep_1, ep_2, best, etc.)
        checkpoint_dirs = [d for d in bbox_debug_path.iterdir() if d.is_dir()]

        if len(checkpoint_dirs) == 0:
            log_info("❌ No checkpoint folders found in bbox_debug/")
            return None

        results = {}
        log_info(f"\n=== Visualizing {len(checkpoint_dirs)} checkpoint(s) ===")

        for checkpoint_dir in sorted(checkpoint_dirs):
            checkpoint_name = checkpoint_dir.name
            json_path = checkpoint_dir / "embeddings.json"

            if not json_path.exists():
                log_info(f"⚠ Skipping {checkpoint_name}: no embeddings.json found")
                continue

            log_info(f"\n📊 Processing: {checkpoint_name}")

            try:
                figures = visualize_embeddings_from_json(
                    json_path=str(json_path),
                    return_for_notebook=return_for_notebook
                )
                results[checkpoint_name] = figures
            except Exception as e:
                log_info(f"❌ Error processing {checkpoint_name}: {e}")

        log_info(f"\n✓ Processed {len(results)}/{len(checkpoint_dirs)} checkpoint(s)")

        if return_for_notebook:
            return results
        return None

    def check_dataset_images(self, filename: str) -> None:
        """Check and validate dataset images."""
        log_info(f"\n=== Checking images in {filename} ===")
        check_images(filename)

    def debug_bbox(
        self,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        num_workers: int = 0,
        save_visualizations: bool = True,
    ) -> None:
        """
        Generate or view debug bounding box visualizations.

        Args:
            model_name: Name of model to use
            num_classes: Number of classes
            num_workers: Number of data loader workers
            save_visualizations: If True, save to disk. If False, display temporarily.
        """
        import shutil
        from pathlib import Path

        if num_classes is None:
            num_classes = get_active_dataset_config('num_classes')

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        log_info(f"\n=== Debug BBox Visualization ===")
        log_info(f"Model: {model_name}")
        log_info(f"Classes: {num_classes}")
        log_info(f"Save to disk: {save_visualizations}")

        model = get_model(model_name, num_classes=num_classes, device=self.device)
        load_model(model, model_name)

        if "Dino" not in model_name:
            model = add_attn_maps(model)

        if save_visualizations:
            output_dir = 'bbox_debug'
        else:
            output_dir = 'bbox_debug_temp'

        save_bbox_visualizations(
            model=model,
            model_name=model_name,
            device=self.device,
            num_workers=num_workers,
            output_dir=output_dir
        )

        if save_visualizations:
            # Clean up any temporary visualizations first
            temp_dir = Path('bbox_debug_temp')
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            log_info(f"✓ Debug visualizations saved to bbox_debug/ directory")
        else:
            # Clean up temporary directory after display
            temp_dir = Path('bbox_debug_temp')
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            log_info(f"✓ Debug visualization completed (temporary display)")

    def evaluate_wsod(
        self,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        num_workers: int = 0,
        iou_threshold: float = 0.5,
        loc_threshold: float = 0.5,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """
        Evaluate WSOD model with CorLoc and mAP metrics.

        Args:
            model_name: Model name (uses config default if None)
            num_classes: Number of classes (uses config default if None)
            num_workers: Number of workers for dataloader
            iou_threshold: IoU threshold for mAP (default: 0.5)
            loc_threshold: LocAcc threshold for CorLoc (default: 0.5)
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with results:
            {
                'corloc': float,
                'map': float,
                'iou_threshold': float,
                'loc_threshold': float,
                'num_images': int,
                'num_classes': int,
                'model_name': str
            }
        """
        from core.metrics import evaluate_wsod

        if num_classes is None:
            num_classes = get_active_dataset_config('num_classes')

        if model_name is None:
            run_models = get_config("defaults", "run_models")
            model_name = run_models[0] if run_models else "DinoV2_Small"

        log_info(f"\n=== Evaluating WSOD Metrics ===")
        log_info(f"Model: {model_name}")
        log_info(f"Classes: {num_classes}")
        log_info(f"IoU Threshold: {iou_threshold}")
        log_info(f"LocAcc Threshold: {loc_threshold}")

        model = get_model(model_name, num_classes=num_classes, device=self.device)
        load_model(model, model_name)

        if "Dino" not in model_name:
            model = add_attn_maps(model)

        results = evaluate_wsod(
            model=model,
            model_name=model_name,
            device=self.device,
            num_workers=num_workers,
            num_classes=num_classes,
            iou_thresholds=[iou_threshold],
            loc_threshold=loc_threshold,
            batch_size=batch_size,
        )

        return results

    def evaluate_wsod_simple(
        self,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        num_workers: int = 0,
    ) -> Dict[str, float]:
        """
        Simplified WSOD evaluation with default values.

        Args:
            model_name: Model name
            num_classes: Number of classes
            num_workers: Number of workers

        Returns:
            Dictionary with CorLoc and mAP
        """
        return self.evaluate_wsod(
            model_name=model_name,
            num_classes=num_classes,
            num_workers=num_workers,
            iou_threshold=0.5,
            loc_threshold=0.5,
            batch_size=16,
        )


if __name__ == "__main__":
    # Example usage
    print("FireClassification Interface Example\n")

    # Initialize interface
    interface = FireClassificationInterface(seed=42)

    # Get configuration
    config = interface.get_active_config()
    print(f"Active Configuration: {config}\n")

    # List available models
    interface.list_models()