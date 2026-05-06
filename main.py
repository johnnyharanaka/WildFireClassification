import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import gc
import random
import torch
import argparse
import numpy as np

from core import train, vis, list_all_models, load_model, test_model, embedding_vis, check_images, visualize_embeddings_from_json
from core import get_device, get_model, add_attn_maps
from core.debug import process_multiple_checkpoints
from core.config.config import init_results, get_config, get_active_dataset_config, set_root_path, set_global_seed
from core.config.config import log_info
from core.metrics import evaluate_wsod

def set_seed(seed: int = 42):
    set_global_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)



def environment_setup(seed: int = 42):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    set_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a deep learning model.")

    parser.add_argument("--train", action="store_true", help="Train a model")
    parser.add_argument("--models", action="store_true", help="Visualize list of models")
    parser.add_argument("--img-only", action="store_true", help="Visualize image only")
    parser.add_argument("--vis", action="store_true", help="Visualiaze BBs")
    parser.add_argument("--savefile", action="store_true", help="Save Files")
    parser.add_argument("--embeddings", action="store_true", help="visualize embeddings")
    parser.add_argument("--check-images", action="store_true", help="Check Images")
    parser.add_argument("--test", action="store_true", help="Test COCO")
    parser.add_argument("--run", action="store_true", help="Run all models")
    parser.add_argument("--debug-bbox", action="store_true", help="Save bbox visualizations for debugging")
    parser.add_argument("--num-bbox-samples", type=int, default=40,
                        help="Number of samples for bbox visualization (default: 20)")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"],
                        help="Dataset split to use for debug-bbox (default: Test)")
    parser.add_argument("--vis-emb", type=str, nargs='?', const='bbox_debug/embeddings.json',
                        help="Visualize embeddings from JSON file (default: bbox_debug/embeddings.json)")
    parser.add_argument("--eval-wsod", action="store_true", help="Evaluate model with WSOD metrics (CorLoc + mAP)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for mAP calculation")
    parser.add_argument("--loc-threshold", type=float, default=0.5, help="LocAcc threshold for CorLoc calculation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--model-name", type=str, default=None, help="Model name (if None, uses active model from config.yaml)")
    parser.add_argument("--filename", type=str, default="", help="filename")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (if None, uses active dataset config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps). If None, auto-detect")
    parser.add_argument("--root-path", type=str, default=None, help="Root path for dataset (overrides config.yaml if set)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last saved checkpoint (auto-detects latest ep_N.pth)")

    args = parser.parse_args()

    # Handle embedding visualization early (doesn't need model setup)
    if args.vis_emb is not None:
        visualize_embeddings_from_json(args.vis_emb)
        return

    environment_setup(args.seed)

    if args.root_path:
        set_root_path(args.root_path)
        log_info(f"Dataset path: {args.root_path}")

    if args.device:
        device = torch.device(args.device)
        log_info(f"Using device: {device}")
    else:
        device = get_device()

    if args.num_classes is None:
        args.num_classes = get_active_dataset_config('num_classes')
        dataset_name = get_active_dataset_config('name')
        log_info(f"Using active dataset: {dataset_name} with {args.num_classes} classes")

    if args.model_name is None:
        run_models = get_config("defaults", "run_models")
        args.model_name = run_models[0] if run_models else "DinoV2_Small"
        log_info(f"Using model from config.yaml: {args.model_name}")

    config_batch_size = get_config("defaults", "batch_size")
    if config_batch_size is not None:
        args.batch_size = config_batch_size

    if args.run:
        init_results()
        models = get_config("defaults", "run_models")
        for model_name in models:
            model = get_model(model_name, num_classes=args.num_classes, device=device)
            train(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                model=model,
                device=device,
                model_name=model_name,
                num_classes=args.num_classes,
                resume_epoch=args.resume,
            )
    elif args.eval_wsod:
        model = get_model(args.model_name, num_classes=args.num_classes, device=device)
        load_model(model, args.model_name)

        if "Dino" not in args.model_name:
            model = add_attn_maps(model)

        evaluate_wsod(
            model=model,
            model_name=args.model_name,
            device=device,
            num_workers=args.num_workers,
            num_classes=args.num_classes,
            iou_thresholds=[args.iou_threshold],
            loc_threshold=args.loc_threshold,
            batch_size=args.batch_size,
        )
    else:
        model_name = args.model_name
        model = get_model(args.model_name, num_classes=args.num_classes, device=device)
        if args.train:
            train(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                model=model,
                device=device,
                model_name=model_name,
                num_classes=args.num_classes,
                resume_epoch=args.resume,
            )
        elif args.models:
            list_all_models()
        elif args.check_images:
            check_images(args.filename)
        else:
            load_model(model, model_name)
            if "Dino" not in model_name:
                model = add_attn_maps(model)

            if args.vis or args.savefile:
                vis(model=model, model_name=model_name, device=device, save_file=args.savefile)
            elif args.test:
                eval_metric = get_active_dataset_config("eval_metric")
                test_model(model=model, model_name=model_name, device=device, num_workers=args.num_workers, metric=eval_metric)
            elif args.embeddings:
                embedding_vis(model=model, model_name=model_name, device=device)
            elif args.debug_bbox:
                process_multiple_checkpoints(
                    get_model_fn=get_model,
                    model_name=model_name,
                    num_classes=args.num_classes,
                    device=device,
                    num_workers=args.num_workers,
                    num_samples=args.num_bbox_samples,
                    models_dir="models",
                    add_attn_maps_fn=add_attn_maps,
                    split=args.split
                )
        return

if __name__ == "__main__":
    main()


