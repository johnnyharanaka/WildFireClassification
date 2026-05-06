#!/usr/bin/env python3
"""
Multi-seed training and evaluation script.

Trains the model with multiple seeds and reports mean ± std for all metrics.

Usage:
    uv run run_multi_seed.py
    uv run run_multi_seed.py --seeds 42 123 456 789 1234
    uv run run_multi_seed.py --model-name DinoV2RS_Small --seeds 42 123 456
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import argparse
import os
import pty
import random
import re
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path


def parse_eval_output(output: str) -> dict:
    """Parse evaluation output and extract metrics."""
    metrics = {
        # Classification
        'tp': None, 'tn': None, 'fp': None, 'fn': None,
        'accuracy': None, 'error_rate': None,
        'precision': None, 'recall': None, 'fdr': None, 'f1_score': None,
        # Detection
        'tp_det': None, 'fp_det': None, 'fn_det': None,
        'precision_det': None, 'recall_det': None,
        'corloc': None, 'map': None,
        # Val mAP from Stage 1 training (set externally after train_with_seed)
        'val_map': None,
    }

    # Parse Classification TP, TN, FP, FN
    conf_match = re.search(r'TP:\s*(\d+),\s*TN:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)', output)
    if conf_match:
        metrics['tp'] = int(conf_match.group(1))
        metrics['tn'] = int(conf_match.group(2))
        metrics['fp'] = int(conf_match.group(3))
        metrics['fn'] = int(conf_match.group(4))

    # Parse Classification metrics
    patterns = {
        'accuracy': r'Accuracy:\s*([\d.]+)',
        'error_rate': r'Error Rate \(ER\):\s*([\d.]+)',
        'precision': r'(?<!_det:\s)Precision:\s*([\d.]+)',
        'recall': r'(?<!_det:\s)Recall:\s*([\d.]+)',
        'fdr': r'FDR \(False Discovery Rate\):\s*([\d.]+)',
        'f1_score': r'F1-score:\s*([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    # Parse Detection TP_det, FP_det, FN_det
    det_conf_match = re.search(r'TP_det:\s*(\d+),\s*FP_det:\s*(\d+),\s*FN_det:\s*(\d+)', output)
    if det_conf_match:
        metrics['tp_det'] = int(det_conf_match.group(1))
        metrics['fp_det'] = int(det_conf_match.group(2))
        metrics['fn_det'] = int(det_conf_match.group(3))

    # Parse Detection precision/recall
    prec_det_match = re.search(r'Precision_det:\s*([\d.]+)', output)
    if prec_det_match:
        metrics['precision_det'] = float(prec_det_match.group(1))

    recall_det_match = re.search(r'Recall_det:\s*([\d.]+)', output)
    if recall_det_match:
        metrics['recall_det'] = float(recall_det_match.group(1))

    # Parse CorLoc and mAP
    corloc_match = re.search(r'CorLoc\s*\([^)]+\):\s*([\d.]+)', output)
    if corloc_match:
        metrics['corloc'] = float(corloc_match.group(1))

    map_match = re.search(r'mAP@IoU[\d.]+:\s*([\d.]+)', output)
    if map_match:
        metrics['map'] = float(map_match.group(1))

    return metrics


def parse_val_map_from_training(output: str) -> float | None:
    """Extract best val mAP from Stage 1 training output."""
    # Pattern: "Best mAP: X.XXXX at epoch Y"
    match = re.search(r'Best mAP:\s*([\d.]+)\s+at epoch', output)
    if match:
        return float(match.group(1))
    return None


def train_with_seed(seed: int, model_name: str) -> tuple[bool, float | None]:
    """Train model with a specific seed. Returns (success, val_map)."""
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*60}\n")

    # Use PTY so tqdm thinks it's in a real terminal (progress bar works)
    # while still capturing all output for parsing
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        ["uv", "run", "main.py", "--run", "--seed", str(seed)],
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    output_parts = []
    while True:
        try:
            data = os.read(master_fd, 4096)
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            output_parts.append(data.decode('utf-8', errors='replace'))
        except OSError:
            break

    process.wait()
    os.close(master_fd)
    output = ''.join(output_parts)

    if process.returncode != 0:
        print(f"Training failed for seed {seed}")
        return False, None

    val_map = parse_val_map_from_training(output)
    if val_map is not None:
        print(f"  [Val mAP Stage 1] Best val mAP: {val_map:.4f}")

    # Rename the saved model to include seed
    models_dir = Path("models")
    original_path = models_dir / f"{model_name}.pth"
    seed_path = models_dir / f"{model_name}_seed_{seed}.pth"

    if original_path.exists():
        shutil.copy(original_path, seed_path)
        print(f"Model saved as: {seed_path}")

    return True, val_map


def evaluate_seed(seed: int, model_name: str, iou_threshold: float = 0.5, loc_threshold: float = 0.5) -> dict:
    """Evaluate model trained with a specific seed."""
    models_dir = Path("models")
    seed_path = models_dir / f"{model_name}_seed_{seed}.pth"
    original_path = models_dir / f"{model_name}.pth"

    # Temporarily copy seed model to original path for evaluation
    backup_path = None
    if original_path.exists():
        backup_path = models_dir / f"{model_name}.pth.backup"
        shutil.copy(original_path, backup_path)

    shutil.copy(seed_path, original_path)

    try:
        result = subprocess.run(
            ["uv", "run", "main.py", "--eval-wsod",
             "--model-name", model_name,
             "--iou-threshold", str(iou_threshold),
             "--loc-threshold", str(loc_threshold),
             "--seed", str(seed)],
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr
        metrics = parse_eval_output(output)
        metrics['seed'] = seed

        if result.returncode != 0:
            print(f"  [WARN] eval-wsod exited with code {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")

        return metrics

    finally:
        # Restore original model
        if backup_path and backup_path.exists():
            shutil.move(backup_path, original_path)


def print_individual_results(all_results: list, model_name: str, iou_threshold: float):
    """Print results for each seed."""
    print(f"\n{'='*110}")
    print(f"INDIVIDUAL RESULTS - {model_name}")
    print(f"{'='*110}")

    # Classification table
    print(f"\nCLASSIFICATION METRICS")
    print("-"*110)
    print(f"{'Seed':<8} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Acc':<10} {'ER':<10} {'Prec':<10} {'Recall':<10} {'FDR':<10} {'F1':<10}")
    print("-"*110)

    for r in all_results:
        seed = r.get('seed', '?')
        tp = r['tp'] if r['tp'] is not None else 'N/A'
        tn = r['tn'] if r['tn'] is not None else 'N/A'
        fp = r['fp'] if r['fp'] is not None else 'N/A'
        fn = r['fn'] if r['fn'] is not None else 'N/A'
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else 'N/A'
        er = f"{r['error_rate']:.4f}" if r['error_rate'] is not None else 'N/A'
        prec = f"{r['precision']:.4f}" if r['precision'] is not None else 'N/A'
        rec = f"{r['recall']:.4f}" if r['recall'] is not None else 'N/A'
        fdr = f"{r['fdr']:.4f}" if r['fdr'] is not None else 'N/A'
        f1 = f"{r['f1_score']:.4f}" if r['f1_score'] is not None else 'N/A'

        print(f"{seed:<8} {tp:<6} {tn:<6} {fp:<6} {fn:<6} {acc:<10} {er:<10} {prec:<10} {rec:<10} {fdr:<10} {f1:<10}")

    # Detection table
    print(f"\nDETECTION METRICS (IoU >= {iou_threshold})")
    print("-"*115)
    print(f"{'Seed':<8} {'TP_det':<8} {'FP_det':<8} {'FN_det':<8} {'Prec_det':<12} {'Recall_det':<12} {'CorLoc':<12} {'mAP (test)':<14} {'mAP (val S1)':<14}")
    print("-"*115)

    for r in all_results:
        seed = r.get('seed', '?')
        tp_det = r['tp_det'] if r['tp_det'] is not None else 'N/A'
        fp_det = r['fp_det'] if r['fp_det'] is not None else 'N/A'
        fn_det = r['fn_det'] if r['fn_det'] is not None else 'N/A'
        prec_det = f"{r['precision_det']:.4f}" if r['precision_det'] is not None else 'N/A'
        rec_det = f"{r['recall_det']:.4f}" if r['recall_det'] is not None else 'N/A'
        corloc = f"{r['corloc']:.4f}" if r['corloc'] is not None else 'N/A'
        map_val = f"{r['map']:.4f}" if r['map'] is not None else 'N/A'
        val_map = f"{r['val_map']:.4f}" if r.get('val_map') is not None else 'N/A'

        print(f"{seed:<8} {tp_det:<8} {fp_det:<8} {fn_det:<8} {prec_det:<12} {rec_det:<12} {corloc:<12} {map_val:<14} {val_map:<14}")


def print_summary_statistics(all_results: list, model_name: str, iou_threshold: float):
    """Print mean ± std for all metrics."""
    print(f"\n{'='*110}")
    print(f"SUMMARY STATISTICS - {model_name} (n={len(all_results)} seeds)")
    print(f"{'='*110}")

    # Calculate statistics for each metric
    def calc_stats(key):
        values = [r[key] for r in all_results if r[key] is not None]
        if len(values) == 0:
            return None, None
        return np.mean(values), np.std(values)

    # Classification metrics
    print(f"\nCLASSIFICATION METRICS")
    print("-"*100)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Mean ± Std':<30}")
    print("-"*100)

    # TP, TN, FP, FN
    conf_metrics = ['tp', 'tn', 'fp', 'fn']
    conf_labels = ['TP', 'TN', 'FP', 'FN']

    for metric, label in zip(conf_metrics, conf_labels):
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.1f} {std:<15.1f} {mean:.1f} ± {std:.1f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A':<30}")

    print("-"*100)

    # Other classification metrics
    cls_metrics = ['accuracy', 'error_rate', 'precision', 'recall', 'fdr', 'f1_score']
    cls_labels = ['Accuracy', 'Error Rate', 'Precision', 'Recall', 'FDR', 'F1-score']

    for metric, label in zip(cls_metrics, cls_labels):
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.4f} {std:<15.4f} {mean:.4f} ± {std:.4f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A':<30}")

    # Detection metrics
    print(f"\nDETECTION METRICS (IoU >= {iou_threshold})")
    print("-"*100)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Mean ± Std':<30}")
    print("-"*100)

    # TP_det, FP_det, FN_det
    det_conf_metrics = ['tp_det', 'fp_det', 'fn_det']
    det_conf_labels = ['TP_det', 'FP_det', 'FN_det']

    for metric, label in zip(det_conf_metrics, det_conf_labels):
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.1f} {std:<15.1f} {mean:.1f} ± {std:.1f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A':<30}")

    print("-"*100)

    # Other detection metrics
    det_metrics = ['precision_det', 'recall_det', 'corloc', 'map', 'val_map']
    det_labels = ['Precision_det', 'Recall_det', 'CorLoc', 'mAP (test)', 'mAP (val S1)']

    for metric, label in zip(det_metrics, det_labels):
        mean, std = calc_stats(metric)
        if mean is not None:
            print(f"{label:<20} {mean:<15.4f} {std:<15.4f} {mean:.4f} ± {std:.4f}")
        else:
            print(f"{label:<20} {'N/A':<15} {'N/A':<15} {'N/A':<30}")

    print(f"\n{'='*110}")

    # Print in paper-ready format
    print(f"\nPAPER-READY FORMAT:")
    print("-"*60)

    acc_mean, acc_std = calc_stats('accuracy')
    f1_mean, f1_std = calc_stats('f1_score')
    map_mean, map_std = calc_stats('map')
    val_map_mean, val_map_std = calc_stats('val_map')
    corloc_mean, corloc_std = calc_stats('corloc')

    if acc_mean is not None:
        print(f"Accuracy:  {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    if f1_mean is not None:
        print(f"F1-score:  {f1_mean:.4f} ± {f1_std:.4f}")
    if map_mean is not None:
        print(f"mAP@{iou_threshold} (test):  {map_mean:.4f} ± {map_std:.4f}")
    if val_map_mean is not None:
        print(f"mAP@{iou_threshold} (val S1): {val_map_mean:.4f} ± {val_map_std:.4f}")
    if corloc_mean is not None:
        print(f"CorLoc:    {corloc_mean:.4f} ± {corloc_std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training and evaluation")
    parser.add_argument("--run", type=int, default=1,
                        help="Number of runs with random seeds (default: 1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of specific seeds to use for multi-seed experiment")
    parser.add_argument("--model-name", type=str, default="DinoV2RS_Small",
                        help="Model name (default: DinoV2RS_Small)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for detection metrics (default: 0.5)")
    parser.add_argument("--loc-threshold", type=float, default=0.5,
                        help="Localization threshold (default: 0.5)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing models")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train, skip evaluation")

    args = parser.parse_args()

    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = [random.randint(0, 2**31 - 1) for _ in range(args.run)]
        print(f"Using random seeds: {seeds}")

    print(f"\n{'#'*60}")
    print(f"MULTI-SEED EXPERIMENT")
    print(f"{'#'*60}")
    print(f"Model: {args.model_name}")
    print(f"Seeds: {seeds}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"{'#'*60}\n")

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Processing seed {seed}...")

        # Train
        val_map = None
        if not args.eval_only:
            success, val_map = train_with_seed(seed, args.model_name)
            if not success:
                print(f"Skipping seed {seed} due to training failure")
                continue

        # Evaluate
        if not args.train_only:
            seed_model_path = Path("models") / f"{args.model_name}_seed_{seed}.pth"
            if not seed_model_path.exists():
                print(f"Model not found for seed {seed}: {seed_model_path}")
                continue

            metrics = evaluate_seed(seed, args.model_name, args.iou_threshold, args.loc_threshold)
            metrics['val_map'] = val_map
            all_results.append(metrics)
            acc_str = f"{metrics['accuracy']:.4f}" if metrics['accuracy'] is not None else "N/A"
            f1_str = f"{metrics['f1_score']:.4f}" if metrics['f1_score'] is not None else "N/A"
            map_str = f"{metrics['map']:.4f}" if metrics['map'] is not None else "N/A"
            val_map_str = f"{val_map:.4f}" if val_map is not None else "N/A"
            print(f"  Seed {seed}: Acc={acc_str}, F1={f1_str}, mAP={map_str}, val_mAP={val_map_str}")

    # Print results
    if all_results and not args.train_only:
        print_individual_results(all_results, args.model_name, args.iou_threshold)
        print_summary_statistics(all_results, args.model_name, args.iou_threshold)


if __name__ == "__main__":
    main()