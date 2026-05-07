"""Compare embeddings from 2 model checkpoints via UMAP side-by-side plots.

Runs a grid search over n_neighbors and min_dist, saving all results to an output folder.
"""

import argparse
import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from scipy.cluster.hierarchy import fcluster, linkage

from core.config.config import get_active_dataset_config, get_config, get_root_path
from core.datasets import get_data_loaders
from core.models import get_model

CLUSTER_RANGE = range(3, 8)
NEIGHBORS_GRID = [20,30,50]
MIN_DIST_GRID = [0.2 ,0.25,0.5]
METRIC_GRID = ["euclidean"]
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="UMAP grid search comparison of 2 model checkpoints")
    parser.add_argument("--weights1", required=True, help="Path to checkpoint 1")
    parser.add_argument("--weights2", required=True, help="Path to checkpoint 2")
    parser.add_argument("--label1", default="CE", help="Label for plot 1")
    parser.add_argument("--label2", default="SupCon", help="Label for plot 2")
    parser.add_argument("--model-name", default="DinoV2RS_Small", help="Model name from Models enum")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--output-dir", default="umap_gridsearch", help="Output directory for images")
    return parser.parse_args()


def get_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract backbone cls_features and labels from a model on the test set."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        imgs, labels = batch[0].to(device), batch[1]
        features_dict = model.backbone.forward_features(imgs)
        embeddings = features_dict["x_norm_clstoken"]
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def plot_umap(t1, lab1, lbl1, t2, lab2, lbl2, classes, save_path):
    class_colors = ["#FF4B4B", "#4B9BFF"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, t, lab, title in [(ax1, t1, lab1, lbl1), (ax2, t2, lab2, lbl2)]:
        for cls_idx, cls_name in enumerate(classes):
            mask = lab == cls_idx
            ax.scatter(
                t[mask, 0], t[mask, 1],
                c=class_colors[cls_idx], label=cls_name,
                s=12, alpha=0.7, edgecolors="none",
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc="best")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_purity(emb1, lab1, lbl1, emb2, lab2, lbl2, classes, output_dir, n_clusters):
    """Hierarchical clustering on raw embeddings, stacked bar chart of cluster composition."""
    class_colors = ["#FF4B4B", "#4B9BFF"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, emb, lab, title in [(ax1, emb1, lab1, lbl1), (ax2, emb2, lab2, lbl2)]:
        Z = linkage(emb, method="ward")
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

        fire_counts = []
        not_counts = []
        cluster_ids = sorted(set(clusters))
        for c in cluster_ids:
            mask = clusters == c
            fire_counts.append(np.sum(lab[mask] == 0))
            not_counts.append(np.sum(lab[mask] == 1))

        x = np.arange(len(cluster_ids))
        ax.bar(x, fire_counts, color=class_colors[0], label=classes[0])
        ax.bar(x, not_counts, bottom=fire_counts, color=class_colors[1], label=classes[1])

        for j, c in enumerate(cluster_ids):
            total = fire_counts[j] + not_counts[j]
            purity = max(fire_counts[j], not_counts[j]) / total
            ax.text(j, total + 1, f"{purity:.0%}", ha="center", fontsize=9)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Samples")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in cluster_ids])
        ax.legend(loc="best")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"cluster_purity_k{n_clusters}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  k={n_clusters} -> {save_path}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load config
    num_classes = get_active_dataset_config("num_classes")
    classes = get_active_dataset_config("classes")
    model_cfg = get_config("models", args.model_name) or {}
    img_size = model_cfg.get("img_size", 224)
    root_path = get_root_path()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create test dataloader
    data_loaders, _ = get_data_loaders(
        root_path=root_path, batch_size=args.batch_size, img_size=img_size
    )
    test_loader = data_loaders["Test"]

    # Extract embeddings once for each checkpoint
    results = []
    for ckpt_path, label in [(args.weights1, args.label1), (args.weights2, args.label2)]:
        print(f"Extracting embeddings: {label} ({ckpt_path})")
        model = get_model(args.model_name, num_classes, device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        embeddings, labels = extract_embeddings(model, test_loader, device)
        results.append((embeddings, labels, label))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    emb1, lab1, lbl1 = results[0]
    emb2, lab2, lbl2 = results[1]
    combined = np.concatenate([emb1, emb2], axis=0)

    # Grid search
    combos = list(product(NEIGHBORS_GRID, MIN_DIST_GRID, METRIC_GRID))
    total = len(combos)
    disconnected = []
    for i, (n_neighbors, min_dist, metric) in enumerate(combos, 1):
        print(f"[{i}/{total}] n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed = reducer.fit_transform(combined)
            if any("not fully connected" in str(w.message) for w in caught):
                disconnected.append((n_neighbors, min_dist, metric))

        t1 = transformed[: len(emb1)]
        t2 = transformed[len(emb1):]

        filename = f"umap_nn{n_neighbors}_md{min_dist}_{metric}.png"
        save_path = os.path.join(args.output_dir, filename)
        plot_umap(t1, lab1, lbl1, t2, lab2, lbl2, classes, save_path)

    print(f"\nDone! {total} images saved to {args.output_dir}/")
    if disconnected:
        print(f"\n⚠ Graph disconnected in {len(disconnected)} combos (results may have artificial clusters):")
        for nn, md, mt in disconnected:
            print(f"  n_neighbors={nn}, min_dist={md}, metric={mt}")
    else:
        print("All combinations had fully connected graphs.")

    # Cluster purity analysis for each k in CLUSTER_RANGE
    print(f"\nCluster purity analysis (k={CLUSTER_RANGE.start}..{CLUSTER_RANGE.stop - 1}):")
    for k in CLUSTER_RANGE:
        plot_cluster_purity(emb1, lab1, lbl1, emb2, lab2, lbl2, classes, args.output_dir, k)


if __name__ == "__main__":
    main()
