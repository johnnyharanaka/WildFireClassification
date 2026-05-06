"""Compare intra-class vs inter-class cosine similarity distributions.

Extracts embeddings from two model checkpoints (e.g. CE vs SupCon),
computes pairwise cosine similarities, and plots overlapping histograms
of intra-class vs inter-class pairs side by side.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from core.config.config import get_active_dataset_config, get_config, get_root_path
from core.datasets import get_data_loaders
from core.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Cosine similarity distribution comparison")
    parser.add_argument("--weights1", required=True, help="Path to checkpoint 1 (e.g. CE)")
    parser.add_argument("--weights2", required=True, help="Path to checkpoint 2 (e.g. SupCon)")
    parser.add_argument("--label1", default="CE", help="Label for checkpoint 1")
    parser.add_argument("--label2", default="SupCon", help="Label for checkpoint 2")
    parser.add_argument("--model-name", default="DinoV2RS_Small", help="Model name from Models enum")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", default="Test", choices=["Val", "Test"], help="Split to evaluate")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--output", default="cosine_similarity_comparison.png", help="Output image path")
    parser.add_argument("--use-projection", action="store_true",
                        help="Use projection head embeddings instead of backbone features")
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
def extract_embeddings(model, dataloader, device, use_projection=False):
    """Extract embeddings and labels from a model."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        imgs, labels = batch[0].to(device), batch[1]

        if use_projection:
            _, embeddings = model(imgs, return_embedding=True)
        else:
            features_dict = model.backbone.forward_features(imgs)
            embeddings = features_dict["x_norm_clstoken"]

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def compute_intra_inter_similarities(embeddings, labels):
    """Compute cosine similarities separated into intra-class and inter-class pairs."""
    sim_matrix = cosine_similarity(embeddings)
    n = len(labels)

    intra_sims = []
    inter_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j]
            if labels[i] == labels[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    return np.array(intra_sims), np.array(inter_sims)


def plot_comparison(intra1, inter1, label1, intra2, inter2, label2, save_path):
    """Plot side-by-side histograms of cosine similarity distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    bins = np.linspace(-0.2, 1.0, 80)

    for ax, intra, inter, title in [(ax1, intra1, inter1, label1), (ax2, intra2, inter2, label2)]:
        ax.hist(inter, bins=bins, alpha=0.6, color="#4B9BFF", label="Inter-class", density=True)
        ax.hist(intra, bins=bins, alpha=0.6, color="#FF4B4B", label="Intra-class", density=True)

        ax.axvline(np.mean(intra), color="#CC0000", linestyle="--", linewidth=1.5,
                   label=f"Intra mean: {np.mean(intra):.3f}")
        ax.axvline(np.mean(inter), color="#0055CC", linestyle="--", linewidth=1.5,
                   label=f"Inter mean: {np.mean(inter):.3f}")

        separation = np.mean(intra) - np.mean(inter)
        ax.set_title(f"{title}\n(separation: {separation:.3f})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cosine Similarity")
        ax.legend(loc="upper left", fontsize=9)

    ax1.set_ylabel("Density")
    plt.suptitle("Intra-class vs Inter-class Cosine Similarity", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load config
    num_classes = get_active_dataset_config("num_classes")
    model_cfg = get_config("models", args.model_name) or {}
    img_size = model_cfg.get("img_size", 224)
    root_path = get_root_path()

    # Create dataloader
    data_loaders, _ = get_data_loaders(
        root_path=root_path, batch_size=args.batch_size, img_size=img_size
    )
    loader = data_loaders[args.split]

    # Process each checkpoint
    all_results = []
    for ckpt_path, label in [(args.weights1, args.label1), (args.weights2, args.label2)]:
        print(f"\nExtracting embeddings: {label} ({ckpt_path})")
        model = get_model(args.model_name, num_classes, device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        embeddings, labels = extract_embeddings(model, loader, device, args.use_projection)
        print(f"  Embeddings shape: {embeddings.shape}")

        print(f"  Computing pairwise cosine similarities...")
        intra, inter = compute_intra_inter_similarities(embeddings, labels)
        print(f"  Intra-class pairs: {len(intra):,}  |  Inter-class pairs: {len(inter):,}")
        print(f"  Intra mean: {np.mean(intra):.4f} (std: {np.std(intra):.4f})")
        print(f"  Inter mean: {np.mean(inter):.4f} (std: {np.std(inter):.4f})")
        print(f"  Separation (intra_mean - inter_mean): {np.mean(intra) - np.mean(inter):.4f}")

        all_results.append((intra, inter, label))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    intra1, inter1, lbl1 = all_results[0]
    intra2, inter2, lbl2 = all_results[1]

    # Plot
    plot_comparison(intra1, inter1, lbl1, intra2, inter2, lbl2, args.output)


if __name__ == "__main__":
    main()
