"""Utility functions for dataset inspection, visualization, and device management.

Provides tools for:
- Loading and analyzing dataset results
- Displaying image comparisons
- Interactive dataset cleaning based on distance metrics
- Device detection and management
- Model persistence
"""

import os
import torch
from PIL import Image

import matplotlib.pyplot as plt
from core.config.config import log_info


path_fire = 'data/Train/Fire'
path_not = 'data/Train/Not'

key_pressed = None


def load_results(filename: str) -> list:
    """Load distance analysis results from text file.

    Parses a text file containing image pairs and their distance metrics.
    Expected format: "labelNot,labelFire -> distance"

    Args:
        filename: Path to the results file to load.

    Returns:
        List of tuples containing (labelNot, labelFire, distance) for each pair.
    """
    results = []
    with open(filename, "r") as f:
        for linha in f:
            linha = linha.strip()
            if '->' in linha:
                parte_labels, dist_str = linha.split(' -> ')
                labelNot, labelFire = parte_labels.split(',')
                dist = float(dist_str)
                results.append((labelNot, labelFire, dist))
    return results


def show(labelNot: str, labelFire: str, dist: float) -> None:
    """Display side-by-side comparison of fire and non-fire images.

    Shows two images in a matplotlib figure with their distance metric as title.
    Waits for user keyboard input before closing.

    Args:
        labelNot: Path to the non-fire image.
        labelFire: Path to the fire image.
        dist: Distance metric between the two images.
    """
    img1 = Image.open(labelNot)
    img2 = Image.open(labelFire)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1)
    axs[0].set_title(f"Not Fire")
    axs[0].axis("off")

    axs[1].imshow(img2)
    axs[1].set_title(f"Fire")
    axs[1].axis("off")

    plt.suptitle(f"Distance {dist}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    resposta = plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return


def on_key(event) -> None:
    """Handle keyboard events during image display.

    Sets the global key_pressed variable and closes the current plot window.

    Args:
        event: Matplotlib key press event object containing key information.
    """
    global key_pressed
    key_pressed = event.key
    plt.close()


def check_images(filename: str) -> None:
    """Interactive dataset cleaning tool based on distance analysis.

    Loads image pairs from results file and displays them for manual review.
    User can mark images for removal using keyboard inputs:
    - '1': Remove non-fire image
    - '2': Remove fire image
    - 'b': Remove both images

    Args:
        filename: Path to the results file containing image pairs and distances.
    """
    results = load_results(filename)
    to_remove = []

    for labelNot, labelFire, dist in results:
        global key_pressed
        key_pressed = None

        labelNot, labelFire = f'{path_not}/{labelNot}', f'{path_fire}/{labelFire}'

        if labelNot in to_remove or labelFire in to_remove:
            continue

        show(labelNot, labelFire, dist)

        if key_pressed == '1':
            to_remove.append(labelNot)
        elif key_pressed == '2':
            to_remove.append(labelFire)
        elif key_pressed == 'b':
            to_remove.append(labelNot)
            to_remove.append(labelFire)

    for path in to_remove:
        if os.path.exists(path):
            os.remove(path)
            log_info(f"Removed: {path}")
    return


# Device and Model Management Functions


def get_device() -> torch.device:
    """Get the best available device for computation.

    Returns:
        torch.device: CUDA if available, otherwise MPS if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def save_model(model, path="models/model.pth"):
    """Save model weights to disk.

    Args:
        model: Model to save
        path: Path where to save the model weights
    """
    torch.save(model.state_dict(), os.path.abspath(path))
    log_info(f"Model saved in: {path}")


