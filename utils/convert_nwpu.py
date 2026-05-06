#!/usr/bin/env python3
"""
Dataset converter for NWPU VHR-10.

Supports two output formats:
- COCO: images/ + annotations.json
- ImageFolder: class-based folders

Input structure:
    NWPU-VHR-10/
    ├── positive image set/
    ├── negative image set/
    └── ground truth/

Output (COCO):
    data_nwpu/
    ├── Train/
    │   ├── images/
    │   └── annotations.json
    └── ...

Output (ImageFolder):
    data_nwpu/
    ├── Train/
    │   ├── airplane/
    │   ├── ship/
    │   └── ...
    └── ...
"""

import os
import json
import shutil
import argparse
import sys
import random
from typing import Dict, List, Tuple
from collections import defaultdict

from PIL import Image
from core.config.config import log_info


# =============================================================================
# Constants
# =============================================================================

CLASS_MAPPING = {
    1: "airplane",
    2: "ship",
    3: "storage_tank",
    4: "baseball_diamond",
    5: "tennis_court",
    6: "basketball_court",
    7: "ground_track_field",
    8: "harbor",
    9: "bridge",
    10: "vehicle"
}

ALL_CLASSES = list(CLASS_MAPPING.values()) + ["background"]
SPLITS = ["Train", "Val", "Test"]


# =============================================================================
# Parsing
# =============================================================================

def parse_annotation_line(line: str) -> Tuple[int, int, int, int, int]:
    """Parse NWPU annotation line: (x1,y1),(x2,y2),class_id"""
    try:
        line = line.strip().replace('(', '').replace(')', '')
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    except (ValueError, IndexError) as e:
        log_info(f"WARNING: Error parsing '{line}': {e}")
    return None


def load_annotations(gt_dir: str, images_dir: str) -> Dict[str, List[Tuple]]:
    """Load all NWPU annotations. Returns {image_name: [(x1,y1,x2,y2,class_id), ...]}"""
    annotations = defaultdict(list)

    for gt_file in os.listdir(gt_dir):
        if not gt_file.endswith('.txt'):
            continue

        image_name = gt_file.replace('.txt', '.jpg')
        if not os.path.exists(os.path.join(images_dir, image_name)):
            continue

        try:
            with open(os.path.join(gt_dir, gt_file), 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parsed = parse_annotation_line(line)
                        if parsed:
                            annotations[image_name].append(parsed)
        except Exception as e:
            log_info(f"WARNING: Error reading {gt_file}: {e}")

    return dict(annotations)


def load_background_images(negative_dir: str) -> List[str]:
    """Load background image filenames."""
    if not os.path.exists(negative_dir):
        return []
    return [f for f in os.listdir(negative_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


# =============================================================================
# Splitting
# =============================================================================

def split_dataset(
    annotations: Dict[str, List[Tuple]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """Split dataset maintaining class distribution."""
    random.seed(seed)

    # Group by main class
    class_to_images = defaultdict(list)
    for image_name, anns in annotations.items():
        if not anns:
            continue
        class_counts = defaultdict(int)
        for _, _, _, _, class_id in anns:
            class_counts[class_id] += 1
        main_class = max(class_counts.items(), key=lambda x: x[1])[0]
        class_to_images[main_class].append(image_name)

    train, val, test = {}, {}, {}

    for class_id, images in class_to_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for img in images[:n_train]:
            train[img] = annotations[img]
        for img in images[n_train:n_train + n_val]:
            val[img] = annotations[img]
        for img in images[n_train + n_val:]:
            test[img] = annotations[img]

        log_info(f"  {CLASS_MAPPING[class_id]:20s}: Train={n_train}, Val={n_val}, Test={n - n_train - n_val}")

    return train, val, test


def split_background(images: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """Split background images."""
    random.seed(seed)
    random.shuffle(images)
    n = len(images)
    t_idx = int(n * train_ratio)
    v_idx = t_idx + int(n * val_ratio)
    return images[:t_idx], images[t_idx:v_idx], images[v_idx:]


# =============================================================================
# COCO Format
# =============================================================================

def create_coco_annotations(split_data: Dict[str, List[Tuple]], images_dir: str) -> Dict:
    """Create COCO format annotations."""
    coco = {"images": [], "annotations": [], "categories": []}

    for class_id, class_name in CLASS_MAPPING.items():
        coco["categories"].append({"id": class_id, "name": class_name, "supercategory": "object"})

    ann_id, img_id = 1, 1

    for image_name in sorted(split_data.keys()):
        try:
            with Image.open(os.path.join(images_dir, image_name)) as img:
                width, height = img.size
        except Exception:
            continue

        coco["images"].append({"id": img_id, "file_name": image_name, "width": width, "height": height})

        for x1, y1, x2, y2, class_id in split_data[image_name]:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    return coco


def convert_to_coco(
    nwpu_dir: str,
    output_dir: str = "data_nwpu",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """Convert NWPU to COCO format."""
    log_info("=" * 60)
    log_info("Converting NWPU VHR-10 to COCO format")
    log_info("=" * 60)

    images_dir = os.path.join(nwpu_dir, "positive image set")
    gt_dir = os.path.join(nwpu_dir, "ground truth")

    if not os.path.exists(images_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(f"NWPU directories not found in {nwpu_dir}")

    log_info(f"\nInput: {nwpu_dir}")
    log_info(f"Output: {output_dir}\n")

    annotations = load_annotations(gt_dir, images_dir)
    log_info(f"Loaded {len(annotations)} images\n")

    train, val, test = split_dataset(annotations, train_ratio, val_ratio, seed)

    for split_name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        if not split_data:
            continue

        out_images = os.path.join(output_dir, split_name, "images")
        os.makedirs(out_images, exist_ok=True)

        log_info(f"\n{split_name}: Copying {len(split_data)} images...")
        for img_name in split_data:
            shutil.copy2(os.path.join(images_dir, img_name), os.path.join(out_images, img_name))

        coco = create_coco_annotations(split_data, images_dir)
        with open(os.path.join(output_dir, split_name, "annotations.json"), 'w') as f:
            json.dump(coco, f, indent=2)

        log_info(f"{split_name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    log_info("\n" + "=" * 60)
    log_info("Conversion completed!")
    log_info("=" * 60)


# =============================================================================
# ImageFolder Format
# =============================================================================

def convert_to_imagefolder(
    nwpu_dir: str,
    output_dir: str = "data_nwpu",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """Convert NWPU to ImageFolder format."""
    log_info("=" * 60)
    log_info("Converting NWPU VHR-10 to ImageFolder format")
    log_info("=" * 60)

    positive_dir = os.path.join(nwpu_dir, "positive image set")
    negative_dir = os.path.join(nwpu_dir, "negative image set")
    gt_dir = os.path.join(nwpu_dir, "ground truth")

    if not os.path.exists(positive_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(f"NWPU directories not found in {nwpu_dir}")

    log_info(f"\nInput: {nwpu_dir}")
    log_info(f"Output: {output_dir}\n")

    annotations = load_annotations(gt_dir, positive_dir)
    background = load_background_images(negative_dir)

    log_info(f"Loaded {len(annotations)} positive, {len(background)} background images\n")

    train, val, test = split_dataset(annotations, train_ratio, val_ratio, seed)
    train_bg, val_bg, test_bg = split_background(background, train_ratio, val_ratio, seed)

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data, bg_list in [("Train", train, train_bg), ("Val", val, val_bg), ("Test", test, test_bg)]:
        log_info(f"\n{split_name}:")

        if split_name == "Test":
            # Test: single images folder
            out_dir = os.path.join(output_dir, split_name, "images")
            os.makedirs(out_dir, exist_ok=True)
            count = 0
            for img_name in split_data:
                shutil.copy2(os.path.join(positive_dir, img_name), os.path.join(out_dir, img_name))
                count += 1
            for img_name in bg_list:
                src = os.path.join(negative_dir, img_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(out_dir, img_name))
                    count += 1
            log_info(f"  images/: {count}")
        else:
            # Train/Val: class folders
            for cls in ALL_CLASSES:
                os.makedirs(os.path.join(output_dir, split_name, cls), exist_ok=True)

            counts = {cls: 0 for cls in ALL_CLASSES}

            for img_name, anns in split_data.items():
                classes_in_img = set(CLASS_MAPPING[a[4]] for a in anns if a[4] in CLASS_MAPPING)
                for cls in classes_in_img:
                    shutil.copy2(
                        os.path.join(positive_dir, img_name),
                        os.path.join(output_dir, split_name, cls, img_name)
                    )
                    counts[cls] += 1

            for img_name in bg_list:
                src = os.path.join(negative_dir, img_name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_dir, split_name, "background", img_name))
                    counts["background"] += 1

            for cls in ALL_CLASSES:
                if counts[cls] > 0:
                    log_info(f"  {cls}: {counts[cls]}")

    log_info("\n" + "=" * 60)
    log_info("Conversion completed!")
    log_info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    # No arguments: run COCO with defaults
    if len(sys.argv) == 1:
        convert_to_coco(nwpu_dir="NWPU-VHR-10", output_dir="data_nwpu")
        return

    parser = argparse.ArgumentParser(description="Convert NWPU VHR-10 to COCO or ImageFolder format")
    parser.add_argument("--nwpu-dir", type=str, default="NWPU-VHR-10", help="NWPU root directory")
    parser.add_argument("--output-dir", type=str, default="data_nwpu", help="Output directory")
    parser.add_argument("--format", type=str, choices=["coco", "imagefolder"], default="coco", help="Output format")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.format == "coco":
        convert_to_coco(args.nwpu_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
    else:
        convert_to_imagefolder(args.nwpu_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
