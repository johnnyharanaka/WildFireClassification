"""
Dataset converter for DIOR (Object Detection in Optical Remote Sensing Images).

This module provides utilities to download and convert the DIOR dataset from Hugging Face
into both COCO format (for Fire dataset compatibility) and ImageFolder format
(for NWPU compatibility).

The DIOR dataset contains 20 classes and is organized with bounding box annotations.
It can be converted to support both classification and object detection tasks.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

from core.config.config import log_info


# DIOR class mapping - 20 object classes in remote sensing
DIOR_CLASS_MAPPING = {
    0: "airplane",
    1: "airport",
    2: "baseball_diamond",
    3: "basketball_court",
    4: "bridge",
    5: "chaparral",
    6: "dense_residential",
    7: "forest",
    8: "freeway",
    9: "golf_course",
    10: "ground_track_field",
    11: "harbor",
    12: "industrial",
    13: "intersection",
    14: "medium_residential",
    15: "mobile_home_park",
    16: "overpass",
    17: "parking_lot",
    18: "sparse_residential",
    19: "tennis_court"
}

DIOR_CLASS_NAME_TO_ID = {v: k for k, v in DIOR_CLASS_MAPPING.items()}


def download_dior_dataset(cache_dir: str = "./data/dior_raw") -> Dict[str, any]:
    """
    Download DIOR dataset from Hugging Face.

    Args:
        cache_dir: Directory to cache the downloaded dataset.

    Returns:
        Dictionary with splits: {'train': dataset, 'validation': dataset, 'test': dataset}

    Raises:
        ImportError: If datasets library is not installed.
        RuntimeError: If download fails.
    """
    if not HAS_HF_DATASETS:
        raise ImportError(
            "The 'datasets' library is required to download from Hugging Face. "
            "Install it with: pip install datasets"
        )

    os.makedirs(cache_dir, exist_ok=True)
    log_info(f"Downloading DIOR dataset to {cache_dir}...")

    try:
        dataset = load_dataset("HichTala/dior", cache_dir=cache_dir)
        log_info(f"DIOR dataset downloaded successfully")
        log_info(f"Available splits: {list(dataset.keys())}")

        for split, data in dataset.items():
            log_info(f"  {split}: {len(data)} images")

        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to download DIOR dataset: {e}")


def extract_images_from_hf_dataset(
    hf_dataset,
    output_dir: str,
    split: str = "train",
    sample_limit: Optional[int] = None
) -> List[Tuple[str, str, List[Dict]]]:
    """
    Extract images and annotations from Hugging Face dataset format.

    Args:
        hf_dataset: Hugging Face dataset object.
        output_dir: Directory to save extracted images.
        split: Split name (train/validation/test).
        sample_limit: Maximum number of samples to process (for testing).

    Returns:
        List of tuples (image_path, image_name, annotations).
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    dataset_size = len(hf_dataset[split]) if split in hf_dataset else len(hf_dataset)

    if sample_limit:
        dataset_size = min(dataset_size, sample_limit)

    log_info(f"Extracting {dataset_size} images from {split} split...")

    for idx in tqdm(range(dataset_size), desc=f"Extracting {split}"):
        sample = hf_dataset[split][idx] if split in hf_dataset else hf_dataset[idx]

        # Extract image
        image = sample['image']
        image_name = sample.get('image_name', f"{split}_{idx:06d}.jpg")

        if isinstance(image, Image.Image):
            image_path = os.path.join(output_dir, image_name)
            image.save(image_path)
        else:
            log_info(f"Warning: Image format unexpected for {image_name}")
            continue

        # Extract annotations
        annotations = []
        if 'objects' in sample and sample['objects'] is not None:
            objects_dict = sample['objects']

            # Handle DIOR format: objects is a dict with lists
            # Each key has a list of values (one per object)
            if isinstance(objects_dict, dict) and 'category' in objects_dict:
                categories = objects_dict.get('category', [])
                bboxes = objects_dict.get('bbox', [])

                for i, (category, bbox) in enumerate(zip(categories, bboxes)):
                    if bbox and category is not None:
                        # bbox format: [x, y, width, height]
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x, y, w, h = bbox
                            annotations.append({
                                'bbox': [x, y, w, h],
                                'category': int(category),
                                'class_name': DIOR_CLASS_MAPPING.get(int(category), 'unknown')
                            })
            # Fallback: handle list of objects (in case format is different)
            elif isinstance(objects_dict, list):
                for obj in objects_dict:
                    bbox = obj.get('bbox') if isinstance(obj, dict) else None
                    category = obj.get('category') if isinstance(obj, dict) else None

                    if bbox and category is not None:
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x, y, w, h = bbox
                            annotations.append({
                                'bbox': [x, y, w, h],
                                'category': int(category),
                                'class_name': DIOR_CLASS_MAPPING.get(int(category), 'unknown')
                            })

        samples.append((image_path, image_name, annotations))

    log_info(f"Extracted {len(samples)} images")
    return samples


def create_coco_format(
    samples: List[Tuple[str, str, List[Dict]]],
    output_dir: str,
    split: str = "train"
) -> Dict:
    """
    Convert extracted samples to COCO format.

    Args:
        samples: List of (image_path, image_name, annotations) tuples.
        output_dir: Directory to save COCO files.
        split: Split name (train/test/val).

    Returns:
        COCO format dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create category list
    for class_id, class_name in DIOR_CLASS_MAPPING.items():
        coco_format["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "object"
        })

    # Create images and annotations
    annotation_id = 1
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    log_info(f"Creating COCO format for {len(samples)} samples...")

    for image_id, (_, image_name, annotations) in enumerate(tqdm(samples, desc="COCO format")):
        # Copy image to images directory
        src_path = _
        dst_path = os.path.join(images_dir, image_name)

        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        # Add image info
        img = Image.open(dst_path)
        width, height = img.size

        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Add annotations
        for ann in annotations:
            x, y, w, h = ann['bbox']
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann['category'],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    # Save COCO JSON
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    log_info(f"COCO format saved to {output_dir}")
    log_info(f"  Images: {len(coco_format['images'])}")
    log_info(f"  Annotations: {len(coco_format['annotations'])}")
    log_info(f"  Categories: {len(coco_format['categories'])}")

    return coco_format


def create_imagefolder_format(
    samples: List[Tuple[str, str, List[Dict]]],
    output_dir: str,
    classification_method: str = "primary_class"
) -> None:
    """
    Convert samples to ImageFolder format with class directories (Train/Val).

    Args:
        samples: List of (image_path, image_name, annotations) tuples.
        output_dir: Root directory for ImageFolder structure.
        classification_method: How to assign class:
            - 'primary_class': Use the first object's class (default)
            - 'multi_object': Create symlinks for each object class in image
            - 'dominant_class': Use most frequent class in image
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create class directories
    for class_name in DIOR_CLASS_MAPPING.values():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    log_info(f"Creating ImageFolder format with {classification_method} method...")

    for src_path, image_name, annotations in tqdm(samples, desc="ImageFolder format"):
        if not os.path.exists(src_path):
            continue

        if not annotations:
            # Skip images without annotations (DIOR has annotations for all images)
            continue
        elif classification_method == "primary_class":
            # Use first object's class
            class_id = annotations[0]['category']
            class_name = DIOR_CLASS_MAPPING.get(class_id, "unknown")
            dst_dir = os.path.join(output_dir, class_name)
        elif classification_method == "dominant_class":
            # Use most frequent class
            class_counts = defaultdict(int)
            for ann in annotations:
                class_counts[ann['category']] += 1
            class_id = max(class_counts.items(), key=lambda x: x[1])[0]
            class_name = DIOR_CLASS_MAPPING.get(class_id, "unknown")
            dst_dir = os.path.join(output_dir, class_name)
        elif classification_method == "multi_object":
            # Copy to each object's class
            for ann in annotations:
                class_id = ann['category']
                class_name = DIOR_CLASS_MAPPING.get(class_id, "unknown")
                dst_dir = os.path.join(output_dir, class_name)
                dst_path = os.path.join(dst_dir, image_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            continue
        else:
            # Default: use primary class
            class_id = annotations[0]['category']
            class_name = DIOR_CLASS_MAPPING.get(class_id, "unknown")
            dst_dir = os.path.join(output_dir, class_name)

        # Copy image to class directory
        dst_path = os.path.join(dst_dir, image_name)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    # Log statistics
    log_info("\nImageFolder class distribution:")
    for class_name in DIOR_CLASS_MAPPING.values():
        class_dir = os.path.join(output_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff'))])
        if count > 0:
            log_info(f"  {class_name}: {count}")


def create_test_flat_structure(
    samples: List[Tuple[str, str, List[Dict]]],
    output_dir: str
) -> None:
    """
    Create Test set structure with flat images folder + annotations.json (NWPU compatible).

    Args:
        samples: List of (image_path, image_name, annotations) tuples.
        output_dir: Root directory for Test split (e.g., output_path/Test).
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    log_info("Creating Test set with flat images structure...")

    # Copy all images to images/ folder
    for src_path, image_name, _ in tqdm(samples, desc="Test images"):
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(images_dir, image_name)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    log_info(f"Test: Copied {len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tiff'))])} images")


def create_test_annotations_json(
    samples: List[Tuple[str, str, List[Dict]]],
    output_dir: str,
    include_boxes: bool = True
) -> None:
    """
    Create annotations.json for Test set (COCO format).

    Args:
        samples: List of (image_path, image_name, annotations) tuples.
        output_dir: Root directory for Test split.
        include_boxes: If True, include bounding box annotations (default: True)
    """
    images_dir = os.path.join(output_dir, "images")

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create category list
    for class_id, class_name in DIOR_CLASS_MAPPING.items():
        coco_format["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "object"
        })

    # Create images list and annotations
    annotation_id = 1
    image_id = 0
    for _, image_name, annotations in samples:
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            continue

        # Get image dimensions
        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception as e:
            log_info(f"Warning: Could not read image {image_name}: {e}")
            width, height = 0, 0

        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Add bounding box annotations if requested
        if include_boxes:
            for ann in annotations:
                x, y, w, h = ann['bbox']
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann['category'],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # Save annotations.json
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    num_boxes = len(coco_format['annotations'])
    log_info(f"Test: annotations.json created with {len(coco_format['images'])} images, {num_boxes} boxes")


def convert_dior_to_coco(
    dior_dataset_path: Optional[str] = None,
    output_path: str = "./data/DIOR_COCO",
    splits: List[str] = ["train", "validation", "test"],
    split_names: Dict[str, str] = None,
    sample_limits: Optional[Dict[str, int]] = None
) -> Dict[str, str]:
    """
    Complete pipeline to convert DIOR to COCO format (Fire dataset compatible).

    Args:
        dior_dataset_path: Path to locally downloaded DIOR dataset.
                          If None, downloads from Hugging Face.
        output_path: Base output directory.
        splits: List of splits to process (train/validation/test).
        split_names: Optional mapping from source split to output split name.
                    Default: {'train': 'Train', 'validation': 'Val', 'test': 'Test'}
        sample_limits: Optional dict with sample limits per split (for testing).

    Returns:
        Dictionary mapping split names to output directories.
    """
    if split_names is None:
        split_names = {
            "train": "Train",
            "validation": "Val",
            "test": "Test"
        }

    os.makedirs(output_path, exist_ok=True)

    # Load or download dataset
    if dior_dataset_path and os.path.exists(dior_dataset_path):
        log_info(f"Loading DIOR dataset from {dior_dataset_path}...")
        # Assume Parquet format
        try:
            from datasets import load_from_disk
            hf_dataset = load_from_disk(dior_dataset_path)
        except Exception as e:
            log_info(f"Could not load from disk: {e}. Downloading from HF...")
            hf_dataset = download_dior_dataset()
    else:
        hf_dataset = download_dior_dataset()

    output_dirs = {}

    # Process each split
    for src_split in splits:
        if src_split not in hf_dataset:
            log_info(f"Split {src_split} not found in dataset")
            continue

        dst_split = split_names.get(src_split, src_split)
        split_output = os.path.join(output_path, dst_split)

        log_info(f"\n{'='*60}")
        log_info(f"Processing {src_split} -> {dst_split}")
        log_info(f"{'='*60}")

        # Extract images
        sample_limit = sample_limits.get(src_split) if sample_limits else None
        samples = extract_images_from_hf_dataset(
            hf_dataset,
            os.path.join(split_output, "images"),
            split=src_split,
            sample_limit=sample_limit
        )

        # Create COCO format
        create_coco_format(samples, split_output, split=dst_split)
        output_dirs[dst_split] = split_output

    log_info(f"\n{'='*60}")
    log_info(f"COCO format conversion complete!")
    log_info(f"Output directory: {output_path}")
    log_info(f"{'='*60}")

    return output_dirs


def convert_dior_to_imagefolder(
    dior_dataset_path: Optional[str] = None,
    output_path: str = "./data/DIOR_ImageFolder",
    splits: List[str] = ["train", "validation", "test"],
    split_names: Dict[str, str] = None,
    classification_method: str = "primary_class",
    sample_limits: Optional[Dict[str, int]] = None
) -> Dict[str, str]:
    """
    Complete pipeline to convert DIOR to ImageFolder format (NWPU compatible).

    Args:
        dior_dataset_path: Path to locally downloaded DIOR dataset.
                          If None, downloads from Hugging Face.
        output_path: Base output directory.
        splits: List of splits to process.
        split_names: Optional mapping from source split to output split name.
        classification_method: How to assign classes ('primary_class', 'dominant_class', 'multi_object').
        sample_limits: Optional dict with sample limits per split.

    Returns:
        Dictionary mapping split names to output directories.
    """
    if split_names is None:
        split_names = {
            "train": "Train",
            "validation": "Val",
            "test": "Test"
        }

    os.makedirs(output_path, exist_ok=True)

    # Load or download dataset
    if dior_dataset_path and os.path.exists(dior_dataset_path):
        log_info(f"Loading DIOR dataset from {dior_dataset_path}...")
        try:
            from datasets import load_from_disk
            hf_dataset = load_from_disk(dior_dataset_path)
        except Exception as e:
            log_info(f"Could not load from disk: {e}. Downloading from HF...")
            hf_dataset = download_dior_dataset()
    else:
        hf_dataset = download_dior_dataset()

    output_dirs = {}

    # Process each split
    for src_split in splits:
        if src_split not in hf_dataset:
            log_info(f"Split {src_split} not found in dataset")
            continue

        dst_split = split_names.get(src_split, src_split)
        split_output = os.path.join(output_path, dst_split)

        log_info(f"\n{'='*60}")
        log_info(f"Processing {src_split} -> {dst_split}")
        log_info(f"{'='*60}")

        # Extract images
        sample_limit = sample_limits.get(src_split) if sample_limits else None
        samples = extract_images_from_hf_dataset(
            hf_dataset,
            os.path.join(split_output, "_images_temp"),
            split=src_split,
            sample_limit=sample_limit
        )

        # Handle Test split differently (flat structure with annotations.json)
        if dst_split == "Test" or src_split == "test":
            # Create flat Test structure (NWPU compatible)
            create_test_flat_structure(samples, split_output)
            create_test_annotations_json(samples, split_output)
        else:
            # Create ImageFolder format with class directories (Train/Val)
            create_imagefolder_format(
                samples,
                split_output,
                classification_method=classification_method
            )

        # Clean up temporary images
        shutil.rmtree(os.path.join(split_output, "_images_temp"), ignore_errors=True)

        output_dirs[dst_split] = split_output

    log_info(f"\n{'='*60}")
    log_info(f"ImageFolder conversion complete!")
    log_info(f"Output directory: {output_path}")
    log_info(f"{'='*60}")

    return output_dirs


def main():
    import sys

    # No arguments: run COCO with defaults
    if len(sys.argv) == 1:
        log_info("=" * 60)
        log_info("Converting DIOR to COCO format (default)")
        log_info("=" * 60)
        convert_dior_to_coco(output_path="data_dior")
        log_info("\n" + "=" * 60)
        log_info("Conversion complete!")
        log_info("=" * 60)
        return

    parser = argparse.ArgumentParser(description="Convert DIOR dataset to COCO or ImageFolder format")
    parser.add_argument("--format", type=str, choices=["coco", "imagefolder"], default="coco", help="Output format")
    parser.add_argument("--output-dir", type=str, default="data_dior", help="Output directory")
    parser.add_argument("--dior-path", type=str, default=None, help="Local DIOR path (optional)")
    parser.add_argument("--classification-method", type=str, choices=["primary_class", "dominant_class", "multi_object"], default="primary_class", help="ImageFolder classification method")
    parser.add_argument("--sample-limit", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=None, help="Limit samples per split")

    args = parser.parse_args()

    log_info("=" * 60)
    log_info(f"Converting DIOR to {args.format.upper()} format")
    log_info("=" * 60)

    sample_limits = None
    if args.sample_limit:
        sample_limits = {"train": args.sample_limit[0], "validation": args.sample_limit[1], "test": args.sample_limit[2]}

    if args.format == "coco":
        convert_dior_to_coco(dior_dataset_path=args.dior_path, output_path=args.output_dir, sample_limits=sample_limits)
    else:
        convert_dior_to_imagefolder(dior_dataset_path=args.dior_path, output_path=args.output_dir, classification_method=args.classification_method, sample_limits=sample_limits)

    log_info("\n" + "=" * 60)
    log_info(f"Conversion complete! Files at: {args.output_dir}")
    log_info("=" * 60)


if __name__ == "__main__":
    main()