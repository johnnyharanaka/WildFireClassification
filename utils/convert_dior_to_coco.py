"""
Converte o dataset DIOR para o formato COCO

Entrada (DIOR format - Hugging Face):
- Automaticamente baixado de HuggingFace
- Contém splits: train, validation, test
- Cada amostra tem imagem + objetos com bounding boxes

Saída (COCO format):
data_dior/
├── Train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── annotations.json
├── Val/
│   ├── images/
│   └── annotations.json
└── Test/
    ├── images/
    └── annotations.json

annotations.json segue o formato COCO:
{
    "images": [{"id": 1, "file_name": "img1.jpg", "width": 800, "height": 600}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": w*h}],
    "categories": [{"id": 1, "name": "airplane", "supercategory": "object"}]
}
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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


def download_dior_dataset(cache_dir: str = "./data/dior_raw") -> Dict:
    """
    Download DIOR dataset from Hugging Face.

    Returns:
        Dictionary with splits: {'train': dataset, 'validation': dataset, 'test': dataset}
    """
    if not HAS_HF_DATASETS:
        raise ImportError(
            "The 'datasets' library is required. Install with: pip install datasets"
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


def extract_and_save_images(hf_dataset, split: str, output_dir: str,
                            sample_limit: Optional[int] = None) -> List[Tuple[str, List[Dict]]]:
    """
    Extract images from HF dataset and save to disk.

    Returns:
        List of tuples (image_name, annotations)
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

            if isinstance(objects_dict, dict) and 'category' in objects_dict:
                categories = objects_dict.get('category', [])
                bboxes = objects_dict.get('bbox', [])

                for category, bbox in zip(categories, bboxes):
                    if bbox and category is not None:
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x, y, w, h = bbox
                            annotations.append({
                                'bbox': [x, y, w, h],
                                'category': int(category),
                            })
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
                            })

        samples.append((image_name, annotations))

    log_info(f"Extracted {len(samples)} images")
    return samples


def create_coco_annotations(samples: List[Tuple[str, List[Dict]]],
                           images_dir: str,
                           split_name: str) -> Dict:
    """
    Create COCO format annotations from samples.
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    for class_id, class_name in DIOR_CLASS_MAPPING.items():
        coco_format["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1
    image_id = 1

    for image_name, annotations in samples:
        image_path = os.path.join(images_dir, image_name)

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            log_info(f"Warning: Could not read image {image_name}: {e}")
            continue

        # Add image info
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

        image_id += 1

    return coco_format


def copy_images_and_save_annotations(samples: List[Tuple[str, List[Dict]]],
                                     temp_images_dir: str,
                                     output_dir: str,
                                     split_name: str):
    """
    Copy images and save COCO annotations.
    """
    # Create output directory for images
    images_output_dir = os.path.join(output_dir, split_name, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    # Copy images
    log_info(f"\n{split_name}: Copying {len(samples)} images...")
    for image_name, _ in samples:
        src_path = os.path.join(temp_images_dir, image_name)
        dst_path = os.path.join(images_output_dir, image_name)

        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    # Create COCO annotations
    log_info(f"{split_name}: Creating annotations.json...")
    coco_annotations = create_coco_annotations(samples, temp_images_dir, split_name)

    # Save annotations.json
    annotations_path = os.path.join(output_dir, split_name, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(coco_annotations, f, indent=2)

    # Statistics
    num_images = len(coco_annotations["images"])
    num_annotations = len(coco_annotations["annotations"])
    avg_annotations = num_annotations / num_images if num_images > 0 else 0

    log_info(f"{split_name}: Saved {num_images} images, {num_annotations} annotations (avg: {avg_annotations:.1f} boxes/img)")


def convert_dior_to_coco(output_dir: str = "data_dior",
                        dior_dataset_path: Optional[str] = None,
                        sample_limits: Optional[Dict[str, int]] = None):
    """
    Convert DIOR dataset to COCO format.
    """
    log_info("="*70)
    log_info("Converting DIOR to COCO format")
    log_info("="*70)

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

    log_info("")

    # Process each split
    splits_data = {}
    temp_dirs = {}

    for src_split in ["train", "validation", "test"]:
        if src_split not in hf_dataset:
            log_info(f"Split {src_split} not found in dataset")
            continue

        # Map split names
        split_mapping = {"train": "Train", "validation": "Val", "test": "Test"}
        dst_split = split_mapping[src_split]

        # Create temp directory for extracted images
        temp_dir = os.path.join(output_dir, ".temp", src_split)
        os.makedirs(temp_dir, exist_ok=True)
        temp_dirs[dst_split] = temp_dir

        # Extract images and annotations
        sample_limit = sample_limits.get(src_split) if sample_limits else None
        samples = extract_and_save_images(
            hf_dataset,
            src_split,
            temp_dir,
            sample_limit=sample_limit
        )

        splits_data[dst_split] = (samples, temp_dir)

    # Copy images and save annotations
    for split_name, (samples, temp_dir) in splits_data.items():
        copy_images_and_save_annotations(samples, temp_dir, output_dir, split_name)

    # Cleanup temp directories
    temp_base = os.path.join(output_dir, ".temp")
    if os.path.exists(temp_base):
        shutil.rmtree(temp_base, ignore_errors=True)

    log_info("\n" + "="*70)
    log_info("Conversion completed successfully!")
    log_info("="*70)
    log_info(f"\nOutput structure (COCO format):")
    log_info(f"{output_dir}/")

    for split_name in ["Train", "Val", "Test"]:
        if split_name in splits_data:
            num_images = len(splits_data[split_name][0])
            log_info(f"├── {split_name}/")
            log_info(f"│   ├── images/          ({num_images} images)")
            log_info(f"│   └── annotations.json")

    log_info(f"\nDirectory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DIOR to COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_dior_to_coco.py --output-dir data_dior
  python convert_dior_to_coco.py --output-dir data_dior --sample-limit 100 20 20

Output structure (COCO):
  data_dior/
  ├── Train/
  │   ├── images/
  │   └── annotations.json
  ├── Val/
  │   ├── images/
  │   └── annotations.json
  └── Test/
      ├── images/
      └── annotations.json
        """
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_dior",
        help="Output directory (default: data_dior)"
    )
    parser.add_argument(
        "--dior-path",
        type=str,
        default=None,
        help="Path to locally downloaded DIOR dataset (optional)"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Limit samples per split (e.g., --sample-limit 100 20 20)"
    )

    args = parser.parse_args()

    sample_limits = None
    if args.sample_limit:
        sample_limits = {
            "train": args.sample_limit[0],
            "validation": args.sample_limit[1],
            "test": args.sample_limit[2]
        }

    convert_dior_to_coco(
        output_dir=args.output_dir,
        dior_dataset_path=args.dior_path,
        sample_limits=sample_limits
    )