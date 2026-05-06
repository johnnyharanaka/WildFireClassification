#!/usr/bin/env python3
"""
Dataset converter for Fire dataset (LabelMe to COCO format).

Input structure:
    data_fire/
    ├── Train/
    │   ├── Fire/       (.jpg + .json annotations)
    │   └── Not/
    ├── Val/
    │   ├── Fire/
    │   └── Not/
    └── Test/
        ├── Fire/
        └── Not/

Output structure:
    data_fire/
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

import os
import json
import shutil
import argparse
import sys
from typing import List

from PIL import Image
from core.config.config import log_info


# =============================================================================
# Constants
# =============================================================================

CLASSES = ["Fire", "Not"]
SPLITS = ["Train", "Val", "Test"]


# =============================================================================
# Parsing
# =============================================================================

def parse_labelme_json(json_path: str) -> List[tuple]:
    """
    Parse a LabelMe JSON file.

    Returns:
        List of bboxes: [(x, y, width, height), ...]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        bboxes = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]

                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1, x2)
                y_max = max(y1, y2)

                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                bboxes.append(bbox)

        return bboxes
    except Exception as e:
        log_info(f"WARNING: Error parsing {json_path}: {e}")
        return []


# =============================================================================
# Conversion
# =============================================================================

def convert_split(split_path: str, output_dir: str, split_name: str):
    """Convert a single split (Train/Val/Test) to COCO format."""
    log_info(f"\n{'='*60}")
    log_info(f"Converting {split_name}")
    log_info(f"{'='*60}\n")

    images_output_dir = os.path.join(output_dir, split_name, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Fire", "supercategory": "object"}]
    }

    annotation_id = 1
    image_id = 1

    for class_name in CLASSES:
        class_dir = os.path.join(split_path, class_name)

        if not os.path.exists(class_dir):
            log_info(f"WARNING: Directory not found: {class_dir}")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        log_info(f"Processing '{class_name}': {len(image_files)} images")

        for image_file in sorted(image_files):
            image_path = os.path.join(class_dir, image_file)
            json_path = image_path.replace('.jpg', '.json')

            # Copy image
            dst_path = os.path.join(images_output_dir, image_file)
            try:
                shutil.copy2(image_path, dst_path)
            except Exception as e:
                log_info(f"WARNING: Error copying {image_file}: {e}")
                continue

            # Get dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                log_info(f"WARNING: Error reading {image_file}: {e}")
                continue

            coco_format["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })

            # Annotations (only for Fire class)
            if class_name == "Fire" and os.path.exists(json_path):
                bboxes = parse_labelme_json(json_path)

                for bbox in bboxes:
                    x, y, w, h = bbox
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

            image_id += 1

    # Save annotations
    annotations_path = os.path.join(output_dir, split_name, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    num_images = len(coco_format["images"])
    num_annotations = len(coco_format["annotations"])
    avg_boxes = num_annotations / num_images if num_images > 0 else 0

    log_info(f"\n{split_name}: {num_images} images, {num_annotations} annotations (avg: {avg_boxes:.1f} boxes/img)")


def convert_fire_to_coco(fire_root_dir: str = "data_fire", output_dir: str = None):
    """
    Convert Fire dataset to COCO format.

    Args:
        fire_root_dir: Root directory of Fire dataset
        output_dir: Output directory (default: same as fire_root_dir)
    """
    if output_dir is None:
        output_dir = fire_root_dir

    log_info("=" * 60)
    log_info("Converting Fire Dataset to COCO format")
    log_info("=" * 60)
    log_info(f"\nInput:  {fire_root_dir}")
    log_info(f"Output: {output_dir}")

    for split_name in SPLITS:
        split_path = os.path.join(fire_root_dir, split_name)

        if not os.path.exists(split_path):
            log_info(f"\nWARNING: Split not found: {split_name}")
            continue

        convert_split(split_path, output_dir, split_name)

    log_info("\n" + "=" * 60)
    log_info("Conversion completed!")
    log_info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    # No arguments: run with defaults
    if len(sys.argv) == 1:
        convert_fire_to_coco()
        return

    parser = argparse.ArgumentParser(description="Convert Fire dataset to COCO format")
    parser.add_argument("--fire-dir", type=str, default="data_fire", help="Fire dataset directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()
    convert_fire_to_coco(fire_root_dir=args.fire_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
