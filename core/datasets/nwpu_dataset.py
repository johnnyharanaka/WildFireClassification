"""Dataset class and utilities for NWPU VHR-10 v2 with bounding box annotations.

Supports both classification and object detection evaluation with mAP
"""

import os
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

from core.config.config import log_info


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

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAPPING.items()}


def parse_annotation_file(annotation_path: str) -> List[Dict]:
    """Parse NWPU VHR-10 annotation file.

    Expected format: (x1,y1),(x2,y2),class_id

    Args:
        annotation_path: Path to the annotation file.

    Returns:
        List of dicts with {bbox: [x1, y1, x2, y2], class_id: int, class_name: str}.
    """
    annotations = []

    if not os.path.exists(annotation_path):
        return annotations

    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        x1 = int(parts[0].strip('( '))
                        y1 = int(parts[1].strip(') '))
                        x2 = int(parts[2].strip('( '))
                        y2 = int(parts[3].strip(') '))
                        class_id = int(parts[4].strip())

                        if class_id in CLASS_MAPPING:
                            annotations.append({
                                'bbox': [x1, y1, x2, y2],
                                'class_id': class_id,
                                'class_name': CLASS_MAPPING[class_id]
                            })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        log_info(f"Erro ao processar {annotation_path}: {e}")

    return annotations


class NWPUDatasetWithAnnotations(Dataset):
    """NWPU VHR-10 dataset that loads images and bounding box annotations.

    Supports multi-label classification based on bounding boxes.
    """

    def __init__(self,
                 data_dir: str,
                 annotations_dir: str,
                 transform=None,
                 split: str = "Train",
                 num_classes: int = 10):
        """Initialize the NWPU dataset.

        Args:
            data_dir: Directory with class subfolders (Train/Val/Test/airplane/...).
            annotations_dir: Directory with .txt annotation files.
            transform: Transformations to apply to images.
            split: Split name (Train/Val/Test).
            num_classes: Number of classes (10 for NWPU).
        """
        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.split = split
        self.num_classes = num_classes

        self.classes = []
        self.class_to_idx = {}
        self._detect_classes()

        self.samples = []
        self._load_samples()

    def _detect_classes(self):
        """Detect classes based on directory structure."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        potential_classes = [d for d in os.listdir(self.data_dir)
                           if os.path.isdir(os.path.join(self.data_dir, d))]

        self.classes = sorted(potential_classes)

        for idx, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = idx

        log_info(f"Classes detectadas no {self.split}: {self.classes}")

    def _load_samples(self):
        """Load all samples with paths and information."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')

        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                if filename.lower().endswith(valid_extensions):
                    filepath = os.path.join(class_dir, filename)

                    annotation_file = filename.replace('.jpg', '.txt').replace('.png', '.txt')
                    annotation_path = os.path.join(self.annotations_dir, annotation_file)

                    annotations = parse_annotation_file(annotation_path)

                    self.samples.append({
                        'filepath': filepath,
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'filename': filename,
                        'annotations': annotations
                    })

        log_info(f"Dataset {self.split}: {len(self.samples)} samples carregados\n")

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image, class_idx, filepath, annotations).
        """
        sample = self.samples[idx]

        image = Image.open(sample['filepath']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return (
            image,
            sample['class_idx'],
            sample['filepath'],
            sample['annotations']
        )

    def get_annotations_for_index(self, idx: int) -> List[Dict]:
        """Get annotations for a specific index.

        Args:
            idx: Index of the sample.

        Returns:
            List of annotation dictionaries.
        """
        return self.samples[idx]['annotations']

    def get_image_info(self, idx: int) -> Dict:
        """Get complete image information.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with image information.
        """
        return self.samples[idx]


def load_all_annotations(annotations_dir: str) -> Dict[str, List[Dict]]:
    """Load all annotations from a directory.

    Args:
        annotations_dir: Directory with .txt annotation files.

    Returns:
        Dict mapping image filename to list of annotations.
    """
    all_annotations = {}

    if not os.path.exists(annotations_dir):
        log_info(f"WARNING: Annotations directory not found: {annotations_dir}")
        return all_annotations

    for filename in os.listdir(annotations_dir):
        if filename.endswith('.txt'):
            annotation_path = os.path.join(annotations_dir, filename)
            annotations = parse_annotation_file(annotation_path)

            image_name = filename.replace('.txt', '.jpg')
            all_annotations[image_name] = annotations

    log_info(f"{len(all_annotations)} annotation files loaded")
    return all_annotations


def get_class_distribution(annotations: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Calculate class distribution in annotations.

    Args:
        annotations: Dict with loaded annotations.

    Returns:
        Dict with count per class.
    """
    class_counts = defaultdict(int)

    for image_name, image_annotations in annotations.items():
        for ann in image_annotations:
            class_name = ann['class_name']
            class_counts[class_name] += 1

    return dict(class_counts)


def create_nwpu_dataloaders(
    root_path: str,
    annotations_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int = 224,
    transform_dict: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """Create DataLoaders for NWPU with annotation support.

    Args:
        root_path: Root directory with Train/Val/Test.
        annotations_dir: Directory with .txt annotations.
        batch_size: Batch size.
        num_workers: Number of workers.
        img_size: Image size.
        transform_dict: Dict with transformations (if None, uses default).

    Returns:
        Tuple of (dataloaders, datasets).
    """
    from core.datasets.datasets import _get_data_transforms

    if transform_dict is None:
        transform_dict = _get_data_transforms(img_size)

    datasets = {}
    dataloaders = {}

    for split in ["Train", "Val", "Test"]:
        split_dir = os.path.join(root_path, split)

        if os.path.exists(split_dir):
            datasets[split] = NWPUDatasetWithAnnotations(
                data_dir=split_dir,
                annotations_dir=annotations_dir,
                transform=transform_dict[split] if split in transform_dict else transform_dict["Test"],
                split=split
            )

            shuffle = (split == "Train")
            g = torch.Generator().manual_seed(42)

            dataloaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=False,
                generator=g,
                collate_fn=nwpu_collate_fn
            )
        else:
            log_info(f"WARNING: Directory {split_dir} not found")

    return dataloaders, datasets


def nwpu_collate_fn(batch):
    """Custom collate function to handle variable-sized annotations.

    Args:
        batch: List of tuples (image, label, path, annotations).

    Returns:
        Tuple of (images_tensor, labels_tensor, paths, annotations_list).
    """
    images = []
    labels = []
    paths = []
    annotations = []

    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        paths.append(item[2])
        annotations.append(item[3])

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels, paths, annotations


if __name__ == "__main__":
    annotations_dir = "/Users/haranaka/Development/FireClassification/NWPU VHR-10 dataset/ground truth"

    log_info("="*60)
    log_info("Testing NWPU VHR-10 annotation loading")
    log_info("="*60)

    all_annotations = load_all_annotations(annotations_dir)
    log_info(f"\nTotal images with annotations: {len(all_annotations)}")

    class_dist = get_class_distribution(all_annotations)
    log_info("\nClass distribution:")
    for class_name, count in sorted(class_dist.items()):
        log_info(f"  {class_name}: {count} objetos")

    test_file = os.path.join(annotations_dir, "002.txt")
    if os.path.exists(test_file):
        log_info(f"\nTeste de parsing: {test_file}")
        annotations = parse_annotation_file(test_file)
        for ann in annotations:
            log_info(f"  - {ann['class_name']}: bbox={ann['bbox']}")