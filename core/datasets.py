"""Dataset utilities for image classification tasks.

This module provides dataset classes and data loader utilities for handling
various image classification datasets including Fire detection and NWPU VHR-10.
It supports both COCO format and ImageFolder structure, with custom augmentation
capabilities and proper handling of multi-class classification.
"""

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from core.config.config import get_global_seed, get_augmentation_config, log_info


def seed_worker(worker_id):
    """Initialize random seeds for DataLoader workers to ensure reproducibility.

    Args:
        worker_id: Worker process ID assigned by DataLoader.
    """
    seed = get_global_seed() + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def collate_fn_with_bboxes(batch):
    """Custom collate function for batches with variable-length bboxes.

    Handles batches where some samples have bboxes and others don't.
    Bboxes are kept as Python lists (not stacked into tensors).

    Args:
        batch: List of tuples from dataset

    Returns:
        Tuple of (images, labels, paths, bboxes) where bboxes is a list of lists
    """
    images = []
    labels = []
    paths = []
    bboxes = []

    for item in batch:
        if len(item) == 4:
            img, label, path, bbox = item
            images.append(img)
            labels.append(label)
            paths.append(path)
            bboxes.append(bbox if bbox else [])
        else:
            # Fallback para batches sem bboxes
            images.append(item[0])
            labels.append(item[1])
            paths.append(item[2])
            bboxes.append([])

    # Stack images and labels normally
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels, paths, bboxes


class PadResize:
    """Resize transform that maintains aspect ratio and adds padding to achieve square size.

    This transform preserves all image information without cropping by resizing to fit
    within the target size while maintaining aspect ratio, then adding padding to create
    a square image.

    Attributes:
        size: Target size for both height and width.
        fill: Padding fill value (0 for black, (0,0,0) for black RGB).
    """

    def __init__(self, size=224, fill=0):
        """Initialize PadResize transform.

        Args:
            size: Target final size (height and width). Defaults to 224.
            fill: Padding fill value. Defaults to 0 (black).
        """
        self.size = size
        self.fill = fill

    def __call__(self, img):
        """Apply padding and resize to image.

        Args:
            img: PIL Image to transform.

        Returns:
            PIL Image with size (size, size).
        """
        w, h = img.size
        aspect_ratio = w / h

        if aspect_ratio > 1:
            new_w = self.size
            new_h = int(self.size / aspect_ratio)
        else:
            new_h = self.size
            new_w = int(self.size * aspect_ratio)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )

        img = transforms.functional.pad(img, padding, fill=self.fill, padding_mode='constant')

        return img


class COCOFormatDataset(torch.utils.data.Dataset):
    """Dataset handler for COCO format (images/ + annotations.json).

    Supports both binary Fire dataset and multi-class NWPU VHR-10 dataset.
    Automatically detects dataset type based on category structure and applies
    appropriate label mapping.

    Attributes:
        root: Root directory containing images/ and annotations.json.
        transform: Optional transform to apply to images.
        is_nwpu: Whether this is an NWPU VHR-10 dataset.
        is_fire: Whether this is a Fire detection dataset.
        classes: List of class names.
        class_to_idx: Mapping from class name to index.
        samples: List of (filepath, label) tuples.
        targets: List of labels for all samples.
    """

    def __init__(self, root, transform=None):
        """Initialize COCO format dataset.

        Args:
            root: Root directory containing images/ or Images/ folder and annotations.json.
            transform: Optional torchvision transform to apply to images.
        """
        self.root = root
        self.transform = transform

        annotations_path = os.path.join(root, "annotations.json")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        self.id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        num_classes = len(self.categories)
        category_names = set(self.categories.values())

        NWPU_CLASSES = {"airplane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
                        "basketball_court", "ground_track_field", "harbor", "bridge", "vehicle", "background"}
        DIOR_CLASSES = {"airplane", "airport", "baseball_diamond", "basketball_court", "bridge",
                        "chaparral", "dense_residential", "forest", "freeway", "golf_course",
                        "ground_track_field", "harbor", "industrial", "intersection", "medium_residential",
                        "mobile_home_park", "overpass", "parking_lot", "sparse_residential", "tennis_court"}

        is_nwpu = category_names == NWPU_CLASSES
        is_dior = category_names == DIOR_CLASSES or num_classes == 20
        is_fire = category_names == {"Fire", "Not"} or (num_classes == 2 and not is_dior)

        self.is_nwpu = is_nwpu
        self.is_dior = is_dior
        self.is_fire = is_fire

        self.image_to_class = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            class_id = ann['category_id']
            if image_id not in self.image_to_class:
                self.image_to_class[image_id] = class_id

        self.samples = []
        images_dir = os.path.join(root, "images") if os.path.exists(os.path.join(root, "images")) else os.path.join(root, "Images")

        if is_nwpu:
            NWPU_CLASS_MAPPING = {
                "airplane": 0,
                "ship": 1,
                "storage_tank": 2,
                "baseball_diamond": 3,
                "tennis_court": 4,
                "basketball_court": 5,
                "ground_track_field": 6,
                "harbor": 7,
                "bridge": 8,
                "vehicle": 9,
                "background": 10
            }

            self.classes = list(NWPU_CLASS_MAPPING.keys())
            self.class_to_idx = NWPU_CLASS_MAPPING

            self.coco_id_to_nwpu_idx = {}
            for coco_id, class_name in self.categories.items():
                self.coco_id_to_nwpu_idx[coco_id] = NWPU_CLASS_MAPPING[class_name]

            for img_id, filename in self.id_to_filename.items():
                filepath = os.path.join(images_dir, filename)
                if os.path.exists(filepath):
                    coco_class_id = self.image_to_class.get(img_id, None)
                    if coco_class_id is not None:
                        label = self.coco_id_to_nwpu_idx[coco_class_id]
                        self.samples.append((filepath, label))

            self.targets = [s[1] for s in self.samples]
            log_info(f"COCO NWPU dataset loaded: {len(self.samples)} images")
            for class_name in self.classes:
                count = sum(1 for s in self.samples if s[1] == NWPU_CLASS_MAPPING[class_name])
                log_info(f"  {class_name}: {count}")

        elif is_dior:
            DIOR_CLASS_MAPPING = {
                "airplane": 0,
                "airport": 1,
                "baseball_diamond": 2,
                "basketball_court": 3,
                "bridge": 4,
                "chaparral": 5,
                "dense_residential": 6,
                "forest": 7,
                "freeway": 8,
                "golf_course": 9,
                "ground_track_field": 10,
                "harbor": 11,
                "industrial": 12,
                "intersection": 13,
                "medium_residential": 14,
                "mobile_home_park": 15,
                "overpass": 16,
                "parking_lot": 17,
                "sparse_residential": 18,
                "tennis_court": 19
            }

            self.classes = list(DIOR_CLASS_MAPPING.keys())
            self.class_to_idx = DIOR_CLASS_MAPPING

            self.coco_id_to_dior_idx = {}
            for coco_id, class_name in self.categories.items():
                self.coco_id_to_dior_idx[coco_id] = DIOR_CLASS_MAPPING[class_name]

            for img_id, filename in self.id_to_filename.items():
                filepath = os.path.join(images_dir, filename)
                if os.path.exists(filepath):
                    coco_class_id = self.image_to_class.get(img_id, None)
                    if coco_class_id is not None:
                        label = self.coco_id_to_dior_idx[coco_class_id]
                        self.samples.append((filepath, label))

            self.targets = [s[1] for s in self.samples]
            log_info(f"COCO DIOR dataset loaded: {len(self.samples)} images")
            for class_name in self.classes:
                count = sum(1 for s in self.samples if s[1] == DIOR_CLASS_MAPPING[class_name])
                log_info(f"  {class_name}: {count}")

        else:
            images_with_annotations = set()
            for ann in coco_data['annotations']:
                images_with_annotations.add(ann['image_id'])

            self.classes = ["Fire", "Not"]
            self.class_to_idx = {"Fire": 0, "Not": 1}

            # Create reverse mapping: filename -> img_id
            filename_to_id = {v: k for k, v in self.id_to_filename.items()}

            # Iterate over all files in images/ folder (not just JSON entries)
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            for filename in os.listdir(images_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in valid_extensions:
                    continue

                filepath = os.path.join(images_dir, filename)
                if os.path.isfile(filepath):
                    # Check if image has annotation (Fire) or not (Not)
                    img_id = filename_to_id.get(filename)
                    label = 0 if img_id is not None and img_id in images_with_annotations else 1
                    self.samples.append((filepath, label))

            self.targets = [s[1] for s in self.samples]
            counts = {cls: sum(1 for s in self.samples if s[1] == idx) for cls, idx in self.class_to_idx.items()}
            counts_str = ", ".join(f"{cls}: {n}" for cls, n in counts.items())
            split_name = os.path.basename(root)
            log_info(f"[{split_name}] COCO dataset: {len(self.samples)} images ({counts_str})")

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image, label, path) where image is the transformed PIL Image.
        """
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, path


class ImageFolderWithPaths(ImageFolder):
    """Extended ImageFolder that returns file paths and supports NWPU class mapping.

    This class extends torchvision's ImageFolder to return the image path along with
    the image and label. It also automatically detects and applies the official NWPU
    VHR-10 class mapping when appropriate.

    Attributes:
        classes: List of class names in the correct order.
        class_to_idx: Mapping from class name to index.
        samples: List of (path, label) tuples.
        targets: List of labels for all samples.
    """

    def __init__(self, root, transform=None, target_transform=None):
        """Initialize ImageFolder with paths and dataset-specific mapping support.

        Args:
            root: Root directory path containing class subdirectories.
            transform: Optional transform to apply to images.
            target_transform: Optional transform to apply to labels.
        """
        NWPU_CLASS_MAPPING = {
            "airplane": 0,
            "ship": 1,
            "storage_tank": 2,
            "baseball_diamond": 3,
            "tennis_court": 4,
            "basketball_court": 5,
            "ground_track_field": 6,
            "harbor": 7,
            "bridge": 8,
            "vehicle": 9,
            "background": 10
        }

        DIOR_CLASS_MAPPING = {
            "airplane": 0,
            "airport": 1,
            "baseball_diamond": 2,
            "basketball_court": 3,
            "bridge": 4,
            "chaparral": 5,
            "dense_residential": 6,
            "forest": 7,
            "freeway": 8,
            "golf_course": 9,
            "ground_track_field": 10,
            "harbor": 11,
            "industrial": 12,
            "intersection": 13,
            "medium_residential": 14,
            "mobile_home_park": 15,
            "overpass": 16,
            "parking_lot": 17,
            "sparse_residential": 18,
            "tennis_court": 19
        }

        # Detect dataset type and apply mapping if needed
        potential_classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        is_nwpu = all(cls in NWPU_CLASS_MAPPING for cls in potential_classes) if potential_classes else False
        is_dior = all(cls in DIOR_CLASS_MAPPING for cls in potential_classes) if potential_classes else False

        super().__init__(root, transform=transform, target_transform=target_transform)
        mapping = NWPU_CLASS_MAPPING if is_nwpu else (DIOR_CLASS_MAPPING if is_dior else None)
        dataset_name = "NWPU" if is_nwpu else ("DIOR (20 classes)" if is_dior else None)

        if mapping:
            self.classes = [cls for cls in mapping.keys() if cls in potential_classes]

            old_class_to_idx = self.class_to_idx.copy()
            self.class_to_idx = {cls: mapping[cls] for cls in self.classes}

            new_samples = []
            for path, old_label in self.samples:
                class_name = self.classes[list(old_class_to_idx.values()).index(old_label)]
                new_label = self.class_to_idx[class_name]
                new_samples.append((path, new_label))
            self.samples = new_samples
            self.targets = [s[1] for s in self.samples]

            log_info(f"ImageFolder with {dataset_name} mapping applied")

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image, label, path) where image is the transformed PIL Image.
        """
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def _get_data_transforms(train_size=224, eval_size=None):
    """Create data transforms for training and evaluation based on config.yaml.

    Args:
        train_size: Target image size for training. Defaults to 224.
        eval_size: Target image size for val/test. Defaults to train_size.

    Returns:
        Dictionary mapping split names to transform compositions.
    """
    if eval_size is None:
        eval_size = train_size

    aug_cfg = get_augmentation_config("standard") or {}
    norm_cfg = get_augmentation_config("normalization") or {}
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    # Build train transforms from config
    train_transforms = []

    # PadResize (maintains aspect ratio + padding)
    train_transforms.append(PadResize(train_size))

    # RandomHorizontalFlip
    hflip = aug_cfg.get("random_horizontal_flip", {})
    if hflip.get("enabled", True):
        train_transforms.append(transforms.RandomHorizontalFlip(p=hflip.get("probability", 0.5)))

    # RandomVerticalFlip
    vflip = aug_cfg.get("random_vertical_flip", {})
    if vflip.get("enabled", False):
        train_transforms.append(transforms.RandomVerticalFlip(p=vflip.get("probability", 0.5)))

    # ColorJitter
    cj = aug_cfg.get("color_jitter", {})
    if cj.get("enabled", False):
        jitter = transforms.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.1),
        )
        train_transforms.append(transforms.RandomApply([jitter], p=cj.get("probability", 0.2)))

    # GaussianBlur
    gb = aug_cfg.get("gaussian_blur", {})
    if gb.get("enabled", False):
        train_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=gb.get("probability", 0.1)))

    # RandomSolarize
    sol = aug_cfg.get("random_solarize", {})
    if sol.get("enabled", False):
        train_transforms.append(transforms.RandomSolarize(threshold=sol.get("threshold", 128), p=sol.get("probability", 0.2)))

    # RandomGrayscale
    gs = aug_cfg.get("random_grayscale", {})
    if gs.get("enabled", False):
        train_transforms.append(transforms.RandomGrayscale(p=gs.get("probability", 0.2)))

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Val/Test: PadResize + Normalize (no augmentation)
    eval_transforms = transforms.Compose([
        PadResize(eval_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_transforms = {
        "Train": transforms.Compose(train_transforms),
        "Test": eval_transforms,
        "Val": eval_transforms,
    }
    return data_transforms


def get_data_loaders(root_path: str = "data/", batch_size: int = 4, num_workers: int = 0, img_size: int = 224, eval_size: int = None, splits=None):
    """Create data loaders for the specified splits.

    Automatically detects dataset format (COCO or ImageFolder) and creates appropriate
    dataset instances. Supports both Fire detection and NWPU VHR-10 datasets.

    Args:
        root_path: Root directory containing Train/, Test/, and Val/ subdirectories.
        batch_size: Number of samples per batch. Defaults to 4.
        num_workers: Number of worker processes for data loading. Defaults to 0.
        img_size: Target image size for training. Defaults to 224.
        eval_size: Target image size for val/test. Defaults to img_size.
        splits: List of splits to load. Defaults to ["Train", "Test", "Val"].

    Returns:
        Tuple of (data_loaders, image_datasets) where data_loaders is a dict of
        DataLoader instances and image_datasets is a dict of Dataset instances.
    """
    if splits is None:
        splits = ["Train", "Test", "Val"]

    root_path = os.path.abspath(root_path)
    _data_transforms = _get_data_transforms(img_size, eval_size)

    image_datasets = {}

    for split in splits:
        split_path = os.path.join(root_path, split)

        images_dir_lower = os.path.join(split_path, "images")
        images_dir_upper = os.path.join(split_path, "Images")
        annotations_path = os.path.join(split_path, "annotations.json")

        if (os.path.exists(images_dir_lower) or os.path.exists(images_dir_upper)) and os.path.exists(annotations_path):
            image_datasets[split] = COCOFormatDataset(
                split_path,
                _data_transforms[split]
            )
        else:
            image_datasets[split] = ImageFolderWithPaths(
                split_path,
                _data_transforms[split]
            )

    g = torch.Generator().manual_seed(get_global_seed())
    data_loaders = {
        x: DataLoader(
            image_datasets[x],
            shuffle=(x == "Train"),
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=False,
        )
        for x in splits
    }
    return data_loaders, image_datasets
