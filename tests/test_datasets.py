"""Tests for core/datasets.py — transforms, padding, seeding, collation."""
import os
import random as _random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from core.datasets import (
    PadResize,
    _get_data_transforms,
    collate_fn_with_bboxes,
    seed_worker,
)


# ----------------------------------------------------------------- PadResize --
class TestPadResize:
    def test_square_image_unchanged_dims(self):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        out = PadResize(64)(img)
        assert out.size == (64, 64)

    def test_landscape_padded_vertically(self):
        # 200x100 → fit width to 64, height becomes 32, pad 16+16 vertically
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        out = PadResize(64)(img)
        assert out.size == (64, 64)
        arr = np.array(out)
        # Top and bottom rows should be padding (fill=0 by default)
        assert arr[0].sum() == 0
        assert arr[-1].sum() == 0
        # Middle row should still be red
        assert arr[32, 32, 0] > 200  # center pixel keeps red channel

    def test_portrait_padded_horizontally(self):
        img = Image.new("RGB", (100, 200), color=(0, 255, 0))
        out = PadResize(64)(img)
        assert out.size == (64, 64)
        arr = np.array(out)
        assert arr[:, 0].sum() == 0
        assert arr[:, -1].sum() == 0

    def test_custom_fill_value(self):
        img = Image.new("RGB", (200, 100))
        out = PadResize(64, fill=128)(img)
        arr = np.array(out)
        assert arr[0, 0, 0] == 128


# ---------------------------------------------------------------- seed_worker --
class TestSeedWorker:
    def test_seed_worker_seeds_all_libs(self):
        # After calling seed_worker, the three RNGs should be seeded by base+worker_id
        with patch("core.datasets.datasets.get_global_seed", return_value=100):
            seed_worker(7)
            np_val = np.random.rand()
            py_val = _random.random()
            torch_val = torch.rand(1).item()

            seed_worker(7)
            assert np.random.rand() == np_val
            assert _random.random() == py_val
            assert torch.rand(1).item() == torch_val

    def test_different_workers_get_different_seeds(self):
        with patch("core.datasets.datasets.get_global_seed", return_value=42):
            seed_worker(0)
            v0 = np.random.rand()
            seed_worker(1)
            v1 = np.random.rand()
            assert v0 != v1


# ------------------------------------------------------ _get_data_transforms --
class TestGetDataTransforms:
    def test_default_transforms_have_train_and_eval_keys(self):
        with patch("core.datasets.datasets.get_augmentation_config", return_value=None):
            t = _get_data_transforms(train_size=64)
        assert {"Train", "Val", "Test"} <= set(t)

    def test_eval_transforms_are_deterministic(self):
        """Eval pipeline = PadResize + ToTensor + Normalize. No randomness."""
        with patch("core.datasets.datasets.get_augmentation_config", return_value=None):
            t = _get_data_transforms(train_size=64)
        img = Image.new("RGB", (80, 60), color=(120, 130, 140))
        out1 = t["Val"](img)
        out2 = t["Val"](img)
        assert torch.allclose(out1, out2)
        assert out1.shape == (3, 64, 64)

    def test_eval_size_overrides_train_size(self):
        with patch("core.datasets.datasets.get_augmentation_config", return_value=None):
            t = _get_data_transforms(train_size=32, eval_size=128)
        img = Image.new("RGB", (40, 40))
        assert t["Train"](img).shape == (3, 32, 32)
        assert t["Val"](img).shape == (3, 128, 128)

    def test_horizontal_flip_disabled_by_config(self):
        """When hflip is disabled, the train transform should not contain a flip op."""
        aug = {"random_horizontal_flip": {"enabled": False}}
        with patch("core.datasets.datasets.get_augmentation_config", side_effect=lambda key: aug if key == "standard" else None):
            t = _get_data_transforms(train_size=32)
        op_types = [type(op).__name__ for op in t["Train"].transforms]
        assert "RandomHorizontalFlip" not in op_types

    def test_optional_augs_added_when_enabled(self):
        aug = {
            "random_horizontal_flip": {"enabled": False},
            "random_vertical_flip": {"enabled": True, "probability": 0.5},
            "color_jitter": {"enabled": True, "probability": 0.2},
            "gaussian_blur": {"enabled": True, "probability": 0.1},
            "random_solarize": {"enabled": True, "threshold": 128, "probability": 0.2},
            "random_grayscale": {"enabled": True, "probability": 0.2},
        }
        with patch("core.datasets.datasets.get_augmentation_config", side_effect=lambda key: aug if key == "standard" else None):
            t = _get_data_transforms(train_size=32)
        op_types = [type(op).__name__ for op in t["Train"].transforms]
        assert "RandomVerticalFlip" in op_types
        assert "RandomGrayscale" in op_types
        assert "RandomSolarize" in op_types
        # ColorJitter and GaussianBlur are wrapped in RandomApply
        assert op_types.count("RandomApply") >= 2

    def test_custom_normalization_used(self):
        norm = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        with patch(
            "core.datasets.datasets.get_augmentation_config",
            side_effect=lambda key: norm if key == "normalization" else None,
        ):
            t = _get_data_transforms(train_size=8)
        # A pure-gray image at 0.5 should map to ~0 after this normalization
        img = Image.new("RGB", (8, 8), color=(128, 128, 128))
        tensor = t["Val"](img)
        # 128/255 ≈ 0.502, then (0.502-0.5)/0.5 ≈ 0.004
        assert tensor.abs().max().item() < 0.05


# -------------------------------------------------- collate_fn_with_bboxes --
class TestCollateFn:
    def _make_sample(self, label, bbox=None):
        img = torch.zeros(3, 8, 8)
        path = f"img_{label}.jpg"
        if bbox is None:
            return (img, label, path)
        return (img, label, path, bbox)

    def test_4_tuple_with_bboxes(self):
        batch = [
            self._make_sample(0, [[0, 0, 10, 10, 1]]),
            self._make_sample(1, [[20, 20, 30, 30, 2]]),
        ]
        images, labels, paths, bboxes = collate_fn_with_bboxes(batch)
        assert images.shape == (2, 3, 8, 8)
        assert labels.tolist() == [0, 1]
        assert paths == ["img_0.jpg", "img_1.jpg"]
        assert len(bboxes) == 2 and bboxes[0] == [[0, 0, 10, 10, 1]]

    def test_3_tuple_fallback(self):
        batch = [self._make_sample(0), self._make_sample(1)]
        images, labels, paths, bboxes = collate_fn_with_bboxes(batch)
        assert images.shape == (2, 3, 8, 8)
        assert labels.dtype == torch.long
        assert all(b == [] for b in bboxes)

    def test_falsy_bbox_replaced_with_empty_list(self):
        batch = [self._make_sample(0, []), self._make_sample(1, [[0, 0, 1, 1, 1]])]
        _, _, _, bboxes = collate_fn_with_bboxes(batch)
        assert bboxes[0] == [] and bboxes[1] == [[0, 0, 1, 1, 1]]
