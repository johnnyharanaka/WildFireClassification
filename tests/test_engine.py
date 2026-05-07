"""Tests for core/engine.py — train_one_epoch, get_device, get_training_stages."""
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from core.training.engine import get_device, get_training_stages, train_one_epoch


# --------------------------------------------------------------- helpers --
class _PlainModel(nn.Module):
    def __init__(self, in_dim=4, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.fc(x)


class _EmbeddingModel(nn.Module):
    def __init__(self, in_dim=4, embed_dim=8, n_classes=3):
        super().__init__()
        self.backbone = nn.Linear(in_dim, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_embedding=False):
        if x.dim() > 2:
            x = x.flatten(1)
        emb = self.backbone(x)
        logits = self.head(emb)
        if return_embedding:
            return logits, emb
        return logits


class _NoOpScheduler:
    def step(self):
        pass


def _make_loader(n_batches=2, batch_size=4, in_dim=4, n_classes=3):
    batches = []
    for b in range(n_batches):
        feats = torch.randn(batch_size, in_dim)
        labels = torch.tensor([i % n_classes for i in range(batch_size)])
        paths = [f"img_{b}_{i}.jpg" for i in range(batch_size)]
        batches.append((feats, labels, paths))
    return batches


# --------------------------------------------------------------- get_device --
class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")


# --------------------------------------------------------- training_stages --
class TestGetTrainingStages:
    def test_single_stage_when_no_contrastive(self):
        with patch("core.training.engine.get_config", return_value=None):
            stages = get_training_stages(10, 5)
        assert stages["mode"] == "single"
        assert stages["epochs_stage1"] == 0
        assert stages["epochs_stage2"] == 5
        assert stages["contrastive_type"] is None

    def test_two_stage_when_contrastive_selected(self):
        with patch("core.training.engine.get_config", return_value="supcon"):
            stages = get_training_stages(10, 5)
        assert stages["mode"] == "two-stage"
        assert stages["epochs_stage1"] == 10
        assert stages["epochs_stage2"] == 5
        assert stages["contrastive_type"] == "supcon"


# ----------------------------------------------------------- train_one_epoch --
class TestTrainOneEpoch:
    def test_classification_only_path_updates_weights(self):
        torch.manual_seed(0)
        model = _PlainModel()
        loader = _make_loader(n_batches=2, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        weights_before = model.fc.weight.clone()

        with patch("core.training.engine.get_config", return_value=None):
            mean_loss, mean_acc = train_one_epoch(
                loader,
                "cpu",
                optimizer,
                _NoOpScheduler(),
                model,
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=None,
            )

        assert mean_loss > 0
        assert 0.0 <= mean_acc <= 1.0
        # Weights changed → optimizer step ran
        assert not torch.allclose(weights_before, model.fc.weight)

    def test_classification_disabled_no_contrastive_yields_zero_loss(self):
        model = _PlainModel()
        loader = _make_loader(n_batches=1, batch_size=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with patch("core.training.engine.get_config", return_value=None):
            mean_loss, _ = train_one_epoch(
                loader,
                "cpu",
                optimizer,
                _NoOpScheduler(),
                model,
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=None,
                use_classification_loss=False,
            )
        assert mean_loss == pytest.approx(0.0)

    def test_classification_weight_applied_when_contrastive_active(self):
        """cls_weight from config is applied only when at least one contrastive loss is active."""
        model = _EmbeddingModel()
        loader = _make_loader(n_batches=1, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: no weight changes

        contrastive_losses = {
            "supcon": {
                "loss_func": lambda emb, lbl: torch.tensor(0.5, requires_grad=True),
                "weight": 1.0,
                "miner": None,
            },
            "triplet": None,
            "center": None,
        }

        def fake(*keys):
            if keys == ("contrastive", "classification_weight"):
                return 3.0
            return None

        with patch("core.training.engine.get_config", side_effect=fake):
            mean_loss_a, _ = train_one_epoch(
                loader,
                "cpu",
                optimizer,
                _NoOpScheduler(),
                model,
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=contrastive_losses,
            )

        # Same setup but classification_weight=1.0
        torch.manual_seed(0)
        model2 = _EmbeddingModel()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.0)

        def fake_one(*keys):
            if keys == ("contrastive", "classification_weight"):
                return 1.0
            return None

        with patch("core.training.engine.get_config", side_effect=fake_one):
            mean_loss_b, _ = train_one_epoch(
                loader,
                "cpu",
                optimizer2,
                _NoOpScheduler(),
                model2,
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=contrastive_losses,
            )

        # Weight=3.0 produces strictly larger total loss than weight=1.0
        assert mean_loss_a > mean_loss_b

    def test_triplet_path_invokes_loss_with_embeddings(self):
        model = _EmbeddingModel()
        loader = _make_loader(n_batches=1, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        triplet_calls = []

        def triplet_fn(emb, lbl):
            triplet_calls.append(emb.shape)
            return torch.tensor(0.2, requires_grad=True)

        contrastive_losses = {
            "supcon": None,
            "triplet": {"loss_func": triplet_fn, "weight": 1.0, "miner": None},
            "center": None,
        }

        with patch("core.training.engine.get_config", return_value=None):
            train_one_epoch(
                loader,
                "cpu",
                optimizer,
                _NoOpScheduler(),
                model,
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=contrastive_losses,
                use_classification_loss=False,
            )

        assert len(triplet_calls) == 1
        # Embeddings (batch=4, embed_dim=8) and L2-normalized in engine
        assert triplet_calls[0] == (4, 8)
