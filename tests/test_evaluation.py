"""Tests for core/evaluation.py — eval_fn and val_epoch."""
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from core.evaluation import eval_fn, val_epoch


# --------------------------------------------------------------- helpers --
class _PlainModel(nn.Module):
    """Tiny linear model returning logits — no embedding head."""

    def __init__(self, in_dim=4, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.fc(x)


class _EmbeddingModel(nn.Module):
    """Model that supports return_embedding=True for contrastive paths."""

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


def _make_loader(n_batches=2, batch_size=4, in_dim=4, n_classes=3):
    """Yields (features, labels, paths) tuples like the real datasets."""
    batches = []
    for b in range(n_batches):
        feats = torch.randn(batch_size, in_dim)
        labels = torch.tensor([i % n_classes for i in range(batch_size)])
        paths = [f"img_{b}_{i}.jpg" for i in range(batch_size)]
        batches.append((feats, labels, paths))
    return batches


# --------------------------------------------------------------- eval_fn --
class TestEvalFn:
    def test_returns_outputs_preds_accuracy(self):
        model = _PlainModel()
        feats = torch.randn(5, 4)
        targets = torch.tensor([0, 1, 2, 0, 1])
        outputs, preds, acc = eval_fn(model, feats, targets)
        assert outputs.shape == (5, 3)
        assert preds.shape == (5,)
        assert 0.0 <= acc.item() <= 1.0

    def test_perfect_model_gives_full_accuracy(self):
        # Identity-like model: argmax of one-hot inputs should equal label
        model = _PlainModel(in_dim=3, n_classes=3)
        with torch.no_grad():
            model.fc.weight.copy_(torch.eye(3))
            model.fc.bias.zero_()
        feats = torch.eye(3)
        targets = torch.tensor([0, 1, 2])
        _, preds, acc = eval_fn(model, feats, targets)
        assert preds.tolist() == [0, 1, 2]
        assert acc.item() == pytest.approx(1.0)


# --------------------------------------------------------------- val_epoch --
class TestValEpoch:
    def test_pure_classification_path(self):
        """No contrastive losses → standard CE val loop, returns (loss, acc)."""
        model = _PlainModel()
        loader = _make_loader()
        criterion = nn.CrossEntropyLoss()

        with patch("core.evaluation.get_config", return_value=None):
            mean_loss, mean_acc = val_epoch(
                model, loader, "cpu", criterion, epoch=0, total_epoch=1
            )

        assert mean_loss > 0
        assert 0.0 <= mean_acc <= 1.0

    def test_classification_loss_disabled_returns_zero_cls_loss(self):
        model = _PlainModel()
        loader = _make_loader(n_batches=1, batch_size=2)
        criterion = nn.CrossEntropyLoss()
        with patch("core.evaluation.get_config", return_value=None):
            mean_loss, _ = val_epoch(
                model,
                loader,
                "cpu",
                criterion,
                epoch=0,
                total_epoch=1,
                use_classification_loss=False,
            )
        # No contrastive, no classification → loss is 0
        assert mean_loss == pytest.approx(0.0)

    def test_classification_weight_applied_when_contrastive_active(self):
        """When contrastive_losses dict is provided, criterion is scaled by classification_weight."""
        model = _EmbeddingModel()
        loader = _make_loader(n_batches=1, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Mock contrastive loss func: returns a fixed scalar
        contrastive_losses = {
            "supcon": {
                "loss_func": lambda emb, lbl: torch.tensor(0.5, requires_grad=True),
                "weight": 1.0,
                "miner": None,
            },
            "triplet": None,
            "center": None,
        }

        # classification_weight=2.0
        def fake(*keys):
            if keys == ("contrastive", "classification_weight"):
                return 2.0
            return None

        with patch("core.evaluation.get_config", side_effect=fake):
            mean_loss, _ = val_epoch(
                model,
                loader,
                "cpu",
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=contrastive_losses,
            )

        # Loss = 2.0 * cls_loss + 1.0 * 0.5 → must be > 0.5
        assert mean_loss > 0.5

    def test_center_loss_path(self):
        """Center contrastive uses (embeddings, labels) and skips the miner."""
        model = _EmbeddingModel()
        loader = _make_loader(n_batches=1, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        center_calls = []

        def center_fn(emb, lbl):
            center_calls.append((emb.shape, lbl.shape))
            return torch.tensor(0.25, requires_grad=True)

        contrastive_losses = {
            "supcon": None,
            "triplet": None,
            "center": {"loss_func": center_fn, "weight": 0.5, "miner": None},
        }
        with patch("core.evaluation.get_config", return_value=None):
            val_epoch(
                model,
                loader,
                "cpu",
                criterion,
                epoch=0,
                total_epoch=1,
                contrastive_losses=contrastive_losses,
                use_classification_loss=False,
            )
        assert len(center_calls) == 1
        # Embedding flattened to (batch, embed_dim)
        assert center_calls[0][0][0] == 4
