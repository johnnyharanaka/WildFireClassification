"""Tests for core/losses.py — FocalLoss, CenterLoss, and config-driven setup."""
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses import (
    CenterLoss,
    FocalLoss,
    _setup_single_loss,
    get_loss,
    setup_contrastive_learning,
)


# ----------------------------------------------------------------- FocalLoss --
class TestFocalLoss:
    def test_focal_reduces_to_alpha_ce_when_gamma_zero(self):
        """With gamma=0, focal loss == alpha * cross_entropy."""
        logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 1.5, 0.3]])
        targets = torch.tensor([0, 1])

        focal = FocalLoss(alpha=0.5, gamma=0.0, reduction="mean")
        ce = F.cross_entropy(logits, targets)
        assert focal(logits, targets).item() == pytest.approx(0.5 * ce.item(), rel=1e-5)

    def test_focal_downweights_easy_examples_more_than_ce(self):
        """For a confident-but-not-saturated prediction, focal < CE."""
        logits = torch.tensor([[2.0, -1.0]])  # p_t ≈ 0.95, both losses are nonzero
        targets = torch.tensor([0])
        focal = FocalLoss(alpha=1.0, gamma=2.0)(logits, targets)
        ce = F.cross_entropy(logits, targets)
        assert focal.item() > 0.0
        assert focal.item() < ce.item()

    def test_focal_reduction_none_returns_per_sample(self):
        logits = torch.zeros(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = FocalLoss(reduction="none")(logits, targets)
        assert loss.shape == (4,)

    def test_focal_reduction_sum(self):
        logits = torch.zeros(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss_sum = FocalLoss(reduction="sum")(logits, targets)
        loss_none = FocalLoss(reduction="none")(logits, targets)
        assert loss_sum.item() == pytest.approx(loss_none.sum().item(), rel=1e-5)

    def test_focal_loss_is_differentiable(self):
        logits = torch.randn(2, 3, requires_grad=True)
        targets = torch.tensor([0, 2])
        loss = FocalLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# ----------------------------------------------------------------- CenterLoss --
class TestCenterLoss:
    def test_zero_loss_when_embedding_matches_normalized_center(self):
        torch.manual_seed(0)
        cl = CenterLoss(num_classes=3, embedding_size=4)
        # Force a known center, normalize it, then feed it as the embedding
        with torch.no_grad():
            cl.centers.zero_()
            cl.centers[1] = torch.tensor([3.0, 0.0, 0.0, 0.0])  # will L2-normalize → e1

        emb = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # already unit
        labels = torch.tensor([1])
        loss = cl(emb, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_positive_when_misaligned(self):
        cl = CenterLoss(num_classes=2, embedding_size=4)
        emb = F.normalize(torch.randn(8, 4), p=2, dim=1)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        assert cl(emb, labels).item() > 0

    def test_centers_remain_trainable(self):
        cl = CenterLoss(num_classes=2, embedding_size=4)
        assert cl.centers.requires_grad

    def test_lazy_init_when_embedding_size_none(self):
        cl = CenterLoss(num_classes=3, embedding_size=None)
        assert cl.centers is None


# -------------------------------------------------------------------- get_loss --
class TestGetLoss:
    def test_returns_cross_entropy_with_label_smoothing(self):
        def fake(*keys):
            if keys == ("loss", "selected"):
                return "ce"
            if keys == ("loss", "ce"):
                return {"label_smoothing": 0.1}
            return None

        with patch("core.losses.losses.get_config", side_effect=fake):
            loss = get_loss()
        assert isinstance(loss, nn.CrossEntropyLoss)
        assert loss.label_smoothing == pytest.approx(0.1)

    def test_returns_cross_entropy_no_smoothing(self):
        def fake(*keys):
            return "ce" if keys == ("loss", "selected") else None

        with patch("core.losses.losses.get_config", side_effect=fake):
            loss = get_loss()
        assert isinstance(loss, nn.CrossEntropyLoss)
        assert loss.label_smoothing == 0.0

    def test_returns_focal_loss(self):
        def fake(*keys):
            if keys == ("loss", "selected"):
                return "focalLoss"
            if keys == ("loss", "focalLoss"):
                return {"alpha": 0.3, "gamma": 1.5}
            return None

        with patch("core.losses.losses.get_config", side_effect=fake):
            loss = get_loss()
        assert isinstance(loss, FocalLoss)
        assert loss.alpha == pytest.approx(0.3)
        assert loss.gamma == pytest.approx(1.5)

    def test_unknown_loss_raises(self):
        def fake(*keys):
            return "bogus" if keys == ("loss", "selected") else None

        with patch("core.losses.losses.get_config", side_effect=fake):
            with pytest.raises(ValueError, match="Unknown loss"):
                get_loss()


# --------------------------------------------------- setup_contrastive_learning --
class TestSetupContrastiveLearning:
    def test_returns_none_if_no_config(self):
        with patch("core.losses.losses.get_config", return_value=None):
            assert setup_contrastive_learning() is None

    def test_returns_none_if_selected_is_none(self):
        cfg = {"selected": None}
        with patch("core.losses.losses.get_config", return_value=cfg):
            # selected=None → "Single-stage training: Classification only"
            assert setup_contrastive_learning() is None

    def test_supcon_setup(self):
        cfg = {"selected": "supcon", "contrastive_weight": 0.4}

        def fake(*keys):
            if keys == ("contrastive",):
                return cfg
            if keys == ("contrastive", "losses", "supcon"):
                return {"temperature": 0.07, "miner": "none"}
            return None

        with patch("core.losses.losses.get_config", side_effect=fake):
            result = setup_contrastive_learning()
        assert result is not None
        assert result["supcon"] is not None
        assert result["triplet"] is None
        assert result["center"] is None
        assert result["supcon"]["weight"] == pytest.approx(0.4)
        assert result["supcon"]["miner"] is None  # 'none' miner

    def test_triplet_setup_with_miner(self):
        cfg = {"selected": "triplet", "contrastive_weight": 0.5}

        def fake(*keys):
            if keys == ("contrastive",):
                return cfg
            if keys == ("contrastive", "losses", "triplet"):
                return {"margin": 0.3, "miner": "semihard"}
            return None

        with patch("core.losses.losses.get_config", side_effect=fake):
            result = setup_contrastive_learning()
        assert result["triplet"] is not None
        assert result["supcon"] is None
        assert result["triplet"]["miner"] is not None  # mined

    def test_center_setup(self):
        cfg = {"selected": "center", "contrastive_weight": 0.3}

        def fake(*keys):
            if keys == ("contrastive",):
                return cfg
            if keys == ("contrastive", "losses", "center"):
                return {"num_classes": 5, "embedding_size": 64}
            return None

        with patch("core.losses.losses.get_config", side_effect=fake):
            result = setup_contrastive_learning()
        assert result["center"] is not None
        cl = result["center"]["loss_func"]
        assert isinstance(cl, CenterLoss)
        assert cl.num_classes == 5
        assert cl.centers.shape == (5, 64)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown contrastive mode"):
            _setup_single_loss("unknown_mode")
