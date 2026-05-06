"""Shared fixtures for the test suite."""
import os
import sys

import numpy as np
import pytest
import torch

# Ensure the project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(autouse=True)
def deterministic_rng():
    """Reset RNGs before each test for reproducibility."""
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def tiny_logits():
    """A 4-sample, 3-class logits tensor with known argmax targets [0,1,2,1]."""
    return torch.tensor(
        [
            [3.0, 0.5, -1.0],
            [0.1, 2.5, 0.2],
            [-0.5, 0.1, 4.0],
            [0.3, 1.7, 0.4],
        ]
    )


@pytest.fixture
def tiny_labels():
    return torch.tensor([0, 1, 2, 1], dtype=torch.long)
