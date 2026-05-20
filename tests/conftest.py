"""Pytest fixtures for spops tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    return torch.cuda.is_available()
