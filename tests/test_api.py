"""Top-level API smoke tests."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
spops = pytest.importorskip("spops")


def test_public_api_exposed():
    for name in ("spmm", "sddmm", "csr_add", "csr_transpose", "set_num_threads"):
        assert hasattr(spops, name), f"missing public symbol: {name}"


def test_set_num_threads_sets_env():
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        spops.set_num_threads(3)
        assert os.environ["OMP_NUM_THREADS"] == "3"
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


def test_backends_importable():
    """The native extensions must load (the CPU one is always required)."""
    import spops_backend_cpu  # noqa: F401

    if torch.cuda.is_available():
        import spops_backend  # noqa: F401
