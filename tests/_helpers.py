"""Shared helpers for spops tests."""

from __future__ import annotations

import pytest
import torch


def make_random_csr(m: int, k: int, density: float, dtype, device, seed: int = 0):
    """Create a random sparse matrix in CSR form alongside its dense equivalent.

    Returns ``(dense, a_val, row_offsets, row_indices, col_indices)`` where
    ``row_indices`` follows the sputnik convention used by spops' CUDA kernels:
    a length-M permutation of rows sorted by descending nnz count (used for
    load balancing). This is the same construction used by RoSA's spops adapter:
    ``torch.argsort(-torch.diff(row_offsets))``.

    Indices are returned as ``int32`` and can be cast down to ``int16`` by
    callers that need the spops fp16/csr_add layout.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dense = torch.randn(m, k, generator=gen, dtype=torch.float32)
    mask = torch.rand(m, k, generator=gen) < density
    dense = dense * mask.float()

    nnz_row, col_idx = mask.nonzero(as_tuple=True)
    counts = torch.bincount(nnz_row, minlength=m)
    row_offs = torch.zeros(m + 1, dtype=torch.int32)
    row_offs[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    row_perm = torch.argsort(-counts).to(torch.int32)

    a_val = dense[mask].to(dtype).to(device)
    return (
        dense.to(dtype).to(device),
        a_val,
        row_offs.to(device),
        row_perm.to(device),
        col_idx.to(torch.int32).to(device),
    )


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
