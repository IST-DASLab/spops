"""Unit tests for spops.sddmm on CPU and CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spops = pytest.importorskip("spops")

from tests._helpers import make_random_csr, require_cuda  # noqa: E402


def _reference_sddmm(
    mask_row_offsets: torch.Tensor,
    mask_col_indices: torch.Tensor,
    a: torch.Tensor,
    bt: torch.Tensor,
) -> torch.Tensor:
    """Dense reference: out[k] = <A[i, :], BT[j, :]> for the k-th (i, j) nonzero."""
    m = mask_row_offsets.shape[0] - 1
    nnz = mask_col_indices.shape[0]
    out = torch.zeros(nnz, dtype=torch.float32)
    a32 = a.to(torch.float32)
    bt32 = bt.to(torch.float32)
    for i in range(m):
        start = int(mask_row_offsets[i].item())
        end = int(mask_row_offsets[i + 1].item())
        for k in range(start, end):
            j = int(mask_col_indices[k].item())
            out[k] = torch.dot(a32[i], bt32[j])
    return out


@pytest.mark.parametrize("backend", ["sddmm", "sddmm_v2", "sddmm_v3"])
@pytest.mark.parametrize("shape", [(4, 8, 5), (16, 32, 8)])
def test_sddmm_cpu_matches_reference(backend, shape):
    m, n, k = shape  # A: (m, k), B^T: (n, k), output sparsity over (m, n)
    _, _, row_offs, row_idx, col_idx = make_random_csr(
        m, n, density=0.4, dtype=torch.float32, device="cpu"
    )
    a = torch.randn(m, k, dtype=torch.float32)
    bt = torch.randn(n, k, dtype=torch.float32)

    out = spops.sddmm(row_offs, row_idx, col_idx, a, bt, backend=backend)
    expected = _reference_sddmm(row_offs, col_idx, a, bt)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_sddmm_cpu_empty_mask():
    m, n, k = 3, 4, 5
    row_offs = torch.zeros(m + 1, dtype=torch.int32)
    row_idx = torch.zeros(0, dtype=torch.int32)
    col_idx = torch.zeros(0, dtype=torch.int32)
    a = torch.randn(m, k)
    bt = torch.randn(n, k)

    out = spops.sddmm(row_offs, row_idx, col_idx, a, bt)
    assert out.numel() == 0


def test_sddmm_cpu_rejects_int_dtypes():
    m, n, k = 2, 2, 2
    row_offs = torch.tensor([0, 1, 2], dtype=torch.int32)
    row_idx = torch.tensor([0, 1], dtype=torch.int32)
    col_idx = torch.tensor([0, 1], dtype=torch.int32)
    a = torch.ones(m, k, dtype=torch.int32)
    bt = torch.ones(n, k, dtype=torch.float32)
    with pytest.raises(AssertionError):
        spops.sddmm(row_offs, row_idx, col_idx, a, bt)


def test_sddmm_cpu_return_type_cast():
    m, n, k = 4, 4, 4
    _, _, row_offs, row_idx, col_idx = make_random_csr(m, n, 0.5, torch.float32, "cpu")
    a = torch.randn(m, k)
    bt = torch.randn(n, k)
    out = spops.sddmm(row_offs, row_idx, col_idx, a, bt, return_type=torch.float16)
    assert out.dtype == torch.float16


@pytest.mark.parametrize("backend", ["sputnik", "structure_aware"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sddmm_cuda_matches_reference(backend, dtype):
    require_cuda()
    m, n, k = 32, 32, 16
    _, _, row_offs, row_perm, col_idx = make_random_csr(
        m, n, density=0.3, dtype=torch.float32, device="cuda"
    )
    if backend == "structure_aware":
        # The structure_aware kernel takes int16 indices directly, with no
        # implicit casting in the Python wrapper.
        row_perm = row_perm.to(torch.int16)
        col_idx = col_idx.to(torch.int16)

    a = torch.randn(m, k, dtype=dtype, device="cuda")
    bt = torch.randn(n, k, dtype=dtype, device="cuda")

    out = spops.sddmm(row_offs, row_perm, col_idx, a, bt, backend=backend)
    expected = _reference_sddmm(row_offs.cpu(), col_idx.cpu(), a.cpu(), bt.cpu())
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(out.cpu().float(), expected, atol=tol, rtol=tol)
