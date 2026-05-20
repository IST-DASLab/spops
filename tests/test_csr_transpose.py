"""Unit tests for spops.csr_transpose."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spops = pytest.importorskip("spops")

from tests._helpers import make_random_csr, require_cuda  # noqa: E402


def _csr_to_dense(a_val, row_offs, col_idx, m, n):
    dense = torch.zeros(m, n, dtype=a_val.dtype, device=a_val.device)
    for i in range(m):
        for k in range(int(row_offs[i].item()), int(row_offs[i + 1].item())):
            dense[i, int(col_idx[k].item())] = a_val[k]
    return dense


@pytest.mark.parametrize("backend", ["torch", "scipy"])
def test_csr_transpose_cpu_matches_dense(backend):
    m, n = 6, 8
    dense, a_val, row_offs, _row_idx, col_idx = make_random_csr(
        m, n, density=0.4, dtype=torch.float32, device="cpu"
    )

    t_val, t_row_offs, t_col_idx = spops.csr_transpose(
        a_val, row_offs, col_idx, m, n, backend=backend
    )

    expected = dense.T
    reconstructed = _csr_to_dense(t_val, t_row_offs, t_col_idx, n, m)
    torch.testing.assert_close(reconstructed, expected, atol=1e-6, rtol=1e-6)


def test_csr_transpose_preserves_dtypes():
    m, n = 4, 4
    _, a_val, row_offs, _, col_idx = make_random_csr(
        m, n, 0.5, torch.float32, "cpu"
    )
    t_val, t_row_offs, t_col_idx = spops.csr_transpose(
        a_val, row_offs, col_idx, m, n, backend="torch"
    )
    assert t_val.dtype == a_val.dtype
    assert t_row_offs.dtype == row_offs.dtype
    assert t_col_idx.dtype == col_idx.dtype


def test_csr_transpose_empty():
    m, n = 3, 5
    a_val = torch.zeros(0, dtype=torch.float32)
    row_offs = torch.zeros(m + 1, dtype=torch.int32)
    col_idx = torch.zeros(0, dtype=torch.int32)
    t_val, t_row_offs, t_col_idx = spops.csr_transpose(
        a_val, row_offs, col_idx, m, n, backend="torch"
    )
    assert t_val.numel() == 0
    assert t_col_idx.numel() == 0
    assert t_row_offs.shape == (n + 1,)
    assert torch.all(t_row_offs == 0)


def test_csr_transpose_double_transpose_is_identity():
    m, n = 5, 7
    dense, a_val, row_offs, _, col_idx = make_random_csr(
        m, n, 0.4, torch.float32, "cpu"
    )
    t_val, t_row_offs, t_col_idx = spops.csr_transpose(
        a_val, row_offs, col_idx, m, n, backend="torch"
    )
    tt_val, tt_row_offs, tt_col_idx = spops.csr_transpose(
        t_val, t_row_offs, t_col_idx, n, m, backend="torch"
    )
    reconstructed = _csr_to_dense(tt_val, tt_row_offs, tt_col_idx, m, n)
    torch.testing.assert_close(reconstructed, dense, atol=1e-6, rtol=1e-6)


def test_csr_transpose_scipy_rejects_cuda():
    require_cuda()
    _, a_val, row_offs, _, col_idx = make_random_csr(
        4, 4, 0.5, torch.float32, "cuda"
    )
    with pytest.raises(AssertionError):
        spops.csr_transpose(a_val, row_offs, col_idx, 4, 4, backend="scipy")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_csr_transpose_cuda_matches_dense(dtype):
    require_cuda()
    m, n = 8, 12
    dense, a_val, row_offs, _, col_idx = make_random_csr(
        m, n, 0.4, dtype, "cuda"
    )
    t_val, t_row_offs, t_col_idx = spops.csr_transpose(
        a_val, row_offs, col_idx, m, n, backend="torch"
    )
    reconstructed = _csr_to_dense(t_val, t_row_offs, t_col_idx, n, m)
    torch.testing.assert_close(
        reconstructed.float().cpu(), dense.float().cpu().T, atol=1e-3, rtol=1e-3
    )
