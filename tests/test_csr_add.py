"""Unit tests for spops.csr_add on CPU and CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spops = pytest.importorskip("spops")

from tests._helpers import make_random_csr, require_cuda  # noqa: E402


def _scatter_csr_to_dense(a_val, row_offs, col_idx, m, n, dtype=torch.float32):
    dense = torch.zeros(m, n, dtype=dtype, device=a_val.device)
    for i in range(m):
        for k in range(int(row_offs[i].item()), int(row_offs[i + 1].item())):
            dense[i, int(col_idx[k].item())] = a_val[k].to(dtype)
    return dense


def test_csr_add_cpu_matches_dense():
    m, n = 8, 6
    dense_sparse, a_val, row_offs, row_idx, col_idx = make_random_csr(
        m, n, density=0.4, dtype=torch.float32, device="cpu"
    )
    b = torch.randn(m, n, dtype=torch.float32)

    out = spops.csr_add(a_val, row_offs, row_idx, col_idx, b)
    expected = dense_sparse + b
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_csr_add_cpu_does_not_mutate_b_when_not_inplace():
    m, n = 4, 4
    _, a_val, row_offs, row_idx, col_idx = make_random_csr(m, n, 0.5, torch.float32, "cpu")
    b = torch.randn(m, n)
    b_orig = b.clone()
    spops.csr_add(a_val, row_offs, row_idx, col_idx, b, inplace=False)
    torch.testing.assert_close(b, b_orig)


def test_csr_add_cpu_empty_sparse_is_identity():
    m, n = 3, 5
    a_val = torch.zeros(0, dtype=torch.float32)
    row_offs = torch.zeros(m + 1, dtype=torch.int32)
    row_idx = torch.zeros(0, dtype=torch.int32)
    col_idx = torch.zeros(0, dtype=torch.int32)
    b = torch.randn(m, n)
    out = spops.csr_add(a_val, row_offs, row_idx, col_idx, b)
    torch.testing.assert_close(out, b)


def test_csr_add_cpu_rejects_int_dtypes():
    m, n = 2, 2
    a_val = torch.tensor([1, 2], dtype=torch.int32)
    row_offs = torch.tensor([0, 1, 2], dtype=torch.int32)
    row_idx = torch.tensor([0, 1], dtype=torch.int32)
    col_idx = torch.tensor([0, 1], dtype=torch.int32)
    b = torch.zeros(m, n)
    with pytest.raises(AssertionError):
        spops.csr_add(a_val, row_offs, row_idx, col_idx, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_csr_add_cuda_matches_dense(dtype):
    require_cuda()
    m, n = 16, 24
    dense_sparse, a_val, row_offs, row_perm, col_idx = make_random_csr(
        m, n, density=0.3, dtype=dtype, device="cuda"
    )
    row_perm = row_perm.to(torch.int16)
    col_idx = col_idx.to(torch.int16)
    b = torch.randn(m, n, dtype=dtype, device="cuda")
    b_ref = b.clone()

    out = spops.csr_add(a_val, row_offs, row_perm, col_idx, b)
    expected = dense_sparse + b_ref
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(out.float(), expected.float(), atol=tol, rtol=tol)


def test_csr_add_cuda_inplace_modifies_b():
    require_cuda()
    m, n = 8, 8
    _, a_val, row_offs, row_perm, col_idx = make_random_csr(
        m, n, 0.5, torch.float32, "cuda"
    )
    row_perm = row_perm.to(torch.int16)
    col_idx = col_idx.to(torch.int16)
    b = torch.randn(m, n, device="cuda")
    b_orig = b.clone()

    out = spops.csr_add(a_val, row_offs, row_perm, col_idx, b, inplace=True)
    assert out.data_ptr() == b.data_ptr()
    assert not torch.allclose(b, b_orig)
