"""Unit tests for spops.spmm on CPU and CUDA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
spops = pytest.importorskip("spops")

from tests._helpers import make_random_csr, require_cuda  # noqa: E402


def _reference_spmm(dense_a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return dense_a.to(torch.float32) @ b.to(torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("shape", [(4, 8, 3), (16, 32, 8), (1, 5, 1)])
def test_spmm_cpu_matches_dense(dtype, shape):
    m, k, n = shape
    dense, a_val, row_offs, row_idx, col_idx = make_random_csr(
        m, k, density=0.5, dtype=dtype, device="cpu"
    )
    b = torch.randn(k, n, dtype=dtype)

    out = spops.spmm(a_val, row_offs, row_idx, col_idx, b, m, backend="spmm")
    expected = _reference_spmm(dense, b)
    assert out.shape == expected.shape
    torch.testing.assert_close(out.float(), expected, atol=1e-4, rtol=1e-4)


def test_spmm_cpu_empty_rows():
    """Rows with no nonzeros must yield zeros."""
    m, k, n = 4, 6, 3
    a_val = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    row_offs = torch.tensor([0, 0, 2, 2, 3], dtype=torch.int32)
    row_idx = torch.tensor([1, 1, 3], dtype=torch.int32)
    col_idx = torch.tensor([0, 4, 2], dtype=torch.int32)
    b = torch.arange(k * n, dtype=torch.float32).reshape(k, n)

    out = spops.spmm(a_val, row_offs, row_idx, col_idx, b, m, backend="spmm")
    assert torch.all(out[0] == 0)
    assert torch.all(out[2] == 0)
    expected_row1 = a_val[0] * b[0] + a_val[1] * b[4]
    expected_row3 = a_val[2] * b[2]
    torch.testing.assert_close(out[1], expected_row1)
    torch.testing.assert_close(out[3], expected_row3)


def test_spmm_cpu_all_zero_sparse():
    """All-empty CSR should produce zero output."""
    m, k, n = 3, 4, 2
    a_val = torch.zeros(0, dtype=torch.float32)
    row_offs = torch.zeros(m + 1, dtype=torch.int32)
    row_idx = torch.zeros(0, dtype=torch.int32)
    col_idx = torch.zeros(0, dtype=torch.int32)
    b = torch.randn(k, n)

    out = spops.spmm(a_val, row_offs, row_idx, col_idx, b, m, backend="spmm")
    assert torch.all(out == 0)
    assert out.shape == (m, n)


def test_spmm_cpu_rejects_int_dtypes():
    """Integer value tensors should fail the dtype assertion."""
    m, k, n = 2, 2, 2
    a_val = torch.tensor([1, 2], dtype=torch.int32)
    row_offs = torch.tensor([0, 1, 2], dtype=torch.int32)
    row_idx = torch.tensor([0, 1], dtype=torch.int32)
    col_idx = torch.tensor([0, 1], dtype=torch.int32)
    b = torch.ones(k, n)
    with pytest.raises(AssertionError):
        spops.spmm(a_val, row_offs, row_idx, col_idx, b, m, backend="spmm")


def test_spmm_cpu_return_type_cast():
    m, k, n = 4, 4, 2
    dense, a_val, row_offs, row_idx, col_idx = make_random_csr(
        m, k, 0.5, torch.float32, "cpu"
    )
    b = torch.randn(k, n)
    out = spops.spmm(
        a_val, row_offs, row_idx, col_idx, b, m, backend="spmm", return_type=torch.float16
    )
    assert out.dtype == torch.float16


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_spmm_cuda_matches_dense(dtype):
    require_cuda()
    m, k, n = 32, 64, 16
    dense, a_val, row_offs, row_perm, col_idx = make_random_csr(
        m, k, density=0.3, dtype=dtype, device="cuda"
    )
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    out = spops.spmm(a_val, row_offs, row_perm, col_idx, b, m)
    expected = _reference_spmm(dense, b)
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(out.float(), expected, atol=tol, rtol=tol)


def test_spmm_cuda_fp16_fast_path():
    """force_fp32=False should run the all-fp16 kernel and still match dense."""
    require_cuda()
    m, k, n = 16, 32, 8  # n even, all row offsets even
    dense, a_val, row_offs, row_perm, col_idx = make_random_csr(
        m, k, density=0.5, dtype=torch.float16, device="cuda"
    )
    if (row_offs.remainder(2).sum().item()) != 0:
        pytest.skip("seeded sparsity didn't produce even row offsets")
    b = torch.randn(k, n, dtype=torch.float16, device="cuda")

    out = spops.spmm(a_val, row_offs, row_perm, col_idx, b, m, force_fp32=False)
    expected = _reference_spmm(dense, b)
    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)
