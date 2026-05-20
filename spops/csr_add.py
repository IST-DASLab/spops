import torch
from torch import bfloat16, float16, float32, int16, int32

from spops._backend import get_cpu_backend, get_cuda_backend


def csr_add(A_val, A_row_offsets, A_row_indices, A_col_indices, B, inplace=False, return_type=None):
    assert A_val.dtype in [float16, float32, bfloat16], 'Only fp32, bf16 and fp16 are supported for sddmm.'

    out = B if inplace else B.clone()

    if A_val.is_cuda:
        cuda_backend = get_cuda_backend()
        assert A_row_offsets.dtype == int32
        for t in [A_row_indices, A_col_indices]:
            assert t.dtype == int16

        if A_val.dtype == float32:
            cuda_backend.csr_add_fp32(A_val, A_row_offsets, A_row_indices, A_col_indices, out)
        elif A_val.dtype == float16:
            cuda_backend.csr_add_fp16(A_val, A_row_offsets, A_row_indices, A_col_indices, out)
        elif A_val.dtype == bfloat16:
            cuda_backend.csr_add_bf16(A_val, A_row_offsets, A_row_indices, A_col_indices, out)
    else:
        cpu_backend = get_cpu_backend()
        A_val = A_val.to(float32)
        A_row_offsets = A_row_offsets.to(int32)
        A_col_indices = A_col_indices.to(int32)
        out = out.to(float32)
        cpu_backend.csr_add(A_val, A_row_offsets, A_col_indices, out)

    return out if return_type is None else out.to(return_type)
