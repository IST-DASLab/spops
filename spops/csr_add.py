import torch
import spops_backend
import spops_backend_cpu
from torch import int16, int32, float16, float32, bfloat16

def csr_add(A_val, A_row_offsets, A_row_indices, A_col_indices, B, inplace=False, return_type=None):
    assert A_val.dtype in [float16, float32, bfloat16], 'Only fp32, bf16 and fp16 are supported for sddmm.'

    out = B if inplace else B.clone()

    if A_val.is_cuda:
        assert A_row_offsets.dtype == int32
        for t in [A_row_indices, A_col_indices]:
            assert t.dtype == int16

        if A_val.dtype == float32:
            spops_backend.csr_add_fp32(A_val, A_row_offsets, A_row_indices, A_col_indices, out)
        elif A_val.dtype == float16:
            spops_backend.csr_add_fp16(A_val, A_row_offsets, A_row_indices, A_col_indices, out)
        elif A_val.dtype == bfloat16:
            spops_backend.csr_add_bf16(A_val, A_row_offsets, A_row_indices, A_col_indices, out)
    else:
        A_val = A_val.to(float32)
        A_row_offsets = A_row_offsets.to(int32)
        A_col_indices = A_col_indices.to(int32)
        out = out.to(float32)
        spops_backend_cpu.csr_add(A_val, A_row_offsets, A_col_indices, out)

    return out if return_type is None else out.to(return_type)