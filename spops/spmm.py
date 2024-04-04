import torch
import spops_backend
import spops_backend_cpu
from torch import int16, int32, float16, float32, bfloat16

def spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, M, backend=None, force_fp32=True, return_type=None):
    assert all([t.dtype in [float16, bfloat16, float32] for t in [A_val, B]]), 'Only fp32, bf16 and fp16 are supported for spmm.'
    
    if A_val.is_cuda:
        if backend is None:
            backend = 'sputnik'
        assert backend in ['sputnik']
        run_full_fp16 = not force_fp32 and A_val.dtype == float16 and B.dtype == float16 and B.shape[1] % 2 == 0 and A_row_offsets.remainder(2).sum().item() == 0
        if run_full_fp16:
            out = spops_backend.sputnik_spmm_fp16(A_val, A_row_offsets.to(int32), A_row_indices.to(int32), A_col_indices.to(int16), B, M)[0]
        else:
            out = spops_backend.sputnik_spmm_fp32(A_val.to(float32), A_row_offsets.to(int32), A_row_indices.to(int32), A_col_indices.to(int32), B.to(float32), M)[0]
    else:
        if backend is None:
            backend = 'spmm'
        assert backend in ['spmm']
        A_val = A_val.to(float32)
        B = B.to(float32)
        A_row_offsets = A_row_offsets.to(int32)
        A_col_indices = A_col_indices.to(int32)
        out = A_val.new_zeros(M, B.shape[1])
        getattr(spops_backend_cpu, backend)(A_val, A_row_offsets, A_col_indices, B, out)
        
    return out if return_type is None else out.to(return_type)