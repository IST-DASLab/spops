import torch
import spops_backend
from torch import int16, int32, float16, float32, bfloat16

def spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, M, backend='sputnik', force_fp32=True, return_type=None):
    assert backend in ['sputnik']
    assert all([t.dtype in [float16, bfloat16, float32] for t in [A_val, B]]), 'Only fp32, bf16 and fp16 are supported for spmm.'
    
    run_full_fp16 = not force_fp32 and A_val.dtype == float16 and B.dtype == float16 and B.shape[1] % 2 == 0 and A_row_offsets.remainder(2).sum().item() == 0

    if run_full_fp16:
        # out = torch.zeros((M, B.shape[-1]), dtype=float16, device=A_val.device)
        # out = spops_backend.sputnik_spmm_fp16(out, A_val, A_row_offsets.to(int32), A_row_indices.to(int32), A_col_indices.to(int16), B, M)[0]
        out = spops_backend.sputnik_spmm_fp16(A_val, A_row_offsets.to(int32), A_row_indices.to(int32), A_col_indices.to(int16), B, M)[0]
    else:
        # out = torch.zeros((M, B.shape[-1]), dtype=float32, device=A_val.device)
        # out = spops_backend.sputnik_spmm_fp32(out, A_val.to(float32), A_row_offsets.to(int32), A_row_indices.to(int32), A_col_indices.to(int32), B.to(float32), M)[0]
        out = spops_backend.sputnik_spmm_fp32(A_val.to(float32), A_row_offsets.to(int32), A_row_indices.to(int32), A_col_indices.to(int32), B.to(float32), M)[0]
    
    if return_type is None:
        return out
    else:
        return out.to(return_type)