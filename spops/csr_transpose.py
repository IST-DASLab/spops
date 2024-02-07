import torch
import spops_backend
from torch import int32, float16, float32, bfloat16

def csr_transpose(A_val, A_row_offsets, A_col_indices, M, N, backend='cusparse'):
    assert backend in ['cusparse']
    assert A_val.dtype in [float16, float32, bfloat16], 'Only fp32, bf16 and fp16 are supported for sddmm.'
    
    # AT_val = torch.zeros_like(A_val)
    # AT_row_offsets = torch.zeros((N + 1, ), dtype=int32, device=A_row_offsets.device)
    # AT_col_indices = torch.zeros_like(A_col_indices)

    if A_val.dtype == float16:
        # spops_backend.csr_transpose_fp16(AT_val, AT_row_offsets, AT_col_indices, A_val, A_row_offsets.to(int32), A_col_indices.to(int32), M, N)
        out = spops_backend.csr_transpose_fp16(A_val, A_row_offsets.to(int32), A_col_indices.to(int32), M, N)
    else:
        # AT_val = AT_val.to(float32)
        # AT_col_indices = AT_row_offsets.to(int32)
        # spops_backend.csr_transpose_fp32(AT_val, AT_row_offsets, AT_col_indices, A_val.to(float32), A_row_offsets.to(int32), A_col_indices.to(int32), M, N)
        out = spops_backend.csr_transpose_fp32(A_val.to(float32), A_row_offsets.to(int32), A_col_indices.to(int32), M, N)
    
    # AT_val = AT_val.to(A_val.dtype)
    # return (AT_val, AT_row_offsets, AT_col_indices)
    out[0] = out[0].to(A_val.dtype)
    return out