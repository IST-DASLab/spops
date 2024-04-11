import torch
import spops_backend
from torch import int32, float16, float32, bfloat16
from scipy.sparse import csr_matrix

def csr_transpose(A_val, A_row_offsets, A_col_indices, M, N, backend=None):
    assert A_val.dtype in [float16, float32, bfloat16], 'Only fp32, bf16 and fp16 are supported for sddmm.'
    if backend is None:
        backend = 'torch'
    
    if backend == 'torch':
        dense = torch.sparse_csr_tensor(
            A_row_offsets,
            A_col_indices,
            A_val,
            size=(M, N),
            dtype=A_val.dtype,
            device=A_val.device
        ).to_dense()
        tsp = dense.T.to_sparse_csr()
        out = (
            tsp.values().to(A_val.dtype),
            tsp.crow_indices().to(A_row_offsets.dtype),
            tsp.col_indices().to(A_col_indices.dtype)
        )
    # elif backend == 'cusparse':
    #     assert A_val.is_cuda
    #     assert backend in ['cusparse']
    #     if A_val.dtype == float16:
    #         out = spops_backend.csr_transpose_fp16(A_val, A_row_offsets.to(int32), A_col_indices.to(int32), M, N)
    #     else:
    #         out = spops_backend.csr_transpose_fp32(A_val.to(float32), A_row_offsets.to(int32), A_col_indices.to(int32), M, N)
    #     out[0] = out[0].to(A_val.dtype)
    else:
        assert backend == 'scipy'
        assert not A_val.is_cuda
        dense = csr_matrix((A_val, A_col_indices, A_row_offsets), shape=(M, N)).toarray()
        tsp = csr_matrix(dense.T)
        out = (
            torch.tensor(tsp.data, dtype=A_val.dtype),
            torch.tensor(tsp.indptr, dtype=A_row_offsets.dtype),
            torch.tensor(tsp.indices, dtype=A_col_indices.dtype)
        )
        
        
    return out