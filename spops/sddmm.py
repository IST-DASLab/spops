import torch
from torch import bfloat16, float16, float32, int32

from spops._backend import get_cpu_backend, get_cuda_backend


def sddmm(mask_row_offsets, mask_row_indices, mask_col_indices, A, BT, backend=None, return_type=None):
    assert all([t.dtype in [float16, float32, bfloat16] for t in [A, BT]]), 'Only fp32, bf16 and fp16 are supported for sddmm.'

    if A.is_cuda:
        cuda_backend = get_cuda_backend()
        if backend is None:
            backend = 'structure_aware'
        assert backend in ['sputnik', 'structure_aware']
        if backend == 'structure_aware':
            sorted_row_counts = torch.diff(mask_row_offsets)[mask_row_indices.int()]
            last0 = (sorted_row_counts != 0).int().sum().cpu().item()
            last1 = sorted_row_counts[0].int()
            out = cuda_backend.structure_aware_sddmm_fp32(mask_row_offsets.to(int32), mask_row_indices, mask_col_indices, last0, last1, A.to(float32), BT.to(float32))[0]
        else:
            out = cuda_backend.sputnik_sddmm_fp32(mask_row_offsets.to(int32), mask_row_indices.to(int32), mask_col_indices.to(int32), A.to(float32), BT.to(float32))[0]
    else:
        cpu_backend = get_cpu_backend()
        if backend is None:
            backend = 'sddmm_v3'
        assert backend in ['sddmm', 'sddmm_v2', 'sddmm_v3']
        A = A.to(float32)
        BT = BT.to(float32)
        mask_row_offsets = mask_row_offsets.to(int32)
        mask_col_indices = mask_col_indices.to(int32)
        out = A.new_zeros(mask_col_indices.shape)
        getattr(cpu_backend, backend)(A, BT, mask_row_offsets, mask_col_indices, out)

    return out if return_type is None else out.to(return_type)
