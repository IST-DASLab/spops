# Sparse Operations (*spops*)

A minimal Pytorch-compatible library supporting basic unstructured sparse operations (spops). Some of the kernels are borrowed from [sputnik](https://github.com/google-research/sputnik).
Additionally, the kernels used in the [Robust Adaptation (RoSA)](https://arxiv.org/abs/2401.04679) paper are included in this repository.

## Installation
Simply make sure you have [pytorch](https://pytorch.org/) installed (preferably install by conda instead of pip to make sure the dependencies are installed correctly), and run 
```
pip install .
```

## Usage
An `m x n` sparse matrix with `nnz` non-zero values in *spops* is stored in CSR format, including the following lists:
- `values`: the list of non-zero values of the matrix with length `nnz`
- `row_offsets`: a list of `m + 1` indices, where the `i`th and `i+1`th elements show the start and end of row `i` in the `values` list, respectively.
- `col_idx`: a list of `nnz` indices, storing the column index of each non-zero value.
- `row_idx`: a permutation of the numbers `0` to `m-1`, sorting the row indices based on the number of non-zeros.

Below you can find a list of supported operations and how to use them.

### Sparse Addition \[dense = sparse + dense\]
Add a sparse CSR matrix `A` to a dense matrix `B` using the `spops.csr_add(A_val, A_row_offsets, A_row_indices, A_col_indices, B)` method. This operation is used in the RoSA paper.

### Sparse Matrix Multiplication (SpMM) \[dense = sparse x dense\]
Multiply a sparse CSR matrix `A` into a dense matrix `B`, resulting in another dense matrix. Simply use the method `spops.spmm(A_val, A_row_offsets, A_row_indices, A_col_indices, B, m)`, where `m` is the number of rows in `A`.

### Sampled Dense-Dense Matrix Multiplication (SDDMM) \[sparse = dense x dense\]
Multiply two dense matrices `A` and `B`, but only calculate the result for a sparse subset of the output elements. This operation is supported in `spops.sddmm(out_row_offsets, out_row_indices, out_col_indices, A, BT)`, where `BT` is the transposed version of `B`, by two different kernels (specify using the `backend` argument):
- The `sputnik` kernel, which works with general sparsity patterns
- The `structure_aware` kernel specifically designed to leverage the sparsity masks that we observe in RoSA, where the non-zero values tend to cluster in a small subset of the rows/columns.

Default is `structure_aware`.

### CSR Transpose \[sparse = sparse.t()\]
Transposes a CSR sparse matrix `A` using the `cuSPARSE` library. Simply use `spops.csr_transpose(A_val, A_row_offsets, A_col_indices, m, n)` to achieve this, where `m` and `n` are the number of rows and columns of `A`, respectively.


## Important Notes
- Make sure that every input to the *spops* methods is [contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html).
- The `row_offsets` list should have `dtype=torch.int32`, while the other index lists should have `dtype=torch.int16`.

## Citation
If you plan to use our work in your projects, please consider citing our paper:

```
@article{nikdan2024rosa,
  title={RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation},
  author={Nikdan, Mahdi and Tabesh, Soroush and Crnčević, Elvir and Alistarh, Dan},
  journal={arXiv preprint arXiv:2401.04679},
  year={2024}
}
```
