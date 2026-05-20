# Sparse Operations (*spops*)

A minimal Pytorch-compatible library supporting basic unstructured sparse operations (spops). Some of the kernels are borrowed from [sputnik](https://github.com/google-research/sputnik).
Additionally, the kernels used in the [Robust Adaptation (RoSA)](https://arxiv.org/abs/2401.04679) paper ([GitHub](https://github.com/IST-DASLab/RoSA)) are included in this repository.

## Installation

`spops` builds a CUDA / CPU extension against your installed PyTorch, so PyTorch
(and a matching CUDA toolkit) must be available **before** the build. The
package uses a standard PEP 517 build (`setup.py` + `pyproject.toml`).

### Requirements

- Python 3.9+
- PyTorch 2.0+ (must be importable at build time)
- CUDA toolkit matching your PyTorch build (for the GPU extension)
- `ninja`, `pybind11`, `numpy`, `setuptools`, `wheel` available in the build env

### Recommended: install with `--no-build-isolation`

Because PEP 517 isolated builds create a fresh env without your CUDA-enabled
PyTorch, you almost always want to build against the env you have:

```bash
pip install ninja pybind11 numpy scipy
pip install --no-build-isolation .
```

For local development:

```bash
pip install --no-build-isolation -e .
```

### Selecting target GPU architectures

The build emits PTX/cubin for `sm_80;sm_89;sm_90+PTX` by default — i.e. A100,
L40/L40S, and H100, with PTX embedded for forward compatibility. Override via
`TORCH_CUDA_ARCH_LIST` for faster, GPU-specific builds:

```bash
TORCH_CUDA_ARCH_LIST="9.0"       pip install --no-build-isolation .  # H100
TORCH_CUDA_ARCH_LIST="8.9"       pip install --no-build-isolation .  # L40
TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" pip install --no-build-isolation .  # A100 + JIT
```

### Using with [`uv`](https://docs.astral.sh/uv/)

Add `spops` as a path or git source and tell uv to skip build isolation so the
build can see the project's `torch`:

```toml
[tool.uv.sources]
spops = { path = "path/to/spops" }       # built wheel install
# spops = { path = "path/to/spops", editable = true }  # editable install

[tool.uv]
no-build-isolation-package = ["spops"]
```

Make sure `torch`, `ninja`, `pybind11`, `numpy`, `setuptools`, and `wheel` are
in your project's dependencies so they are present when uv builds spops.

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
Transposes a CSR sparse matrix `A`. Use `spops.csr_transpose(A_val, A_row_offsets, A_col_indices, m, n)`, where `m` and `n` are the number of rows and columns of `A`. Two backends are available via the `backend` argument:
- `torch` (default): uses PyTorch's built-in sparse CSR support; works for both CPU and CUDA tensors.
- `scipy`: uses `scipy.sparse` on the CPU.


## Important Notes
- Make sure that every input to the *spops* methods is [contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html).
- `row_offsets` should always be `torch.int32`.
- For the CUDA `csr_add` and the fp16 fast path of `spmm`, the other index lists
  (`row_idx`, `col_idx`) must be `torch.int16`. The fp32 CUDA paths and the
  `sputnik` SDDMM backend accept `int32` and cast internally; the
  `structure_aware` SDDMM backend takes `int16` directly.
- `row_idx` is **not** a per-nnz row label — it is a length-`m` permutation of
  row indices sorted by descending non-zero count, used by the underlying
  sputnik kernels for warp-level load balancing. The canonical construction is
  `torch.argsort(-torch.diff(row_offsets)).int()`.

## Testing

A pytest suite covering all four operations on both CPU and CUDA lives in
`tests/`. CUDA tests are skipped automatically when no GPU is available.

```bash
pip install pytest
pytest -q tests                     # full suite
pytest -q tests -k "not cuda"       # CPU only
```

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
