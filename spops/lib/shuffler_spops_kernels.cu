#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "./shuffler/include/spadd.cuh"
#include <iostream>
#include <cstdio>
#include <cuda_bf16.h>

template<class T, class INDEX_TYPE = short, class ACCUMULATOR_TYPE = int>
std::vector <torch::Tensor> csr_cuda_add(
	// torch::Tensor C, // the output
	torch::Tensor A_val, // (M, N)
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B // (M, N)
) {
  int M = B.size(0);
  int N = B.size(1);
  int nnz = A_val.size(0);

  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
	e::spadd_csr_v1<T, T, INDEX_TYPE, ACCUMULATOR_TYPE><<<M, 32>>>(
		M,
			N,
			A_row_offsets.data<ACCUMULATOR_TYPE>(),
			A_row_indices.data<INDEX_TYPE>(),
			A_col_indices.data<INDEX_TYPE>(),
			reinterpret_cast<__nv_bfloat16*>(A_val.data<at::BFloat16>()),
			reinterpret_cast<__nv_bfloat16*>(B.data<at::BFloat16>())
	);
  } else if constexpr (std::is_same<T, at::Half>::value) {
	e::spadd_csr_v1<half, half, INDEX_TYPE, ACCUMULATOR_TYPE><<<M, 32>>>(
		M,
			N,
			A_row_offsets.data<ACCUMULATOR_TYPE>(),
			A_row_indices.data<INDEX_TYPE>(),
			A_col_indices.data<INDEX_TYPE>(),
			reinterpret_cast<half*>(A_val.data<at::Half>()),
			reinterpret_cast<half*>(B.data<at::Half>())
	);
  } else {
	e::spadd_csr_v1<T, T, INDEX_TYPE, ACCUMULATOR_TYPE><<<M, 32>>>(
		M,
			N,
			A_row_offsets.data<ACCUMULATOR_TYPE>(),
			A_row_indices.data<INDEX_TYPE>(),
			A_col_indices.data<INDEX_TYPE>(),
			A_val.data<T>(),
			B.data<T>()
	);
  }

  return {};
}

std::vector <torch::Tensor> csr_cuda_add_fp32(torch::Tensor A_val,
											  torch::Tensor A_row_offsets,
											  torch::Tensor A_row_indices,
											  torch::Tensor A_col_indices,
											  torch::Tensor B) {
  return csr_cuda_add<float>(A_val, A_row_offsets, A_row_indices, A_col_indices, B);
}

std::vector <torch::Tensor> csr_cuda_add_fp16(torch::Tensor A_val,
											  torch::Tensor A_row_offsets,
											  torch::Tensor A_row_indices,
											  torch::Tensor A_col_indices,
											  torch::Tensor B) {
  return csr_cuda_add<at::Half>(A_val, A_row_offsets, A_row_indices, A_col_indices, B);
}

std::vector <torch::Tensor> csr_cuda_add_bf16(torch::Tensor A_val,
											  torch::Tensor A_row_offsets,
											  torch::Tensor A_row_indices,
											  torch::Tensor A_col_indices,
											  torch::Tensor B) {
  return csr_cuda_add<__nv_bfloat16>(A_val, A_row_offsets, A_row_indices, A_col_indices, B);
}
