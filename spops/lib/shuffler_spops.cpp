#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "./sputnik/cuda_utils.h"
#include <vector>

std::vector<torch::Tensor> csr_cuda_add_fp32(
	// torch::Tensor C, // the output
	torch::Tensor A_val, // (M, N)
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B // (M, N)
);

std::vector<torch::Tensor> csr_cuda_add_fp16(
	// torch::Tensor C, // the output
	torch::Tensor A_val, // (M, N)
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B // (M, N)
);

std::vector<torch::Tensor> csr_cuda_add_bf16(
	// torch::Tensor C, // the output
	torch::Tensor A_val, // (M, N)
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B // (M, N)
);


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> csr_add_fp32(
	// torch::Tensor C, // the output
	torch::Tensor A_val,
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B) {
  // CHECK_INPUT(C);
  CHECK_INPUT(B);
  CHECK_INPUT(A_val);
  CHECK_INPUT(A_row_offsets);
  CHECK_INPUT(A_row_indices);
  CHECK_INPUT(A_col_indices);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(C));
  const at::cuda::OptionalCUDAGuard device_guard1(device_of(B));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(A_val));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(A_row_offsets));
  const at::cuda::OptionalCUDAGuard device_guard4(device_of(A_row_indices));
  const at::cuda::OptionalCUDAGuard device_guard5(device_of(A_col_indices));
  // return sputnik_spmm_cuda_fp32(C, A_val, A_row_offsets, A_row_indices, A_col_indices, B, out_dim);
  return csr_cuda_add_fp32(A_val, A_row_offsets, A_row_indices, A_col_indices, B);
}

std::vector<torch::Tensor> csr_add_fp16(
	// torch::Tensor C, // the output
	torch::Tensor A_val,
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B) {
  // CHECK_INPUT(C);
  CHECK_INPUT(B);
  CHECK_INPUT(A_val);
  CHECK_INPUT(A_row_offsets);
  CHECK_INPUT(A_row_indices);
  CHECK_INPUT(A_col_indices);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(C));
  const at::cuda::OptionalCUDAGuard device_guard1(device_of(B));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(A_val));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(A_row_offsets));
  const at::cuda::OptionalCUDAGuard device_guard4(device_of(A_row_indices));
  const at::cuda::OptionalCUDAGuard device_guard5(device_of(A_col_indices));
  return csr_cuda_add_fp16(A_val, A_row_offsets, A_row_indices, A_col_indices, B);
}

std::vector<torch::Tensor> csr_add_bf16(
	// torch::Tensor C, // the output
	torch::Tensor A_val,
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B) {
  // CHECK_INPUT(C);
  CHECK_INPUT(B);
  CHECK_INPUT(A_val);
  CHECK_INPUT(A_row_offsets);
  CHECK_INPUT(A_row_indices);
  CHECK_INPUT(A_col_indices);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(C));
  const at::cuda::OptionalCUDAGuard device_guard1(device_of(B));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(A_val));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(A_row_offsets));
  const at::cuda::OptionalCUDAGuard device_guard4(device_of(A_row_indices));
  const at::cuda::OptionalCUDAGuard device_guard5(device_of(A_col_indices));
  return csr_cuda_add_bf16(A_val, A_row_offsets, A_row_indices, A_col_indices, B);
}
