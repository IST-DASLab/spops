#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "./structure_aware/sputnik/cuda_utils.h"
#include <vector>


std::vector<torch::Tensor> structure_aware_sddmm_cuda_fp32_benchmark( // returns masked(mm(A, BT.T), M)
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	int last0,
	int last1,
	torch::Tensor A,
	torch::Tensor BT,
	bool csr,
	torch::Tensor t);


std::vector<torch::Tensor> structure_aware_sddmm_cuda_fp32( // returns masked(mm(A, BT.T), M)
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	int last0,
	int last1,
	torch::Tensor A,
	torch::Tensor BT,
	bool csr);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> structure_aware_sddmm_fp32_benchmark(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	int last0,
	int last1,
	torch::Tensor A,
	torch::Tensor BT,
	bool csr,
	torch::Tensor t) {
  // CHECK_INPUT(C_val);
  CHECK_INPUT(M_row_offsets);
  CHECK_INPUT(M_row_indices);
  CHECK_INPUT(M_col_indices);
  CHECK_INPUT(A);
  CHECK_INPUT(BT);

  // const at::cuda::OptionalCUDAGuard device_guard1(device_of(C_val));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(M_row_offsets));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(M_row_indices));
  const at::cuda::OptionalCUDAGuard device_guard4(device_of(M_col_indices));
  const at::cuda::OptionalCUDAGuard device_guard5(device_of(A));
  const at::cuda::OptionalCUDAGuard device_guard6(device_of(BT));
  // return sputnik_sddmm_cuda_fp32(C_val, M_row_offsets, M_row_indices, M_col_indices, A, BT);
  return structure_aware_sddmm_cuda_fp32_benchmark(M_row_offsets, M_row_indices, M_col_indices, last0, last1, A, BT, csr, t);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> structure_aware_sddmm_fp32(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	int last0,
	int last1,
	torch::Tensor A,
	torch::Tensor BT
) {
// CHECK_INPUT(C_val);
  CHECK_INPUT(M_row_offsets);
  CHECK_INPUT(M_row_indices);
  CHECK_INPUT(M_col_indices);
  CHECK_INPUT(A);
  CHECK_INPUT(BT);

// const at::cuda::OptionalCUDAGuard device_guard1(device_of(C_val));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(M_row_offsets));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(M_row_indices));
  const at::cuda::OptionalCUDAGuard device_guard4(device_of(M_col_indices));
  const at::cuda::OptionalCUDAGuard device_guard5(device_of(A));
  const at::cuda::OptionalCUDAGuard device_guard6(device_of(BT));
  return structure_aware_sddmm_cuda_fp32(M_row_offsets, M_row_indices, M_col_indices, last0, last1, A, BT, true);
}
