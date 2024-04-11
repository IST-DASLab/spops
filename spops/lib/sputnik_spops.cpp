#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "./sputnik/cuda_utils.h"
#include <vector>

std::vector<torch::Tensor> sputnik_spmm_cuda_fp32( // returns mm(A, B)
    // torch::Tensor C, // the output
    torch::Tensor A_val,
    torch::Tensor A_row_offsets,
    torch::Tensor A_row_indices,
    torch::Tensor A_col_indices,
    torch::Tensor B,
    int out_dim);
  
std::vector<torch::Tensor> sputnik_spmm_cuda_fp16( // returns mm(A, B)
    // torch::Tensor C, // the output
    torch::Tensor A_val,
    torch::Tensor A_row_offsets,
    torch::Tensor A_row_indices,
    torch::Tensor A_col_indices,
    torch::Tensor B,
    int out_dim);


std::vector<torch::Tensor> sputnik_sddmm_cuda_fp32_benchmark( // returns masked(mm(A, BT.T), M)
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	torch::Tensor A,
	torch::Tensor BT,
	torch::Tensor t
);

std::vector<torch::Tensor> sputnik_sddmm_cuda_fp32( // returns masked(mm(A, BT.T), M)
    // torch::Tensor C_val, // the output values
    torch::Tensor M_row_offsets,
    torch::Tensor M_row_indices,
    torch::Tensor M_col_indices,
    torch::Tensor A,
    torch::Tensor BT
	  // torch::Tensor t
  );

// std::vector<torch::Tensor> csr_transpose_cuda_fp32( // returns A.T in csr format
//     // torch::Tensor AT_val, // output values
//     // torch::Tensor AT_row_offsets, // output row offsets
//     // torch::Tensor AT_col_indices, // output column indices
//     torch::Tensor A_val,
//     torch::Tensor A_row_offsets,
//     torch::Tensor A_col_indices,
//     int M,
//     int N);

// std::vector<torch::Tensor> csr_transpose_cuda_fp16( // returns A.T in csr format
//     // torch::Tensor AT_val, // output values
//     // torch::Tensor AT_row_offsets, // output row offsets
//     // torch::Tensor AT_col_indices, // output column indices
//     torch::Tensor A_val,
//     torch::Tensor A_row_offsets,
//     torch::Tensor A_col_indices,
//     int M,
//     int N);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> sputnik_spmm_fp32(
    // torch::Tensor C, // the output
    torch::Tensor A_val,
    torch::Tensor A_row_offsets,
    torch::Tensor A_row_indices,
    torch::Tensor A_col_indices,
    torch::Tensor B,
    int out_dim) {
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
  return sputnik_spmm_cuda_fp32(A_val, A_row_offsets, A_row_indices, A_col_indices, B, out_dim);
}

std::vector<torch::Tensor> sputnik_spmm_fp16(
    // torch::Tensor C, // the output
    torch::Tensor A_val,
    torch::Tensor A_row_offsets,
    torch::Tensor A_row_indices,
    torch::Tensor A_col_indices,
    torch::Tensor B,
    int out_dim) {
  // CHECK_INPUT(C);
  CHECK_INPUT(B);
  CHECK_INPUT(A_val);
  CHECK_INPUT(A_row_offsets);
  CHECK_INPUT(A_row_indices);
  CHECK_INPUT(A_col_indices);

  // const at::cuda::OptionalCUDAGuard device_guard1(device_of(C));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(B));
  const at::cuda::OptionalCUDAGuard device_guard3(device_of(A_val));
  const at::cuda::OptionalCUDAGuard device_guard4(device_of(A_row_offsets));
  const at::cuda::OptionalCUDAGuard device_guard5(device_of(A_row_indices));
  const at::cuda::OptionalCUDAGuard device_guard6(device_of(A_col_indices));
  // return sputnik_spmm_cuda_fp16(C, A_val, A_row_offsets, A_row_indices, A_col_indices, B, out_dim);
  return sputnik_spmm_cuda_fp16(A_val, A_row_offsets, A_row_indices, A_col_indices, B, out_dim);
}

std::vector<torch::Tensor> sputnik_sddmm_fp32(
    // torch::Tensor C_val, // the output values
    torch::Tensor M_row_offsets,
    torch::Tensor M_row_indices,
    torch::Tensor M_col_indices,
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
  return sputnik_sddmm_cuda_fp32(M_row_offsets, M_row_indices, M_col_indices, A, BT);
}


std::vector<torch::Tensor> sputnik_sddmm_fp32_benchmark(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	torch::Tensor A,
	torch::Tensor BT,
	torch::Tensor t
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
  return sputnik_sddmm_cuda_fp32_benchmark(M_row_offsets, M_row_indices, M_col_indices, A, BT, t);
}

// std::vector<torch::Tensor> csr_transpose_fp32(
//     // torch::Tensor AT_val, // output values
//     // torch::Tensor AT_row_offsets, // output row offsets
//     // torch::Tensor AT_col_indices, // output column indices
//     torch::Tensor A_val,
//     torch::Tensor A_row_offsets,
//     torch::Tensor A_col_indices,
//     int M,
//     int N) {
//   // CHECK_INPUT(AT_val);
//   // CHECK_INPUT(AT_row_offsets);
//   // CHECK_INPUT(AT_col_indices);
//   CHECK_INPUT(A_val);
//   CHECK_INPUT(A_row_offsets);
//   CHECK_INPUT(A_col_indices);

//   // const at::cuda::OptionalCUDAGuard device_guard1(device_of(AT_val));
//   // const at::cuda::OptionalCUDAGuard device_guard2(device_of(AT_row_offsets));
//   // const at::cuda::OptionalCUDAGuard device_guard3(device_of(AT_col_indices));
//   const at::cuda::OptionalCUDAGuard device_guard4(device_of(A_val));
//   const at::cuda::OptionalCUDAGuard device_guard5(device_of(A_row_offsets));
//   const at::cuda::OptionalCUDAGuard device_guard6(device_of(A_col_indices));
//   // return csr_transpose_cuda_fp32(AT_val, AT_row_offsets, AT_col_indices, A_val, A_row_offsets, A_col_indices, M, N);
//   return csr_transpose_cuda_fp32(A_val, A_row_offsets, A_col_indices, M, N);
// }

// std::vector<torch::Tensor> csr_transpose_fp16(
//     // torch::Tensor AT_val, // output values
//     // torch::Tensor AT_row_offsets, // output row offsets
//     // torch::Tensor AT_col_indices, // output column indices
//     torch::Tensor A_val,
//     torch::Tensor A_row_offsets,
//     torch::Tensor A_col_indices,
//     int M,
//     int N) {
//   // CHECK_INPUT(AT_val);
//   // CHECK_INPUT(AT_row_offsets);
//   // CHECK_INPUT(AT_col_indices);
//   CHECK_INPUT(A_val);
//   CHECK_INPUT(A_row_offsets);
//   CHECK_INPUT(A_col_indices);

//   // const at::cuda::OptionalCUDAGuard device_guard1(device_of(AT_val));
//   // const at::cuda::OptionalCUDAGuard device_guard2(device_of(AT_row_offsets));
//   // const at::cuda::OptionalCUDAGuard device_guard3(device_of(AT_col_indices));
//   const at::cuda::OptionalCUDAGuard device_guard4(device_of(A_val));
//   const at::cuda::OptionalCUDAGuard device_guard5(device_of(A_row_offsets));
//   const at::cuda::OptionalCUDAGuard device_guard6(device_of(A_col_indices));
//   // return csr_transpose_cuda_fp16(AT_val, AT_row_offsets, AT_col_indices, A_val, A_row_offsets, A_col_indices, M, N);
//   return csr_transpose_cuda_fp16(A_val, A_row_offsets, A_col_indices, M, N);
// }