#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "./sputnik/spmm/cuda_spmm.cu.cc"
#include "./sputnik/sddmm/cuda_sddmm.cu.cc"
#include "shuffler/include/spadd.cuh"
// #include "cusparse.h"
#include <iostream>
#include <cstdio>
// #include "cuda_fp16.h"

inline void start_clock(cudaEvent_t &start) {
  AT_CUDA_CHECK(cudaEventRecord(start, 0));
}

inline float end_clock(cudaEvent_t &start, cudaEvent_t &end) {
  float time;
  AT_CUDA_CHECK(cudaEventRecord(end, 0));
  AT_CUDA_CHECK(cudaEventSynchronize(end));
  AT_CUDA_CHECK(cudaEventElapsedTime(&time, start, end));

  // Returns ms
  return time;
}

struct Timer {
  enum class Type { GPU, CPU };
  using Time = std::chrono::time_point<std::chrono::steady_clock>;

  cudaEvent_t ce_start{}, ce_stop{};
  Time cpu_ce_start{}, cpu_ce_stop{};
  std::vector<float> measurements_;
  std::string name;
  Type type;

  Time now() { return std::chrono::steady_clock::now(); }

  void start() {
	start_clock(ce_start);
  }

  float end() {
	auto timing = end_clock(ce_start, ce_stop);
	return timing;
  }

  Timer(int runs, std::string name) : name(std::move(name)) {
	AT_CUDA_CHECK(cudaEventCreate(&ce_start));
	AT_CUDA_CHECK(cudaEventCreate(&ce_stop));
  }

  Timer(Timer &&timer) = delete;
  Timer(const Timer &timer) = delete;

  ~Timer() {
	if (type == Type::GPU) {
	  AT_CUDA_CHECK(cudaEventDestroy(ce_start));
	  AT_CUDA_CHECK(cudaEventDestroy(ce_stop));
	}
  }
};

std::vector <torch::Tensor> sputnik_spmm_cuda_fp32(
	// torch::Tensor C, // the output
	torch::Tensor A_val, // (M, K)
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B, // (K, N)
	int M) {

  const auto K = B.size(0);
  const auto N = B.size(1);
  const auto nnz = A_val.size(0);

  auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(B.device()));

  sputnik::CudaSpmm(
	  /*m=*/ M,
	  /*k=*/ K,
	  /*n=*/ N,
	  /*nonzeros=*/ nnz,
	  /*row_indices=*/ A_row_indices.data<int>(),
	  /*values=*/ A_val.data<float>(),
	  /*row_offsets=*/ A_row_offsets.data<int>(),
	  /*column_indices=*/ A_col_indices.data<int>(),
	  /*dense_matrix=*/ B.data<float>(),
	  /*output_matrix=*/ C.data<float>(),
	  /*stream=*/ at::cuda::getCurrentCUDAStream().stream()
  );

  return {C};
}

std::vector <torch::Tensor> sputnik_spmm_cuda_fp16(
	// torch::Tensor C, // the output
	torch::Tensor A_val, // (M, K)
	torch::Tensor A_row_offsets,
	torch::Tensor A_row_indices,
	torch::Tensor A_col_indices,
	torch::Tensor B, // (K, N)
	int M) {

  const int K = B.size(0);
  const int N = B.size(1);
  const int nnz = A_val.size(0);

  auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).layout(torch::kStrided).device(B.device()));

  sputnik::CudaSpmm(
	  /*m=*/ M,
	  /*k=*/ K,
	  /*n=*/ N,
	  /*nonzeros=*/ nnz,
	  /*row_indices=*/ A_row_indices.data<int>(),
	  /*values=*/ (half2 *)A_val.data<at::Half>(),
	  /*row_offsets=*/ A_row_offsets.data<int>(),
	  /*column_indices=*/ (short2 *)A_col_indices.data<short>(),
	  /*dense_matrix=*/ (half2 *)B.data<at::Half>(),
	  /*output_matrix=*/ (half2 *)C.data<at::Half>(),
	  /*stream=*/ at::cuda::getCurrentCUDAStream().stream()
  );

  return {C};
}


std::vector <torch::Tensor> sputnik_sddmm_cuda_fp32_benchmark(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	torch::Tensor A, // (M, K)
	torch::Tensor BT, // (N, K)
	torch::Tensor t
) {
   Timer timer(1, "sddmm");

  const auto M = A.size(0);
  const auto K = A.size(1);
  const auto N = BT.size(0);
  const auto nnz = M_col_indices.size(0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto C_val = torch::zeros(nnz, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(A.device()));

   timer.start();
  sputnik::CudaSddmm(
	  /*m=*/ M,
	  /*k=*/ K,
	  /*n=*/ N,
	  /*nonzeros=*/ nnz,
	  /*row_indices=*/ M_row_indices.data<int>(),
	  /*row_offsets=*/ M_row_offsets.data<int>(),
	  /*column_indices=*/ M_col_indices.data<int>(),
	  /*lhs_matrix=*/ A.data<float>(),
	  /*rhs_matrix=*/ BT.data<float>(),
	  /*output_values=*/ C_val.data<float>(),
	  /*stream=*/ stream
  );

   AT_CUDA_CHECK(cudaStreamSynchronize(stream));
   t.data<float>()[0] = timer.end();

  return {C_val};
}

std::vector <torch::Tensor> sputnik_sddmm_cuda_fp32(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	torch::Tensor A, // (M, K)
	torch::Tensor BT // (N, K)
	// torch::Tensor t
) {
//   Timer timer(1, "sddmm");

  const auto M = A.size(0);
  const auto K = A.size(1);
  const auto N = BT.size(0);
  const auto nnz = M_col_indices.size(0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto C_val = torch::zeros(nnz, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(A.device()));

//   timer.start();
  sputnik::CudaSddmm(
	  /*m=*/ M,
	  /*k=*/ K,
	  /*n=*/ N,
	  /*nonzeros=*/ nnz,
	  /*row_indices=*/ M_row_indices.data<int>(),
	  /*row_offsets=*/ M_row_offsets.data<int>(),
	  /*column_indices=*/ M_col_indices.data<int>(),
	  /*lhs_matrix=*/ A.data<float>(),
	  /*rhs_matrix=*/ BT.data<float>(),
	  /*output_values=*/ C_val.data<float>(),
	  /*stream=*/ stream
  );

//   AT_CUDA_CHECK(cudaStreamSynchronize(stream));
//   t.data<float>()[0] = timer.end();

  return {C_val};
}

// std::vector <torch::Tensor> csr_transpose_cuda_fp32(
// 	// torch::Tensor AT_val, // output values
// 	// torch::Tensor AT_row_offsets, // output row offsets
// 	// torch::Tensor AT_col_indices, // output column indices
// 	torch::Tensor A_val,
// 	torch::Tensor A_row_offsets,
// 	torch::Tensor A_col_indices,
// 	int M,
// 	int N) {

//   cusparseHandle_t handle = 0;
//   cusparseCreate(&handle);
//   cusparseSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

//   const auto nnz = A_val.size(0);
//   auto AT_val = torch::zeros_like(A_val);
//   auto AT_row_offsets = torch::zeros(N + 1, torch::dtype(torch::kInt32).layout(torch::kStrided).device(A_val.device()));
//   auto AT_col_indices = torch::zeros_like(A_col_indices);

//   size_t buffer_size = 0;
//   cusparseCsr2cscEx2_bufferSize(
// 	  handle,
// 	  M,
// 	  N,
// 	  nnz,
// 	  A_val.data<float>(),
// 	  A_row_offsets.data<int>(),
// 	  A_col_indices.data<int>(),
// 	  AT_val.data<float>(),
// 	  AT_row_offsets.data<int>(),
// 	  AT_col_indices.data<int>(),
// 	  CUDA_R_32F,
// 	  CUSPARSE_ACTION_NUMERIC,
// 	  CUSPARSE_INDEX_BASE_ZERO,
// 	  CUSPARSE_CSR2CSC_ALG1,
// 	  &buffer_size
//   );

//   int buffer_size_signed = (buffer_size + sizeof(float) - 1) / sizeof(float);
//   auto workspace =
// 	  torch::zeros(buffer_size_signed, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(A_val.device()));

//   cusparseCsr2cscEx2(
// 	  handle,
// 	  M,
// 	  N,
// 	  nnz,
// 	  A_val.data<float>(),
// 	  A_row_offsets.data<int>(),
// 	  A_col_indices.data<int>(),
// 	  AT_val.data<float>(),
// 	  AT_row_offsets.data<int>(),
// 	  AT_col_indices.data<int>(),
// 	  CUDA_R_32F,
// 	  CUSPARSE_ACTION_NUMERIC,
// 	  CUSPARSE_INDEX_BASE_ZERO,
// 	  CUSPARSE_CSR2CSC_ALG1,
// 	  workspace.data<float>()
//   );

//   cusparseDestroy(handle);

//   return {AT_val, AT_row_offsets, AT_col_indices};
// }

// std::vector <torch::Tensor> csr_transpose_cuda_fp16(
// 	// torch::Tensor AT_val, // output values
// 	// torch::Tensor AT_row_offsets, // output row offsets
// 	// torch::Tensor AT_col_indices, // output column indices
// 	torch::Tensor A_val,
// 	torch::Tensor A_row_offsets,
// 	torch::Tensor A_col_indices,
// 	int M,
// 	int N) {

//   cusparseHandle_t handle = 0;
//   cusparseCreate(&handle);
//   cusparseSetStream(handle, at::cuda::getCurrentCUDAStream().stream());

//   const auto nnz = A_val.size(0);
//   auto AT_val = torch::zeros_like(A_val);
//   auto AT_row_offsets = torch::zeros(N + 1, torch::dtype(torch::kInt32).layout(torch::kStrided).device(A_val.device()));
//   auto AT_col_indices = torch::zeros_like(A_col_indices);

//   size_t buffer_size = 0;
//   cusparseCsr2cscEx2_bufferSize(
// 	  handle,
// 	  M,
// 	  N,
// 	  nnz,
// 	  A_val.data<at::Half>(),
// 	  A_row_offsets.data<int>(),
// 	  A_col_indices.data<int>(),
// 	  AT_val.data<at::Half>(),
// 	  AT_row_offsets.data<int>(),
// 	  AT_col_indices.data<int>(),
// 	  CUDA_R_16F,
// 	  CUSPARSE_ACTION_NUMERIC,
// 	  CUSPARSE_INDEX_BASE_ZERO,
// 	  CUSPARSE_CSR2CSC_ALG1,
// 	  &buffer_size
//   );

//   int buffer_size_signed = (buffer_size + sizeof(float) - 1) / sizeof(float);
//   auto workspace =
// 	  torch::zeros(buffer_size_signed, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(A_val.device()));

//   cusparseCsr2cscEx2(
// 	  handle,
// 	  M,
// 	  N,
// 	  nnz,
// 	  A_val.data<at::Half>(),
// 	  A_row_offsets.data<int>(),
// 	  A_col_indices.data<int>(),
// 	  AT_val.data<at::Half>(),
// 	  AT_row_offsets.data<int>(),
// 	  AT_col_indices.data<int>(),
// 	  CUDA_R_16F,
// 	  CUSPARSE_ACTION_NUMERIC,
// 	  CUSPARSE_INDEX_BASE_ZERO,
// 	  CUSPARSE_CSR2CSC_ALG1,
// 	  workspace.data<float>()
//   );

//   cusparseDestroy(handle);

//   return {AT_val, AT_row_offsets, AT_col_indices};
// }