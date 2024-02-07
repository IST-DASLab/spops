#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cstdio>

namespace v2 {
#include "./structure_aware/sputnik/sddmm/cuda_sddmm.cu.cc"
#include "./structure_aware/sputnik/cuda_utils.h"
}
#include "cusparse.h"
#include <iostream>
#include <cstdio>
#include "cuda_fp16.h"

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

std::vector <torch::Tensor> structure_aware_sddmm_cuda_fp32_benchmark(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	int last0,
	int last1,
	torch::Tensor A, // (M, K)
	torch::Tensor BT, // (N, K)
	bool csr,
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
  v2::sputnik::CudaSddmm(
	  /*m=*/ M,
	  /*k=*/ K,
	  /*n=*/ N,
	  /*nonzeros=*/ nnz,
	  /*last0=*/last0,
	  /*last1=*/last1,
	  /*row_indices=*/ M_row_indices.data<v2::sputnik::INDEX_TYPE>(),
	  /*row_offsets=*/ M_row_offsets.data<int>(),
	  /*column_indices=*/ M_col_indices.data<v2::sputnik::INDEX_TYPE>(),
	  /*lhs_matrix=*/ A.data<float>(),
	  /*rhs_matrix=*/ BT.data<float>(),
	  /*output_values=*/ C_val.data<float>(),
	  /*stream=*/ at::cuda::getCurrentCUDAStream().stream(),
	  /*csr=*/ csr
  );

  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  t.data<float>()[0] = timer.end();

  return {C_val};
}

std::vector <torch::Tensor> structure_aware_sddmm_cuda_fp32(
	// torch::Tensor C_val, // the output values
	torch::Tensor M_row_offsets,
	torch::Tensor M_row_indices,
	torch::Tensor M_col_indices,
	int last0,
	int last1,
	torch::Tensor A, // (M, K)
	torch::Tensor BT, // (N, K)
	bool csr
) {
  const auto M = A.size(0);
  const auto K = A.size(1);
  const auto N = BT.size(0);
  const auto nnz = M_col_indices.size(0);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto C_val = torch::zeros(nnz, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(A.device()));

  v2::sputnik::CudaSddmm(
	  /*m=*/ M,
	  /*k=*/ K,
	  /*n=*/ N,
	  /*nonzeros=*/ nnz,
	  /*last0=*/last0,
	  /*last1=*/last1,
	  /*row_indices=*/ M_row_indices.data<v2::sputnik::INDEX_TYPE>(),
	  /*row_offsets=*/ M_row_offsets.data<int>(),
	  /*column_indices=*/ M_col_indices.data<v2::sputnik::INDEX_TYPE>(),
	  /*lhs_matrix=*/ A.data<float>(),
	  /*rhs_matrix=*/ BT.data<float>(),
	  /*output_values=*/ C_val.data<float>(),
	  /*stream=*/ at::cuda::getCurrentCUDAStream().stream(),
	  /*csr=*/ csr
  );

  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  return {C_val};
}
