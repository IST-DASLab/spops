// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>

#include "../barrier.h"
#include "../common.h"
#include "../cuda_utils.h"
#include "../load_store.h"
#include "../sddmm/all_reduce.h"
#include "../sddmm/compute_utils.h"
#include "../sddmm/cuda_sddmm.h"
#include "../sddmm/dense_to_reg.h"
#include "../sddmm/dense_to_shared.h"
#include "../sddmm/output_tile.h"
#include "../tiling_utils.h"

#ifdef SPUTNIK_BUILD_TEST
#include "../matrix_utils.h"
#endif

namespace sputnik {

namespace {
template <typename LoadType, int kBlockItemsY, int kBlockItemsK,
    int kBlockItemsX, int kBlockWidth,
	bool CSR, int kPredicateK = true>
__global__ void __launch_bounds__(kBlockItemsY* kBlockWidth)
    CudaSddmmKernel(int m, int k, int n, const INDEX_TYPE* __restrict__ row_indices,
                    const int* __restrict__ row_offsets,
                    const INDEX_TYPE* __restrict__ column_indices,
                    const float* __restrict__ lhs_matrix,
                    const float* __restrict__ rhs_matrix,
                    float* __restrict__ output_values) {
  static_assert((kBlockItemsY * kBlockWidth) % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
  static_assert((kBlockItemsY * kBlockWidth) > 0,
                "The thread-block size must be nonzero.");
  static_assert(kBlockItemsK >= kBlockWidth,
                "k-dimension tile must be >= block width.");
  static_assert(kBlockItemsK % kBlockWidth == 0,
                "k-dimension tile size must be divisible by block width.");
  static_assert(kBlockItemsX >= kBlockWidth,
                "n-dimension tile size must be >= block width.");
  static_assert(kBlockItemsX % kBlockWidth == 0,
                "n-dimension tile size must be divisible by block width.");
  typedef TilingUtils<kBlockItemsY, kBlockItemsK, kBlockItemsX, CSR> Tiling;
  // Calculate this thread block's indices into the M and N dimensions.
  int m_index = Tiling::IndexM(), n_index = Tiling::IndexN();

  int row_offset, nonzeros;

  // Threads that work on different m-dim indices are independent. If we're
  // out of bounds in the m-dimension we can just return.
  if constexpr (CSR) {
	if (m_index >= m) return;
	m_index = Load(row_indices + m_index);
  } else {
	if (n_index >= n) return;
	n_index = Load(row_indices + n_index);
  }

  // Load the row offset and calculate the number of non-zeros in the row.
  if constexpr (CSR) {
	row_offset = __ldg(row_offsets + m_index);
	nonzeros = __ldg(row_offsets + m_index + 1) - row_offset;

	// If this thread block has no nonzeros in the row to process, exit early.
	if (n_index >= nonzeros) {
	  return;
	}
  } else {
	row_offset = __ldg(row_offsets + n_index);
	nonzeros = __ldg(row_offsets + m_index + 1) - row_offset;

	// If this thread block has no nonzeros in the row to process, exit early.
	if (m_index >= nonzeros) {
	  return;
	}
  }

  // Calculate the number of nonzeros that this thread block processes and
  // substract the x-dim thread index to simplify loop bounds checks.
  if constexpr (CSR) {
	nonzeros = Min(nonzeros - n_index, kBlockItemsX) - threadIdx.x;
  } else {
	nonzeros = Min(nonzeros - m_index, kBlockItemsX) - threadIdx.x;
  }

  // Shared memory tile for the lhs dense matrix values.
  float lhs_fragment[kBlockItemsK / kBlockWidth];

  // Shared memory tile for the output column indices.
  __shared__ INDEX_TYPE column_indices_tile_array[kBlockItemsX * kBlockItemsY];

  INDEX_TYPE* column_indices_tile =
      TilingUtils<kBlockItemsY, kBlockItemsK, kBlockItemsX, CSR>::MaybeOffset(
          column_indices_tile_array, kBlockItemsK * threadIdx.y);

  if constexpr (CSR) {
	// Create a dense-to-shared loader for the lhs matrix.
	DenseToShared<LoadType, kBlockItemsK, kBlockWidth> lhs_tile_loader(
		k, m_index, lhs_matrix, lhs_fragment);

	// Register file fragment for the rhs dense matrix values.
	float rhs_fragment[kBlockItemsK * kBlockItemsX / kBlockWidth];

	// Create a dense-to-register loader for the rhs matrix.
	DenseToReg<LoadType, kBlockItemsK, kBlockItemsX, kBlockWidth> rhs_tile_loader(
		k, row_offset, n_index, column_indices, rhs_matrix, column_indices_tile,
		rhs_fragment);

	// Accumulator registers for the partial results. Initialize the
	// registers to zero s.t. we can always accumulate in-place.
	float accumulator_fragment[kBlockItemsX] = {};

	// Helper for managing syncronization between collaborating threads.
	Barrier<kBlockItemsY, kBlockWidth> barrier(threadIdx.y);

	// Helper for computing tile-level partial matmuls.
	ComputeUtilsSDDMM<kBlockItemsK, kBlockItemsX, kBlockWidth> computer(
		lhs_fragment, rhs_fragment, accumulator_fragment);

	// Registers for the final reduced outputs.
	float output_fragment[kBlockItemsX / kBlockWidth];

	// Helper to reduce the partial accumulators prior to writing.
	AllReduce<LoadType, kBlockItemsX, kBlockWidth> all_reduce(
		barrier.ThreadMask(), accumulator_fragment, output_fragment);

	// Helper for storing the results to the output.
	OutputTileSDDMM<kBlockItemsX, kBlockWidth> output_tile_storer(
		row_offset, n_index, output_fragment, output_values);

	//
	/// Begin kernel main loop.
	//

	// Load the column indices for this n-dimension tile.
	rhs_tile_loader.LoadColumnIndices(nonzeros);
	barrier.Sync();

#pragma nounroll
	for (; k >= kBlockItemsK; k -= kBlockItemsK) {
	  // Load a tile from the dense lhs matrix into smem and sync.
	  lhs_tile_loader.Load();

	  // Load a tile from the dense rhs matrix into registers.
	  rhs_tile_loader.Load();

	  // Multiply the tiles and accumulate the results.
	  computer.TileMAC();
	}

	//
	/// Begin k-dimension residue computation.
	//

	if (kPredicateK) {
	  // Update the residue size to simplify loop bounds checking. Note
	  // that `k` is guaranteed to be a multiple of `kValuesPerLoad`.
	  constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);
	  k -= threadIdx.x * kValuesPerLoad;

	  // Load a partial tile from the lhs matrix and sync.
	  lhs_tile_loader.Residue(k);

	  // Load a tile from the rhs matrix and compute immediately.
	  rhs_tile_loader.ResidueAndCompute(k, lhs_fragment, accumulator_fragment);
	}

	//
	/// Cleanup the partial sums across the (sub)warp.
	//
	all_reduce.Reduce();

	//
	///  Write the results to the output.
	//
	output_tile_storer.Store(nonzeros);
  } else {
	// Create a dense-to-shared loader for the lhs matrix.
	// Register file fragment for the rhs dense matrix values.
	float rhs_fragment[kBlockItemsK * kBlockItemsX / kBlockWidth];

	// Create a dense-to-register loader for the rhs matrix.
	DenseToReg<LoadType, kBlockItemsK, kBlockItemsX, kBlockWidth> rhs_tile_loader(
		k, row_offset, n_index, column_indices, lhs_matrix, column_indices_tile,
		lhs_fragment);

	DenseToShared<LoadType, kBlockItemsK, kBlockWidth> lhs_tile_loader(
		k, m_index, rhs_matrix, rhs_fragment);

	// Accumulator registers for the partial results. Initialize the
	// registers to zero s.t. we can always accumulate in-place.
	float accumulator_fragment[kBlockItemsX] = {};

	// Helper for managing syncronization between collaborating threads.
	Barrier<kBlockItemsY, kBlockWidth> barrier(threadIdx.y);

	// Helper for computing tile-level partial matmuls.
	ComputeUtilsSDDMM<kBlockItemsK, kBlockItemsX, kBlockWidth> computer(
		lhs_fragment, rhs_fragment, accumulator_fragment);

	// Registers for the final reduced outputs.
	float output_fragment[kBlockItemsX / kBlockWidth];

	// Helper to reduce the partial accumulators prior to writing.
	AllReduce<LoadType, kBlockItemsX, kBlockWidth> all_reduce(
		barrier.ThreadMask(), accumulator_fragment, output_fragment);

	// Helper for storing the results to the output.
	OutputTileSDDMM<kBlockItemsX, kBlockWidth> output_tile_storer(
		row_offset, n_index, output_fragment, output_values);

	//
	/// Begin kernel main loop.
	//

	// Load the column indices for this n-dimension tile.
	rhs_tile_loader.LoadColumnIndices(nonzeros);
	barrier.Sync();

#pragma nounroll
	for (; k >= kBlockItemsK; k -= kBlockItemsK) {
	  // Load a tile from the dense lhs matrix into smem and sync.
	  lhs_tile_loader.Load();

	  // Load a tile from the dense rhs matrix into registers.
	  rhs_tile_loader.Load();

	  // Multiply the tiles and accumulate the results.
	  computer.TileMAC();
	}

	//
	/// Begin k-dimension residue computation.
	//

	if (kPredicateK) {
	  // Update the residue size to simplify loop bounds checking. Note
	  // that `k` is guaranteed to be a multiple of `kValuesPerLoad`.
	  constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(float);
	  k -= threadIdx.x * kValuesPerLoad;

	  // Load a partial tile from the lhs matrix and sync.
	  lhs_tile_loader.Residue(k);

	  // Load a tile from the rhs matrix and compute immediately.
	  rhs_tile_loader.ResidueAndCompute(k, lhs_fragment, accumulator_fragment);
	}

	//
	/// Cleanup the partial sums across the (sub)warp.
	//
	all_reduce.Reduce();

	//
	///  Write the results to the output.
	//
	output_tile_storer.Store(nonzeros);
  }
}

}  // namespace

cudaError_t CudaSddmm(int m, int k, int n, int nonzeros, int last0, int last1,
                      const INDEX_TYPE* __restrict__ row_indices,
                      const int* __restrict__ row_offsets,
                      const INDEX_TYPE* __restrict__ column_indices,
                      const float* __restrict__ lhs_matrix,
                      const float* __restrict__ rhs_matrix,
                      float* __restrict__ output_values, cudaStream_t stream,
					  bool csr) {
  // If possible, launch a variant that does not include the k-dimension
  // residue handling code.
  if ((k % 4) == 0) {
    if ((k % 32) == 0) {
      return CudaSddmmEx<float4, /* blockitemsy = */ 4, /* blockitemsk = */ 32, /* kblockitemsx = */ 32, 8, false>(
          m, k, n, nonzeros, last0, last1, row_indices, row_offsets, column_indices,
          lhs_matrix, rhs_matrix, output_values, stream, csr);
    } else {
      return CudaSddmmEx<float4, 4, 32, 32, 8>(
          m, k, n, nonzeros, last0, last1, row_indices, row_offsets, column_indices,
          lhs_matrix, rhs_matrix, output_values, stream, csr);
    }
  } else if ((k % 2) == 0) {
    return CudaSddmmEx<float2, 2, 32, 32, 16>(
        m, k, n, nonzeros, last0, last1, row_indices, row_offsets, column_indices, lhs_matrix,
        rhs_matrix, output_values, stream, csr);
  } else {
  	// Scalar kernel.
  	return CudaSddmmEx<float, 1, 32, 32, 32>(
				m, k, n, nonzeros, last0, last1, row_indices, row_offsets, column_indices, lhs_matrix,
				rhs_matrix, output_values, stream, csr);
  }
}

template <typename LoadType, int kBlockItemsY, int kBlockItemsK,
          int kBlockItemsX, int kBlockWidth, int kPredicateK>
cudaError_t CudaSddmmEx(
    int m, int k, int n, int nonzeros, int last0, int last1, const INDEX_TYPE* __restrict__ row_indices,
    const int* __restrict__ row_offsets, const INDEX_TYPE* __restrict__ column_indices,
    const float* __restrict__ lhs_matrix, const float* __restrict__ rhs_matrix,
    float* __restrict__ output_values, cudaStream_t stream, bool csr) {
  dim3 block_dim(kBlockWidth, kBlockItemsY, 1);
  if (csr) {
	dim3 grid_dim(std::ceil(static_cast<float>(last0) / kBlockItemsY),
				  std::ceil(static_cast<float>(last1) / kBlockItemsX), 1);
	constexpr bool CSR = true;
	CudaSddmmKernel<LoadType, kBlockItemsY, kBlockItemsK, kBlockItemsX,
					kBlockWidth, CSR, kPredicateK><<<grid_dim, block_dim, 0, stream>>>(
		m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix,
		output_values);
  } else {
	dim3 grid_dim(std::ceil(static_cast<float>(last1) / kBlockItemsX), // TODO(elvircrn): Swap X/Y?
				  std::ceil(static_cast<float>(last0) / kBlockItemsY), 1);
	// CSC
	constexpr bool CSR = false;
	CudaSddmmKernel<LoadType, kBlockItemsY, kBlockItemsK, kBlockItemsX,
					kBlockWidth, CSR, kPredicateK><<<grid_dim, block_dim, 0, stream>>>(
		m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix,
		output_values);
  }
  return cudaGetLastError();
}


__global__ void ConvertKernel_(const int *in, short *out, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	out[idx] = static_cast<short>(in[idx]);
}

inline cudaError_t ConvertHost(const int *in, short *out, int n) {
	if (n == 0) return cudaSuccess;
	int threads_per_block = 64;
	int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
	ConvertKernel_<<<blocks_per_grid, threads_per_block, 0, 0>>>(in, out, n);
	return cudaGetLastError();
}

template <typename LoadType, int kBlockItemsY, int kBlockItemsK,
          int kBlockItemsX, int kBlockWidth, int kPredicateK>
cudaError_t CudaSddmmExTest(
    int m, int k, int n, int nonzeros, const int* __restrict__ row_indices,
    const int* __restrict__ row_offsets, const int* __restrict__ column_indices,
    const float* __restrict__ lhs_matrix, const float* __restrict__ rhs_matrix,
    float* __restrict__ output_values, cudaStream_t stream) {
  dim3 grid_dim(std::ceil(static_cast<float>(m) / kBlockItemsY),
                std::ceil(static_cast<float>(n) / kBlockItemsX), 1);
  dim3 block_dim(kBlockWidth, kBlockItemsY, 1);

#if USE_INT16
	INDEX_TYPE *row_indices_;
	INDEX_TYPE *column_indices_;
	cudaMalloc((void**)&row_indices_, size_t(sizeof(short) * (m + 2)));
	cudaMalloc((void**)&column_indices_, size_t(sizeof(short) * (nonzeros + 2)));

	ConvertHost(row_indices, row_indices_, m);
	ConvertHost(column_indices, column_indices_, nonzeros);
	cudaDeviceSynchronize();
#else
	const INDEX_TYPE *row_indices_ = row_indices;
	const INDEX_TYPE *column_indices_ = column_indices;
#endif

  CudaSddmmKernel<LoadType, kBlockItemsY, kBlockItemsK, kBlockItemsX,
                  kBlockWidth, kPredicateK><<<grid_dim, block_dim, 0, stream>>>(
      m, k, n, row_indices_, row_offsets, column_indices_, lhs_matrix, rhs_matrix,
      output_values);

	// TODO(elvircrn): Free up row_indices_
  return cudaGetLastError();
}

#define INSTANTIATE_TILED(fn, ltype, mt, kt, nt, bs)                        \
  template cudaError_t fn<ltype, mt, kt, nt, bs>(                           \
      int, int, int, int, const int*, const int*, const int*, const float*, \
      const float*, float*, cudaStream_t)

#ifdef SPUTNIK_BUILD_TEST
INSTANTIATE_TILED(CudaSddmmExTest, float, 1, 32, 32, 32);
INSTANTIATE_TILED(CudaSddmmExTest, float2, 2, 32, 32, 16);
INSTANTIATE_TILED(CudaSddmmExTest, float4, 4, 32, 32, 8);
#endif  // SPUTNIK_BUILD_TEST

}  // namespace sputnik
