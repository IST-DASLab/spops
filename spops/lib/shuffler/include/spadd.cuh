#pragma once

#include <type_traits>
#include "common.cuh"

namespace e {

// B += A
template<class T, class U,
	class INDEX_TYPE,
	class ACCUMULATOR_TYPE>
__global__ void spadd_csr_v1(
	int m,
	int n,
	ACCUMULATOR_TYPE *row_offsets,
	INDEX_TYPE *row_ptr,
	INDEX_TYPE *col_ptr,
	T *values,
	U *b_values
) {
  auto r = blockIdx.x;
  if (r >= m) {
	return;
  }
  r = row_ptr[r];
  for (int i = row_offsets[r] + threadIdx.x; i < row_offsets[r + 1]; i += blockDim.x) {
	auto c = col_ptr[i];
	auto val = values[i];
	if constexpr (std::is_same<T, float>::value) {
	  b_values[r * n + c] += val;
	} else {
	  auto out = b_values[r * n + c];
	  b_values[r * n + c] = __hadd(out, val);
	}
  }
}
}
