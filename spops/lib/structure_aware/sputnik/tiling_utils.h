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

#ifndef THIRD_PARTY_SPUTNIK_TILING_UTILS_H_
#define THIRD_PARTY_SPUTNIK_TILING_UTILS_H_

/**
 * @file @brief Utilities for mapping threads and thread blocks to matrix tiles.
 */

namespace sputnik {

/**
 * @brief Helper to simplify expressing indexing math efficiently.
 *
 * Offsets the input pointer in bytes.
 */
template <typename OutType, typename InType>
__device__ __forceinline__ OutType* OffsetCast(InType* ptr, int offset) {
  return reinterpret_cast<OutType*>(
      const_cast<char*>(reinterpret_cast<const char*>(ptr)) + offset);
}

/**
 * @brief Utilities for calculating M & N dimension tile mappings.
 */
template <int kBlockItemsY, int kBlockItemsK, int kBlockItemsX, bool CSR>
struct TilingUtils {
  static __device__ __forceinline__ int IndexM() {
	if constexpr (CSR) {
	  return blockIdx.x * kBlockItemsY + threadIdx.y;
	} else {
	  return blockIdx.y * kBlockItemsX;
	}
  }

  static __device__ __forceinline__ int IndexN() {
	if constexpr (CSR) {
	  return blockIdx.y * kBlockItemsX;
	} else {
	  return blockIdx.x * kBlockItemsY + threadIdx.y;
	}
  }


  template <typename T>
  static __device__ __forceinline__ T* MaybeOffset(T* ptr, int off) {
    return ptr + off;
  }
};

/**
 * @brief Specialization for 1-dimensional warp-level tiles.
 *
 * Skip some unescesary indexing math when all threads in a warp share
 * the same tile.
 */
template <int kBlockItemsK, int kBlockItemsX, bool CSR>
struct TilingUtils<1, kBlockItemsK, kBlockItemsX, CSR> {
  static __device__ __forceinline__ int IndexM() {
	if constexpr (CSR) {
	  return blockIdx.x;
	} else {
	  return blockIdx.y * kBlockItemsX;
	}
  }

  static __device__ __forceinline__ int IndexN() {
	if constexpr (CSR) {
	  return blockIdx.y * kBlockItemsX;
	} else {
	  return blockIdx.x;
	}
  }

  template <typename T>
  static __device__ __forceinline__ T* MaybeOffset(T* ptr, int /* unused */) {
    return ptr;
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_TILING_UTILS_H_
