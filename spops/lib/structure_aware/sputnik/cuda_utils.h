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

#ifndef THIRD_PARTY_SPUTNIK_CUDA_UTILS_H_
#define THIRD_PARTY_SPUTNIK_CUDA_UTILS_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sputnik {
typedef __half half;
typedef __half2 half2;

struct __align__(8) half4 {
	half2 x, y;
};

struct __align__(16) half8 {
	half2 x, y, z, w;
};

struct __align__(8) short4 {
	short2 x, y;
};

struct __align__(16) short8 {
	short2 x, y, z, w;
};

#define MISC_PERF 0
#define USE_INT16 1

#if USE_INT16
using INDEX_TYPE = short;
/**
 * @brief Functor to translate vector data types to vector index types.
 */
template<typename Value> struct Value2SharedIndex {
	typedef INDEX_TYPE Index;
};

template<> struct Value2SharedIndex<float2> {
	typedef short2 Index;
};

template<> struct Value2SharedIndex<float4> {
	typedef short4 Index;
};
#else
using INDEX_TYPE = int;
/**
 * @brief Functor to translate vector data types to vector index types.
 */
template <typename Value>
struct Value2SharedIndex {
  typedef int Index;
};

template <>
struct Value2SharedIndex<float2> {
  typedef int2 Index;
};

template <>
struct Value2SharedIndex<float4> {
  typedef int4 Index;
};

#endif
} // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_CUDA_UTILS_H_
