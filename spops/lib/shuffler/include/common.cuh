#pragma once

#include <cuda_fp16.h>

#include <type_traits>

#define CUINLINE __forceinline__
namespace e {
static constexpr int WARP_SIZE = 32;

};
// half2: {x, y} x is lower, y is upper.
struct __align__(16) half8 { half2 x, y, z, w; };

struct __align__(16) half4 { half2 x, y; };

enum class ThreadDim { X, Y, Z, YZ, XYZ };

__device__ __host__ CUINLINE int updiv(int x, int y) { return (x + y - 1) / y; }

template<int x, int y>
__device__ __host__ CUINLINE constexpr int updiv() { return (x + y - 1) / y; }

CUINLINE __device__ unsigned int block_id() {
  return gridDim.y * blockIdx.x + blockIdx.y;
}

CUINLINE __device__ unsigned int total_threads() {
  return blockDim.x * blockDim.y * blockDim.z;
}

// TODO: Remove
CUINLINE __device__ unsigned int thread_id2() {
  return blockDim.y * threadIdx.x + threadIdx.y;
}

template <ThreadDim t> CUINLINE __device__ unsigned int get_thread_count() {
  if constexpr (t == ThreadDim::X) {
    return blockDim.x;
  } else if constexpr (t == ThreadDim::Y) {
    return blockDim.y;
  } else if constexpr (t == ThreadDim::Z) {
    return blockDim.z;
  } else if constexpr (t == ThreadDim::YZ) {
    return blockDim.y * blockDim.z;
  } else {
    static_assert("Invalid ID requested.");
  }
}

CUINLINE __device__ __host__ unsigned int get_global_id() {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

template <ThreadDim t>
CUINLINE __device__ __host__ unsigned int get_thread_id() {
  if constexpr (t == ThreadDim::X) {
    return threadIdx.x;
  } else if constexpr (t == ThreadDim::Y) {
    return threadIdx.y;
  } else if constexpr (t == ThreadDim::Z) {
    return threadIdx.z;
  } else if constexpr (t == ThreadDim::YZ) {
    return threadIdx.y * blockDim.z + threadIdx.z;
  } else if constexpr (t == ThreadDim::XYZ) {
    return threadIdx.x * blockDim.x * blockDim.y + (threadIdx.y * blockDim.z + threadIdx.z);
  } else {
    // static_assert(false, "Invalid ID requested.");
  }
}

/**
 * Async and branchless clear. Zeros out data block of size n using THREAD_COUNT
 * threads.
 * @tparam T
 * @tparam D
 * @param ptr
 * @param n
 */
template <class T, ThreadDim D>
__device__ CUINLINE void clr_bless_async(T *__restrict__ ptr, int n, T val = T(0)) {
  unsigned int thread_id = get_thread_id<D>();
  unsigned int thread_count = get_thread_count<D>();
  unsigned int work_to_do = updiv(n, thread_count);
  for (int i = 0; i < work_to_do; i++) {
    ptr[work_to_do * thread_id + i] = val;
  }
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void
memcpy2d_strided_async_flat(T *__restrict__ dest, T *__restrict__ src, u32 dest_m,
                            u32 dest_n, u32 src_m, u32 src_n) {
  auto thread_id = get_thread_id<D>();
  auto thread_count = get_thread_count<D>();

  auto work_to_do = (dest_m * dest_n) / thread_count; // TODO: Maybe do updiv?
  auto work_offset = thread_id * work_to_do;

  if constexpr (std::is_same<half, T>::value) {
    for (int i = work_offset; i < work_to_do + work_offset; i += 8) {
      reinterpret_cast<half8 *>(dest)[i / 8] =
          reinterpret_cast<half8 *>(src)[i / 8];
    }
  } else {
    for (int i = work_offset; i < work_to_do + work_offset; i += 4) {
    }
  }
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void memcpy2d_strided_async_2d_bounds_check(
    T *__restrict__ dest_base, T *__restrict__ src_base, u32 dest_m, u32 dest_n,
    u32 src_x, u32 src_y, u32 src_m, u32 src_n) {
  auto thread_id = get_thread_id<D>();
  auto thread_count = get_thread_count<D>();
  dest_n /= 2;
  auto total_work = dest_m * dest_n;
  auto work_to_do = total_work / thread_count;
  auto work_offset = thread_id * work_to_do;
  for (int i = work_offset; i < work_to_do + work_offset; i++) {
    u32 dest_x = i / src_n;
    u32 dest_y = i % src_n;
    u32 inbound = (src_x + dest_x < src_m) && (src_y + dest_y < src_n);
    auto src_addr = inbound * ((src_x + dest_x) * src_n + src_y + dest_y);
    dest_base[i] = T(inbound) * src_base[src_addr];
  }
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void memcpy2d_strided_async_2d_bounds_check(
    T *__restrict__ dest_base, T *__restrict__ src_base, u32 dest_m, u32 src_x,
    u32 src_y, u32 src_m, u32 src_n) {
  memcpy2d_strided_async_2d_bounds_check<T, D>(
      dest_base, src_base, dest_m, dest_m, src_x, src_y, src_m, src_n);
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void
memcpy2d_strided_async_flat(T *__restrict__ dest, T *__restrict__ src, u32 n) {
  memcpy2d_strided_async_flat<T, D>(dest, src, n, n, n, n);
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void
memcpy2d_strided_async_warp_half_256_flat(T *__restrict__ dest,
                                          T *__restrict__ src) {
  auto thread_id = get_thread_id<D>();
  reinterpret_cast<half8 *>(dest)[thread_id] =
      reinterpret_cast<half8 *>(src)[thread_id];
}

template <class T, ThreadDim D>
__device__ CUINLINE void memcpy2d_strided_async_2d_256_warp(
    T *__restrict__ dest_base, T *__restrict__ src_base, u32 dest_m, u32 dest_n,
    u32 src_x, u32 src_y, u32 src_m, u32 src_n) {
  auto thread_id = get_thread_id<D>();
  auto work_to_do = 8;
  auto work_offset = thread_id * work_to_do;
  u32 dest_x = work_offset / src_n;
  u32 dest_y = work_offset % src_n;
  auto src = &src_base[(src_x + dest_x) * src_n + src_y];
  auto dst = &dest_base[dest_x * dest_n];

  // TODO: Fix this
  reinterpret_cast<half8 *>(dst)[dest_y / 8] =
      reinterpret_cast<half8 *>(src)[dest_y / 8];
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void memcpy2d_strided_async_2d_256_warp_square(
    T *__restrict__ dest_base, T *__restrict__ src_base, u32 dest_m, u32 src_x,
    u32 src_y, u32 src_m, u32 src_n) {
  int dest_n = dest_m;
  memcpy2d_strided_async_2d_256_warp<T, D>(dest_base, src_base, dest_m, dest_n,
                                           src_x, src_y, src_m, src_n);
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void
memcpy2d_strided_async_2d_128_warp(T *__restrict__ dest_base,
                                   T *__restrict__ src_base, u32 dest_m,
                                   u32 src_x, u32 src_y, u32 src_m, u32 src_n) {
  int dest_n = dest_m;
  auto thread_id = get_thread_id<D>();
  auto work_to_do = 4;
  auto work_offset = thread_id * work_to_do;
  u32 dest_x = work_offset / src_n;
  u32 dest_y = work_offset % src_n;
  auto src = &src_base[(src_x + dest_x) * src_n + src_y];
  reinterpret_cast<half4 *>(dest_base)[work_offset / work_to_do] =
      reinterpret_cast<half4 *>(src_base)[dest_y / work_to_do];
}

template <class T, ThreadDim D>
__device__ CUINLINE void memcpy2d_strided_async_2d_256_warp_masked(
    T *__restrict__ dest_base, T *__restrict__ src_base, u32 dest_m, u32 dest_n,
    u32 src_x, u32 src_y, u32 src_m, u32 src_n, int mask) {
  auto thread_id = get_thread_id<D>();
  auto work_to_do = 8;
  auto work_offset = thread_id * work_to_do;
  u32 dest_x = work_offset / src_n;
  u32 dest_y = work_offset % src_n;
  auto src = &src_base[(src_x + dest_x) * src_n + src_y];
  reinterpret_cast<half8 *>(dest_base)[work_offset / 8] =
      reinterpret_cast<half8 *>(src)[dest_y / 8]; // TODO: src_base?
}

using u32 = unsigned int;
template <class T, ThreadDim D>
__device__ CUINLINE void memcpy2d_strided_async_2d_256_warp_masked(
    T *__restrict__ dest_base, T *__restrict__ src_base, u32 dest_m, u32 src_x,
    u32 src_y, u32 src_m, u32 src_n, int mask) {
  int dest_n = dest_m;
  memcpy2d_strided_async_2d_256_warp_masked<T, D>(
      dest_base, src_base, dest_m, dest_n, src_x, src_y, src_m, src_n, mask);
}
