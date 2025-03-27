/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_UTIL_HPP_
#define LIBRARY_SRC_UTIL_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>

#include "assembly.hpp"
#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "constants.hpp"

namespace rocshmem {

#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

#define CHECK_HIP(cmd)                                                        \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#ifdef DEBUG
#define DPRINTF(...)     \
  do {                   \
    printf(__VA_ARGS__); \
  } while (0);
#else
#define DPRINTF(...) \
  do {               \
  } while (0);
#endif

#ifdef DEBUG
#define GPU_DPRINTF(...)      \
  do {                        \
    gpu_dprintf(__VA_ARGS__); \
  } while (0);
#else
#define GPU_DPRINTF(...) \
  do {                   \
  } while (0);
#endif

extern const int gpu_clock_freq_mhz;

/* Device-side internal functions */
__device__ __forceinline__ uint32_t lowerID() {
  return __ffsll(__ballot(1)) - 1;
}

__device__ __forceinline__ int wave_SZ() { return __popcll(__ballot(1)); }

/*
 * Returns true if the caller's thread index is (0, 0, 0) in its block.
 */
__device__ __forceinline__ bool is_thread_zero_in_block() {
  return hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0;
}

/*
 * Returns true if the caller's block index is (0, 0, 0) in its grid.  All
 * threads in the same block will return the same answer.
 */
__device__ __forceinline__ bool is_block_zero_in_grid() {
  return hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0;
}

/*
 * Returns the number of threads in the caller's flattened thread block.
 */
__device__ __forceinline__ int get_flat_block_size() {
  return hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
}

/*
 * Returns the number of threads in the caller's flattened grid.
 */
__device__ __forceinline__ int get_flat_grid_size() {
  return get_flat_block_size() * hipGridDim_x * hipGridDim_y * hipGridDim_z;
}

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
__device__ __forceinline__ int get_flat_block_id() {
  return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
         hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
}

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
__device__ __forceinline__ int get_flat_grid_id() {
  return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
         hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
}

/*
 * Returns the flattened thread index of the calling thread within the grid.
 */
__device__ __forceinline__ int get_flat_id() {
    return get_flat_grid_id() * (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z) + get_flat_block_id();
}

/*
 * Returns true if the caller's thread flad_id is 0 in its wave.
 */
__device__ __forceinline__ bool is_thread_zero_in_wave() {
  return (get_flat_block_id() % WF_SIZE) == 0;
}

extern __constant__ int* print_lock;

template <typename... Args>
__device__ void gpu_dprintf(const char* fmt, const Args&... args) {
  for (int i{0}; i < WF_SIZE; i++) {
    if ((get_flat_block_id() % WF_SIZE) == i) {
      /*
       * GPU-wide global lock that ensures that both prints are executed
       * by a single thread atomically.  We deliberately break control
       * flow so that only a single thread in a WF accesses the lock at a
       * time.  If multiple threads in the same WF attempt to gain the
       * lock at the same time, you have a classic GPU control flow
       * deadlock caused by threads in the same WF waiting on each other.
       */
      while (atomicCAS(print_lock, 0, 1) == 1) {
      }

      printf("WG (%u, %u, %u) TH (%u, %u, %u) ", hipBlockIdx_x,
             hipBlockIdx_y, hipBlockIdx_z, hipThreadIdx_x, hipThreadIdx_y,
             hipThreadIdx_z);
      printf(fmt, args...);

      *print_lock = 0;
    }
  }
}

__device__ __forceinline__ void memcpy(void* dst, void* src, size_t size) {
  uint8_t* dst_bytes{static_cast<uint8_t*>(dst)};
  uint8_t* src_bytes{static_cast<uint8_t*>(src)};

  for (size_t i = 8; i > 1; i >>= 1) {
    while (size >= i) {
      store_asm(src_bytes, dst_bytes, i);
      src_bytes += i;
      dst_bytes += i;
      size -= i;
    }
  }

  if (size == 1) {
    *dst_bytes = *src_bytes;
  }
}

__device__ __forceinline__ void memcpy_wg(void* dst, void* src, size_t size) {
  int thread_id{get_flat_block_id()};
  int block_size{get_flat_block_size()};

  int cpy_size{};
  uint8_t* dst_bytes{nullptr};
  uint8_t* dst_def{nullptr};
  uint8_t* src_bytes{nullptr};
  uint8_t* src_def{nullptr};

  dst_def = reinterpret_cast<uint8_t*>(dst);
  src_def = reinterpret_cast<uint8_t*>(src);
  dst_bytes = dst_def;
  src_bytes = src_def;

  for (int j{8}; j > 1; j >>= 1) {
    cpy_size = size / j;
    for (int i{thread_id}; i < cpy_size; i += block_size) {
      dst_bytes = dst_def;
      src_bytes = src_def;

      src_bytes += i * j;
      dst_bytes += i * j;

      store_asm(src_bytes, dst_bytes, j);
    }
    size -= cpy_size * j;
    dst_def += cpy_size * j;
    src_def += cpy_size * j;
  }

  if (size == 1) {
    if (is_thread_zero_in_block()) {
      *dst_bytes = *src_bytes;
    }
  }
}

__device__ __forceinline__ void memcpy_wave(void* dst, void* src, size_t size) {
  int wave_tid = get_flat_block_id() % WF_SIZE;
  int wave_size{wave_SZ()};

  int cpy_size{};
  uint8_t* dst_bytes{nullptr};
  uint8_t* dst_def{nullptr};
  uint8_t* src_bytes{nullptr};
  uint8_t* src_def{nullptr};

  dst_def = reinterpret_cast<uint8_t*>(dst);
  src_def = reinterpret_cast<uint8_t*>(src);
  dst_bytes = dst_def;
  src_bytes = src_def;

  for (int j{8}; j > 1; j >>= 1) {
    cpy_size = size / j;
    for (int i{wave_tid}; i < cpy_size; i += wave_size) {
      dst_bytes = dst_def;
      src_bytes = src_def;

      src_bytes += i * j;
      dst_bytes += i * j;

      store_asm(src_bytes, dst_bytes, j);
    }
    size -= cpy_size * j;
    dst_def += cpy_size * j;
    src_def += cpy_size * j;
  }

  if (size == 1) {
    if (is_thread_zero_in_wave()) {
      *dst_bytes = *src_bytes;
    }
  }
}

int rocm_init();

void rocm_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr,
                                    int gpu_id);

// Returns clock frequency used by s_memrealtime() in Mhz
uint64_t wallClk_freq_mhz();

struct rocshmem_env_config_t {
  int ro_disable_ipc = 0;
  int ro_progress_delay = 3;
};
extern struct rocshmem_env_config_t rocshmem_env_config;

void rocshmem_env_config_init(void);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_UTIL_HPP_
