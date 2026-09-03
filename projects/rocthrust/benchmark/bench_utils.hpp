// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <thrust/execution_policy.h>

#include "primbench.hpp"

#ifndef _MSC_VER
using int128_t  = __int128_t;
using uint128_t = __uint128_t;
#endif

#include "generation_utils.hpp" // IWYU pragma: export

/// This allows running rocThrust benchmarks with CCCL Thrust.
#ifndef _THRUST_HAS_DEVICE_SYSTEM_STD
#  define THRUST_HOST_DEVICE __host__ __device__
#  define THRUST_DEVICE      __device__

#  define _THRUST_LIBCXX_INCLUDE(LIB) <cuda/LIB>
#  define _THRUST_STD                 ::cuda::std
#  define _THRUST_LIBCXX              ::cuda
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#  include <hip/hip_runtime.h>
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>

namespace bench_utils
{

#define HIP_CHECK(condition)                                                     \
  {                                                                              \
    hipError_t error = condition;                                                \
    if (error != hipSuccess)                                                     \
    {                                                                            \
      std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
      exit(error);                                                               \
    }                                                                            \
  }

struct less_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }
};

struct max_t
{
  template <typename T>
  __host__ __device__ T operator()(const T& lhs, const T& rhs)
  {
    less_t less{};
    return less(lhs, rhs) ? rhs : lhs;
  }
};

/**
 * This struct is used to reduce noise in benchmarks
 */
struct caching_allocator_t
{
  using value_type = char;

  caching_allocator_t() = default;
  ~caching_allocator_t()
  {
    free_all();
  }

  char* allocate(std::ptrdiff_t num_bytes)
  {
    value_type* result{};
    auto free_block = free_blocks.find(num_bytes);
    if (free_block != free_blocks.end())
    {
      result = free_block->second;
      free_blocks.erase(free_block);
    }
    else
    {
      HIP_CHECK(hipMalloc(&result, num_bytes));
    }

    allocated_blocks.emplace(result, num_bytes);
    return result;
  }

  void deallocate(value_type* ptr, size_t)
  {
    auto iter = allocated_blocks.find(ptr);
    if (iter == allocated_blocks.end())
    {
      throw std::runtime_error("Memory was not allocated by this allocator");
    }

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);
    free_blocks.emplace(num_bytes, ptr);
  }

private:
  using FreeBlocksType      = std::multimap<std::ptrdiff_t, value_type*>;
  using AllocatedBlocksType = std::map<value_type*, std::ptrdiff_t>;

  FreeBlocksType free_blocks;
  AllocatedBlocksType allocated_blocks;

  void free_all()
  {
    for (auto free_block : free_blocks)
    {
      HIP_CHECK(hipFree(free_block.second));
    }

    for (auto allocated_block : allocated_blocks)
    {
      HIP_CHECK(hipFree(allocated_block.first));
    }
  }
};

class large_data
{
public:
  __host__ __device__ large_data()
  {
    data[0] = 0;
  }
  __host__ __device__ large_data(large_data const& val)
  {
    data[0] = val.data[0];
  }
  __host__ __device__ large_data(int n)
  {
    data[0] = static_cast<int8_t>(n);
  }
  large_data& __host__ __device__ operator=(large_data const& val)
  {
    data[0] = val.data[0];
    return *this;
  }
  bool __host__ __device__ operator==(large_data const& val) const
  {
    return data[0] == val.data[0];
  }
  large_data& __host__ __device__ operator++()
  {
    ++data[0];
    return *this;
  }
  __host__ __device__ operator int() const
  {
    return static_cast<int>(data[0]);
  }

  int8_t data[512];
};

template <class T>
bool __host__ __device__ operator==(T const& lhs, large_data const& rhs)
{
  return static_cast<large_data>(lhs).data[0] == rhs.data[0];
}

inline size_t total_global_mem()
{
  static const size_t mem = [] {
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    return prop.totalGlobalMem;
  }();
  return mem;
}

inline bool does_size_fit(size_t bytes_per_element, size_t size)
{
  return bytes_per_element * size <= total_global_mem();
}

inline std::vector<size_t> sizes(size_t bytes_per_element)
{
  constexpr size_t all_sizes[] = {1u << 16, 1u << 20, 1u << 24, 1u << 28};

  std::vector<size_t> result;
  for (size_t size : all_sizes)
  {
    if (does_size_fit(bytes_per_element, size))
    {
      result.push_back(size);
    }
  }
  return result;
}

} // namespace bench_utils

PRIMBENCH_REGISTER_TYPE(int8_t, "i8")
PRIMBENCH_REGISTER_TYPE(int16_t, "i16")
PRIMBENCH_REGISTER_TYPE(int32_t, "i32")
PRIMBENCH_REGISTER_TYPE(int64_t, "i64")

PRIMBENCH_REGISTER_TYPE(uint8_t, "u8")
PRIMBENCH_REGISTER_TYPE(uint16_t, "u16")
PRIMBENCH_REGISTER_TYPE(uint32_t, "u32")
PRIMBENCH_REGISTER_TYPE(uint64_t, "u64")

#ifndef _MSC_VER
PRIMBENCH_REGISTER_TYPE(int128_t, "i128")
PRIMBENCH_REGISTER_TYPE(uint128_t, "u128")
#endif
PRIMBENCH_REGISTER_TYPE(float, "f32")
PRIMBENCH_REGISTER_TYPE(double, "f64")
