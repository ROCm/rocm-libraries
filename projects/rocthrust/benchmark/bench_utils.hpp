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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

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
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

inline constexpr size_t validation_items = 1u << 16;

template <class T>
thrust::host_vector<T> read_range(const thrust::device_vector<T>& values, size_t offset, size_t count)
{
  if (count == 0)
  {
    return {};
  }
  const auto first = values.cbegin() + offset;
  return thrust::host_vector<T>(first, first + count);
}

template <class T, class U, class Equal = std::equal_to<>>
void check_values(const thrust::host_vector<T>& actual, const thrust::host_vector<U>& expected, Equal equal = {})
{
  PRIMBENCH_ASSERT(actual.size(), expected.size());
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin(), equal);
  PRIMBENCH_ASSERT(static_cast<size_t>(mismatch.first - actual.begin()), expected.size());
}

template <class T>
void check_output(const thrust::device_vector<T>& output, const thrust::host_vector<T>& expected, size_t offset = 0)
{
  check_values(read_range(output, offset, expected.size()), expected);
}

template <class T>
thrust::host_vector<T> arithmetic_input(size_t items)
{
  constexpr size_t max_items = sizeof(T) == 1 ? 16 : 4096;
  thrust::host_vector<T> input(std::min(items, max_items));
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = T(1 + (i % 3 == 0));
  }
  return input;
}

template <class Key, class Compare, class Sort>
void check_key_sort(const thrust::device_vector<Key>& keys,
                    const thrust::device_vector<Key>& input,
                    primbench::stream_t stream,
                    Compare compare,
                    Sort sort)
{
  const auto count = std::min(keys.size(), validation_items);
  auto expected    = read_range(input, 0, count);
  auto actual      = read_range(keys, 0, count);
  PRIMBENCH_ASSERT(std::is_sorted(actual.begin(), actual.end(), compare), true);

  if (count != keys.size())
  {
    thrust::device_vector<Key> replay = expected;
    sort(replay);
    HIP_CHECK(hipStreamSynchronize(stream));
    actual = read_range(replay, 0, count);
  }

  std::sort(expected.begin(), expected.end(), compare);
  check_values(actual, expected);
}

template <class Key, class Value, class Compare, class Sort>
void check_pair_sort(
  const thrust::device_vector<Key>& keys,
  const thrust::device_vector<Value>& values,
  const thrust::device_vector<Key>& input_keys,
  const thrust::device_vector<Value>& input_values,
  primbench::stream_t stream,
  Compare compare,
  Sort sort)
{
  const auto count           = std::min(keys.size(), validation_items);
  const auto expected_keys   = read_range(input_keys, 0, count);
  const auto expected_values = read_range(input_values, 0, count);
  auto actual_keys           = read_range(keys, 0, count);
  PRIMBENCH_ASSERT(std::is_sorted(actual_keys.begin(), actual_keys.end(), compare), true);

  thrust::host_vector<Value> actual_values;
  if (count == keys.size())
  {
    actual_values = read_range(values, 0, count);
  }
  else
  {
    thrust::device_vector<Key> replay_keys     = expected_keys;
    thrust::device_vector<Value> replay_values = expected_values;
    sort(replay_keys, replay_values);
    HIP_CHECK(hipStreamSynchronize(stream));
    actual_keys   = read_range(replay_keys, 0, count);
    actual_values = read_range(replay_values, 0, count);
    PRIMBENCH_ASSERT(std::is_sorted(actual_keys.begin(), actual_keys.end(), compare), true);
  }

  std::vector<std::pair<Key, Value>> expected;
  std::vector<std::pair<Key, Value>> actual;
  expected.reserve(count);
  actual.reserve(count);
  for (size_t i = 0; i < count; ++i)
  {
    expected.emplace_back(expected_keys[i], expected_values[i]);
    actual.emplace_back(actual_keys[i], actual_values[i]);
  }

  const auto compare_pairs = [&](const auto& lhs, const auto& rhs) {
    if (compare(lhs.first, rhs.first))
    {
      return true;
    }
    return !compare(rhs.first, lhs.first) && lhs.second < rhs.second;
  };
  std::sort(expected.begin(), expected.end(), compare_pairs);
  std::sort(actual.begin(), actual.end(), compare_pairs);
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
  PRIMBENCH_ASSERT(static_cast<size_t>(mismatch.first - actual.begin()), expected.size());
}

template <class Key>
thrust::host_vector<Key> set_input(size_t left, size_t right)
{
  thrust::host_vector<Key> keys(left + right);
  for (size_t i = 0; i < left; ++i)
  {
    keys[i] = Key(i / 3);
  }
  for (size_t i = 0; i < right; ++i)
  {
    keys[left + i] = Key(2 + i / 2);
  }
  return keys;
}

template <class Key, class Result, class Search, class Reference>
void check_vectorized_search(
  const thrust::device_vector<Key>& data,
  size_t haystack_size,
  const thrust::device_vector<Result>& output,
  primbench::stream_t stream,
  Search search,
  Reference reference)
{
  if (output.empty())
  {
    return;
  }

  thrust::host_vector<Key> input;
  thrust::host_vector<Result> actual;
  if (haystack_size <= validation_items)
  {
    const auto needles = std::min<size_t>(output.size(), 256);
    input              = read_range(data, 0, haystack_size + needles);
    actual             = read_range(output, 0, needles);
  }
  else
  {
    constexpr std::array<int, 9> needles{-1, 0, 1, 2, 17, 31, 62, 63, 64};
    haystack_size = 128;
    input.resize(haystack_size + needles.size());
    for (size_t i = 0; i < haystack_size; ++i)
    {
      input[i] = Key(2 * (i / 4));
    }
    for (size_t i = 0; i < needles.size(); ++i)
    {
      input[haystack_size + i] = Key(needles[i]);
    }

    thrust::device_vector<Key> d_input = input;
    thrust::device_vector<Result> result(needles.size());
    search(d_input, haystack_size, result);
    HIP_CHECK(hipStreamSynchronize(stream));
    actual = read_range(result, 0, result.size());
  }

  const auto first  = input.cbegin();
  const auto middle = first + haystack_size;
  thrust::host_vector<Result> expected(actual.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = reference(first, middle, middle[i]);
  }
  check_values(actual, expected);
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
