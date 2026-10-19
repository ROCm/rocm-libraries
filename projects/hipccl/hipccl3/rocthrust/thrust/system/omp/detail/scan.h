// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// OMP parallel scan implementation
#include <thrust/advance.h>
#include <thrust/detail/function.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/detail/pragma_omp.h>

#include _THRUST_STD_INCLUDE(algorithm)
#include _THRUST_STD_INCLUDE(iterator)
#include _THRUST_STD_INCLUDE(numeric)
#include _THRUST_STD_INCLUDE(type_traits)

#include <omp.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{

struct __no_init_tag
{};

// Threshold below which serial scan is faster than parallel
// Benchmarking shows parallel overhead dominates for small arrays
inline constexpr size_t parallel_scan_threshold = 1024;

template <bool IsInclusive,
          typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator scan_impl(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  [[maybe_unused]] InitialValueType init,
  BinaryFunction binary_op)
{
  using namespace thrust::detail;

  static constexpr bool has_init = !_THRUST_STD::is_same_v<InitialValueType, __no_init_tag>;
  // Use logic from https://wg21.link/P0571
  using accum_t = _THRUST_STD::conditional_t<has_init, InitialValueType, it_value_t<InputIterator>>;
  using Size    = it_difference_t<InputIterator>;

  const Size n = _THRUST_STD::distance(first, last);

  if (n == 0)
  {
    return result;
  }

  auto wrapped_binary_op = wrapped_function<BinaryFunction, accum_t>{binary_op};

  const int num_threads = omp_get_max_threads();

  // Use serial scan for small arrays where parallel overhead dominates
  if (static_cast<size_t>(n) < parallel_scan_threshold || num_threads <= 1)
  {
    if constexpr (IsInclusive)
    {
      if constexpr (has_init)
      {
        return _THRUST_STD::inclusive_scan(first, last, result, wrapped_binary_op, init);
      }
      else
      {
        return _THRUST_STD::inclusive_scan(first, last, result, wrapped_binary_op);
      }
    }
    else
    {
      return _THRUST_STD::exclusive_scan(first, last, result, init, wrapped_binary_op);
    }
  }

  temporary_array<accum_t, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1: Reduce each block (N reads)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = (n + num_threads - 1) / num_threads;
    const Size start      = tid * block_size;
    const Size end        = _THRUST_STD::min(start + block_size, n);

    if (start < n)
    {
      block_sums[tid] = _THRUST_STD::reduce(first + start, first + end, accum_t{}, wrapped_binary_op);
    }
  }

  // Step 2: Scan block sums
  if constexpr (has_init)
  {
    _THRUST_STD::exclusive_scan(block_sums.begin(), block_sums.end(), block_sums.begin(), init, wrapped_binary_op);
  }
  else
  {
    _THRUST_STD::exclusive_scan(block_sums.begin(), block_sums.end(), block_sums.begin(), accum_t{}, wrapped_binary_op);
  }

  // Step 3: Scan each block with offset (N reads/writes)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = (n + num_threads - 1) / num_threads;
    const Size start      = tid * block_size;
    const Size end        = _THRUST_STD::min(start + block_size, n);

    if (start < n)
    {
      const accum_t prefix = block_sums[tid];
      if constexpr (IsInclusive)
      {
        _THRUST_STD::inclusive_scan(first + start, first + end, result + start, wrapped_binary_op, prefix);
      }
      else
      {
        _THRUST_STD::exclusive_scan(first + start, first + end, result + start, prefix, wrapped_binary_op);
      }
    }
  }

  return result + n;
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  return inclusive_scan(exec, first, last, result, __no_init_tag{}, binary_op);
}

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  return scan_impl<true>(exec, first, last, result, init, binary_op);
}

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator exclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  return scan_impl<false>(exec, first, last, result, init, binary_op);
}

} // namespace system::omp::detail
THRUST_NAMESPACE_END
