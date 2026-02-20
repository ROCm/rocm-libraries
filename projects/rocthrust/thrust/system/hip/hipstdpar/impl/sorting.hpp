// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/hipstdpar/impl/sorting.hpp
 *  \brief <tt>Sorting operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#  include <rocprim/rocprim.hpp>

#  include <thrust/execution_policy.h>
#  include <thrust/sort.h>

#  include <algorithm>
#  include <execution>
#  include <utility>

#  include "hipstd.hpp"

// rocThrust is currently missing some API entries, forward calls to rocPRIM until they are added.
namespace thrust
{
// BEGIN PARTIAL_SORT
template <typename KeysIt,
          typename CompareOp,
          std::enable_if_t<hipstd::is_offloadable_iterator<KeysIt>() && hipstd::is_offloadable_callable<CompareOp>()>* =
            nullptr>
inline void
__partial_sort(thrust::hip_rocprim::par_t policy, KeysIt first, KeysIt middle, KeysIt last, CompareOp compare_op)
{
  const size_t count = static_cast<size_t>(thrust::distance(first, last));
  const size_t n     = static_cast<size_t>(thrust::distance(first, middle));

  if (count == 0 || n == 0)
  {
    return;
  }

  const size_t n_index = n - 1;

  size_t storage_size = 0;
  hipStream_t stream  = thrust::hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  hipError_t status;

  status = rocprim::partial_sort(nullptr, storage_size, first, n_index, count, compare_op, stream, debug_sync);
  thrust::hip_rocprim::throw_on_error(status, "partial_sort: failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, decltype(policy)> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  status = rocprim::partial_sort(ptr, storage_size, first, n_index, count, compare_op, stream, debug_sync);
  thrust::hip_rocprim::throw_on_error(status, "partial_sort: failed on 2nd step");
  thrust::hip_rocprim::throw_on_error(
    thrust::hip_rocprim::synchronize_optional(policy), "partial_sort: failed to synchronize");
}
// END PARTIAL_SORT

// BEGIN PARTIAL_SORT_COPY
template <typename ForwardIt,
          typename RandomIt,
          typename CompareOp,
          std::enable_if_t<hipstd::is_offloadable_iterator<ForwardIt, RandomIt>()
                           && hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void __partial_sort_copy(
  thrust::hip_rocprim::par_t policy,
  ForwardIt first,
  ForwardIt last,
  RandomIt d_first,
  RandomIt d_last,
  CompareOp compare_op)
{
  const size_t count   = static_cast<size_t>(thrust::distance(first, last));
  const size_t d_count = static_cast<size_t>(thrust::distance(d_first, d_last));

  if (count == 0 || d_count == 0)
  {
    return;
  }

  const size_t d_index = d_count - 1;

  size_t storage_size = 0;
  hipStream_t stream  = thrust::hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  hipError_t status;

  status =
    rocprim::partial_sort_copy(nullptr, storage_size, first, d_first, d_index, count, compare_op, stream, debug_sync);
  thrust::hip_rocprim::throw_on_error(status, "partial_sort_copy: failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, decltype(policy)> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  status =
    rocprim::partial_sort_copy(ptr, storage_size, first, d_first, d_index, count, compare_op, stream, debug_sync);
  thrust::hip_rocprim::throw_on_error(status, "partial_sort_copy: failed on 2nd step");
  thrust::hip_rocprim::throw_on_error(
    thrust::hip_rocprim::synchronize_optional(policy), "partial_sort_copy: failed to synchronize");
}
// END PARTIAL_SORT_COPY

// BEGIN NTH_ELEMENT
template <typename KeysIt,
          typename CompareOp,
          std::enable_if_t<hipstd::is_offloadable_iterator<KeysIt>() && hipstd::is_offloadable_callable<CompareOp>()>* =
            nullptr>
inline void __nth_element(thrust::hip_rocprim::par_t policy, KeysIt first, KeysIt nth, KeysIt last, CompareOp compare_op)
{
  const size_t count = static_cast<size_t>(thrust::distance(first, last));
  const size_t n     = static_cast<size_t>(thrust::distance(first, nth));

  if (count == 0)
  {
    return;
  }

  size_t storage_size = 0;
  hipStream_t stream  = thrust::hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  hipError_t status;

  status = rocprim::nth_element(nullptr, storage_size, first, n, count, compare_op, stream, debug_sync);
  thrust::hip_rocprim::throw_on_error(status, "nth_element: failed on 1st step");
  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, decltype(policy)> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  status = rocprim::nth_element(ptr, storage_size, first, n, count, compare_op, stream, debug_sync);
  thrust::hip_rocprim::throw_on_error(status, "nth_element: failed on 2nd step");
  thrust::hip_rocprim::throw_on_error(
    thrust::hip_rocprim::synchronize_optional(policy), "nth_element: failed to synchronize");
}
// END NTH_ELEMENT
} // namespace thrust

namespace std
{
// BEGIN SORT
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void sort(execution::parallel_unsequenced_policy, I f, I l)
{
  return ::thrust::sort(::thrust::device, f, l);
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void sort(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::sort(::std::execution::par, f, l);
}

template <typename I,
          typename R,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline void sort(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  return ::thrust::sort(::thrust::device, f, l, ::std::move(r));
}

template <typename I,
          typename R,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline void sort(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<R>())
  {
    ::hipstd::unsupported_callable_type<R>();
  }

  return ::std::sort(::std::execution::par, f, l, ::std::move(r));
}
// END SORT

// BEGIN STABLE_SORT
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void stable_sort(execution::parallel_unsequenced_policy, I f, I l)
{
  return ::thrust::stable_sort(::thrust::device, f, l);
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void stable_sort(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::stable_sort(::std::execution::par, f, l);
}

template <typename I,
          typename R,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline void stable_sort(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  return ::thrust::stable_sort(::thrust::device, f, l, ::std::move(r));
}

template <typename I,
          typename R,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline void stable_sort(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<R>())
  {
    ::hipstd::unsupported_callable_type<R>();
  }

  return ::std::stable_sort(::std::execution::par, f, l, ::std::move(r));
}
// END STABLE_SORT

// BEGIN PARTIAL_SORT
template <
  typename KeysIt,
  typename CompareOp,
  enable_if_t<!hipstd::is_offloadable_iterator<KeysIt>() || !hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void
partial_sort(execution::parallel_unsequenced_policy, KeysIt first, KeysIt middle, KeysIt last, CompareOp compare_op)
{
  if constexpr (!hipstd::is_offloadable_iterator<KeysIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<KeysIt>::iterator_category>();
  }
  if constexpr (!hipstd::is_offloadable_callable<CompareOp>())
  {
    hipstd::unsupported_callable_type<CompareOp>();
  }

  std::partial_sort(std::execution::par, first, middle, last, std::move(compare_op));
}

template <
  typename KeysIt,
  typename CompareOp,
  enable_if_t<hipstd::is_offloadable_iterator<KeysIt>() && hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void
partial_sort(execution::parallel_unsequenced_policy, KeysIt first, KeysIt middle, KeysIt last, CompareOp compare_op)
{
  ::thrust::__partial_sort(::thrust::device, first, middle, last, compare_op);
}

template <typename KeysIt, typename CompareOp, enable_if_t<!hipstd::is_offloadable_iterator<KeysIt>()>* = nullptr>
inline void partial_sort(execution::parallel_unsequenced_policy, KeysIt first, KeysIt middle, KeysIt last)
{
  if constexpr (!hipstd::is_offloadable_iterator<KeysIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<KeysIt>::iterator_category>();
  }

  std::partial_sort(std::execution::par, first, middle, last);
}

template <typename KeysIt, enable_if_t<hipstd::is_offloadable_iterator<KeysIt>()>* = nullptr>
inline void partial_sort(execution::parallel_unsequenced_policy policy, KeysIt first, KeysIt middle, KeysIt last)
{
  using item_type = typename thrust::iterator_value<KeysIt>::type;
  std::partial_sort(policy, first, middle, last, thrust::less<item_type>());
}
// END PARTIAL_SORT

// BEGIN PARTIAL_SORT_COPY
template <typename ForwardIt,
          typename RandomIt,
          typename CompareOp,
          enable_if_t<!hipstd::is_offloadable_iterator<ForwardIt, RandomIt>()
                      || !hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void partial_sort_copy(
  execution::parallel_unsequenced_policy,
  ForwardIt first,
  ForwardIt last,
  RandomIt d_first,
  RandomIt d_last,
  CompareOp compare_op)
{
  if constexpr (!hipstd::is_offloadable_iterator<ForwardIt, RandomIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<ForwardIt>::iterator_category,
                                          typename iterator_traits<RandomIt>::iterator_category>();
  }
  if constexpr (!hipstd::is_offloadable_callable<CompareOp>())
  {
    hipstd::unsupported_callable_type<CompareOp>();
  }

  std::partial_sort_copy(std::execution::par, first, last, d_first, d_last, std::move(compare_op));
}

template <typename ForwardIt,
          typename RandomIt,
          typename CompareOp,
          enable_if_t<hipstd::is_offloadable_iterator<ForwardIt, RandomIt>()
                      && hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void partial_sort_copy(
  execution::parallel_unsequenced_policy,
  ForwardIt first,
  ForwardIt last,
  RandomIt d_first,
  RandomIt d_last,
  CompareOp compare_op)
{
  ::thrust::__partial_sort_copy(::thrust::device, first, last, d_first, d_last, compare_op);
}

template <typename ForwardIt,
          typename RandomIt,
          enable_if_t<!hipstd::is_offloadable_iterator<ForwardIt, RandomIt>()>* = nullptr>
inline void partial_sort_copy(
  execution::parallel_unsequenced_policy, ForwardIt first, ForwardIt last, RandomIt d_first, RandomIt d_last)
{
  if constexpr (!hipstd::is_offloadable_iterator<ForwardIt, RandomIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<ForwardIt>::iterator_category,
                                          typename iterator_traits<RandomIt>::iterator_category>();
  }

  std::partial_sort_copy(std::execution::par, first, last, d_first, d_last);
}

template <typename ForwardIt,
          typename RandomIt,
          enable_if_t<hipstd::is_offloadable_iterator<ForwardIt, RandomIt>()>* = nullptr>
inline void partial_sort_copy(
  execution::parallel_unsequenced_policy policy, ForwardIt first, ForwardIt last, RandomIt d_first, RandomIt d_last)
{
  using item_type = typename thrust::iterator_value<ForwardIt>::type;
  std::partial_sort_copy(policy, first, last, d_first, d_last, thrust::less<item_type>());
}
// END PARTIAL_SORT_COPY

// BEGIN IS_SORTED
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline bool is_sorted(execution::parallel_unsequenced_policy, I f, I l)
{
  return ::thrust::is_sorted(::thrust::device, f, l);
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline bool is_sorted(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::is_sorted(::std::execution::par, f, l);
}

template <typename I,
          typename R,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline bool is_sorted(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  return ::thrust::is_sorted(::thrust::device, f, l, ::std::move(r));
}

template <typename I,
          typename R,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline bool is_sorted(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<R>())
  {
    ::hipstd::unsupported_callable_type<R>();
  }

  return ::std::is_sorted(::std::execution::par, f, l, ::std::move(r));
}
// END IS_SORTED

// BEGIN IS_SORTED_UNTIL
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline I is_sorted_until(execution::parallel_unsequenced_policy, I f, I l)
{
  return ::thrust::is_sorted_until(::thrust::device, f, l);
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline I is_sorted_until(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::is_sorted_until(::std::execution::par, f, l);
}

template <typename I,
          typename R,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline I is_sorted_until(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  return ::thrust::is_sorted_until(::thrust::device, f, l, ::std::move(r));
}

template <typename I,
          typename R,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<R>()>* = nullptr>
inline I is_sorted_until(execution::parallel_unsequenced_policy, I f, I l, R r)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<R>())
  {
    ::hipstd::unsupported_callable_type<R>();
  }

  return ::std::is_sorted_until(::std::execution::par, f, l, ::std::move(r));
}
// END IS_SORTED_UNTIL

// BEGIN NTH_ELEMENT
template <
  typename KeysIt,
  typename CompareOp,
  enable_if_t<!hipstd::is_offloadable_iterator<KeysIt>() || !hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void
nth_element(execution::parallel_unsequenced_policy, KeysIt first, KeysIt nth, KeysIt last, CompareOp compare_op)
{
  if constexpr (!hipstd::is_offloadable_iterator<KeysIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<KeysIt>::iterator_category>();
  }
  if constexpr (!hipstd::is_offloadable_callable<CompareOp>())
  {
    hipstd::unsupported_callable_type<CompareOp>();
  }

  std::nth_element(std::execution::par, first, nth, last, std::move(compare_op));
}

template <
  typename KeysIt,
  typename CompareOp,
  enable_if_t<hipstd::is_offloadable_iterator<KeysIt>() && hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline void
nth_element(execution::parallel_unsequenced_policy, KeysIt first, KeysIt nth, KeysIt last, CompareOp compare_op)
{
  ::thrust::__nth_element(::thrust::device, first, nth, last, compare_op);
}

template <typename KeysIt, enable_if_t<!hipstd::is_offloadable_iterator<KeysIt>()>* = nullptr>
inline void nth_element(execution::parallel_unsequenced_policy, KeysIt first, KeysIt nth, KeysIt last)
{
  if constexpr (!hipstd::is_offloadable_iterator<KeysIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<KeysIt>::iterator_category>();
  }

  std::nth_element(std::execution::par, first, nth, last);
}

template <typename KeysIt, enable_if_t<hipstd::is_offloadable_iterator<KeysIt>()>* = nullptr>
inline void nth_element(execution::parallel_unsequenced_policy policy, KeysIt first, KeysIt nth, KeysIt last)
{
  using item_type = typename thrust::iterator_value<KeysIt>::type;
  std::nth_element(policy, first, nth, last, thrust::less<item_type>());
}
// END NTH_ELEMENT
} // namespace std
#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
