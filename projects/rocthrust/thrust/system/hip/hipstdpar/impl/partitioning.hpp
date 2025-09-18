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

/*! \file thrust/system/hip/hipstdpar/impl/partitioning.hpp
 *  \brief <tt>Partitioning operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#  include <thrust/execution_policy.h>
#  include <thrust/partition.h>

#  include <algorithm>
#  include <execution>
#  include <utility>

#  include "hipstd.hpp"

namespace std
{
// BEGIN IS_PARTITIONED
template <typename I,
          typename P,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline bool is_partitioned(execution::parallel_unsequenced_policy, I f, I l, P p)
{
  return ::thrust::is_partitioned(::thrust::device, f, l, ::std::move(p));
}

template <typename I,
          typename P,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline bool is_partitioned(execution::parallel_unsequenced_policy, I f, I l, P p)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<P>())
  {
    ::hipstd::unsupported_callable_type<P>();
  }

  return ::std::is_partitioned(::std::execution::par, f, l, ::std::move(p));
}
// END IS_PARTITIONED

// BEGIN PARTITION
template <typename I,
          typename P,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline I partition(execution::parallel_unsequenced_policy, I f, I l, P p)
{
  return ::thrust::partition(::thrust::device, f, l, ::std::move(p));
}

template <typename I,
          typename P,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline I partition(execution::parallel_unsequenced_policy, I f, I l, P p)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<P>())
  {
    ::hipstd::unsupported_callable_type<P>();
  }

  return ::std::partition(::std::execution::par, f, l, ::std::move(p));
}
// END PARTITION

// BEGIN PARTITION_COPY
template <
  typename I,
  typename O0,
  typename O1,
  typename P,
  enable_if_t<::hipstd::is_offloadable_iterator<I, O0, O1>() && ::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline pair<O0, O1> partition_copy(execution::parallel_unsequenced_policy, I f, I l, O0 fo0, O1 fo1, P p)
{
  auto [r0, r1] = ::thrust::partition_copy(::thrust::device, f, l, fo0, fo1, ::std::move(p));

  return {::std::move(r0), ::std::move(r1)};
}

template <
  typename I,
  typename O0,
  typename O1,
  typename P,
  enable_if_t<!::hipstd::is_offloadable_iterator<I, O0, O1>() || !::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline pair<O0, O1> partition_copy(execution::parallel_unsequenced_policy, I f, I l, O0 fo0, O1 fo1, P p)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I, O0, O1>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category,
                                            typename iterator_traits<O0>::iterator_category,
                                            typename iterator_traits<O1>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<P>())
  {
    ::hipstd::unsupported_callable_type<P>();
  }

  return ::std::partition_copy(::std::execution::par, f, l, fo0, fo1, ::std::move(p));
}
// END PARTITION_COPY

// BEGIN STABLE_PARTITION
template <typename I,
          typename P,
          enable_if_t<::hipstd::is_offloadable_iterator<I>() && ::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline I stable_partition(execution::parallel_unsequenced_policy, I f, I l, P p)
{
  return ::thrust::stable_partition(::thrust::device, f, l, ::std::move(p));
}

template <typename I,
          typename P,
          enable_if_t<!::hipstd::is_offloadable_iterator<I>() || !::hipstd::is_offloadable_callable<P>()>* = nullptr>
inline I stable_partition(execution::parallel_unsequenced_policy, I f, I l, P p)
{
  if constexpr (!::hipstd::is_offloadable_iterator<I>())
  {
    ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();
  }
  if constexpr (!::hipstd::is_offloadable_callable<P>())
  {
    ::hipstd::unsupported_callable_type<P>();
  }

  return ::std::stable_partition(::std::execution::par, f, l, ::std::move(p));
}
// END STABLE_PARTITION
} // namespace std
#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
