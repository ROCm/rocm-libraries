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

/*! \file thrust/system/hip/hipstdpar/impl/heap.hpp
 *  \brief <tt>Heap operations</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#  include <rocprim/rocprim.hpp>

#  include <thrust/execution_policy.h>
#  include <thrust/find.h>
#  include <thrust/functional.h>
#  include <thrust/iterator/counting_iterator.h>
#  include <thrust/logical.h>

#  include <algorithm>
#  include <execution>
#  include <utility>

#  include "hipstd.hpp"

// rocThrust (and rocPRIM) currently expose no heap primitives at all, so is_heap /
// is_heap_until are composed here out of the all_of / find_if primitives that rocThrust
// does provide, by checking the usual "no child compares greater than its parent" invariant
// for every index in the range.
namespace thrust
{
// BEGIN IS_HEAP
template <typename RandomIt, typename CompareOp>
struct __heap_violation
{
  RandomIt first;
  CompareOp compare_op;

  using difference_type = typename thrust::iterator_difference<RandomIt>::type;

  THRUST_HIP_FUNCTION bool operator()(difference_type i) const
  {
    return compare_op(first[(i - 1) / 2], first[i]);
  }
};

template <typename RandomIt,
          typename CompareOp,
          std::enable_if_t<hipstd::is_offloadable_iterator<RandomIt>() && hipstd::is_offloadable_callable<CompareOp>()>* =
            nullptr>
inline bool __is_heap(thrust::hip_rocprim::par_t policy, RandomIt first, RandomIt last, CompareOp compare_op)
{
  using difference_type = typename thrust::iterator_difference<RandomIt>::type;

  const difference_type count = thrust::distance(first, last);
  if (count < 2)
  {
    return true;
  }

  return thrust::none_of(
    policy,
    thrust::counting_iterator<difference_type>(1),
    thrust::counting_iterator<difference_type>(count),
    __heap_violation<RandomIt, CompareOp>{first, compare_op});
}
// END IS_HEAP

// BEGIN IS_HEAP_UNTIL
template <typename RandomIt,
          typename CompareOp,
          std::enable_if_t<hipstd::is_offloadable_iterator<RandomIt>() && hipstd::is_offloadable_callable<CompareOp>()>* =
            nullptr>
inline RandomIt __is_heap_until(thrust::hip_rocprim::par_t policy, RandomIt first, RandomIt last, CompareOp compare_op)
{
  using difference_type = typename thrust::iterator_difference<RandomIt>::type;

  const difference_type count = thrust::distance(first, last);
  if (count < 2)
  {
    return last;
  }

  const auto violation = thrust::find_if(
    policy,
    thrust::counting_iterator<difference_type>(1),
    thrust::counting_iterator<difference_type>(count),
    __heap_violation<RandomIt, CompareOp>{first, compare_op});

  return first + *violation;
}
// END IS_HEAP_UNTIL
} // namespace thrust

namespace std
{
// BEGIN IS_HEAP
template <
  typename RandomIt,
  typename CompareOp,
  enable_if_t<!hipstd::is_offloadable_iterator<RandomIt>() || !hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline bool is_heap(execution::parallel_unsequenced_policy, RandomIt first, RandomIt last, CompareOp compare_op)
{
  if constexpr (!hipstd::is_offloadable_iterator<RandomIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<RandomIt>::iterator_category>();
  }
  if constexpr (!hipstd::is_offloadable_callable<CompareOp>())
  {
    hipstd::unsupported_callable_type<CompareOp>();
  }

  return std::is_heap(std::execution::par, first, last, std::move(compare_op));
}

template <typename RandomIt,
          typename CompareOp,
          enable_if_t<hipstd::is_offloadable_iterator<RandomIt>() && hipstd::is_offloadable_callable<CompareOp>()>* =
            nullptr>
inline bool is_heap(execution::parallel_unsequenced_policy, RandomIt first, RandomIt last, CompareOp compare_op)
{
  ::hipstd::__maybe_bind_globals();

  ::hipstd::warn_if_no_xnack();
  return ::thrust::__is_heap(::thrust::device, first, last, compare_op);
}

template <typename RandomIt, enable_if_t<!hipstd::is_offloadable_iterator<RandomIt>()>* = nullptr>
inline bool is_heap(execution::parallel_unsequenced_policy, RandomIt first, RandomIt last)
{
  if constexpr (!hipstd::is_offloadable_iterator<RandomIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<RandomIt>::iterator_category>();
  }

  return std::is_heap(std::execution::par, first, last);
}

template <typename RandomIt, enable_if_t<hipstd::is_offloadable_iterator<RandomIt>()>* = nullptr>
inline bool is_heap(execution::parallel_unsequenced_policy policy, RandomIt first, RandomIt last)
{
  ::hipstd::warn_if_no_xnack();
  using item_type = typename thrust::iterator_value<RandomIt>::type;
  return std::is_heap(policy, first, last, thrust::less<item_type>());
}
// END IS_HEAP

// BEGIN IS_HEAP_UNTIL
template <
  typename RandomIt,
  typename CompareOp,
  enable_if_t<!hipstd::is_offloadable_iterator<RandomIt>() || !hipstd::is_offloadable_callable<CompareOp>()>* = nullptr>
inline RandomIt is_heap_until(execution::parallel_unsequenced_policy, RandomIt first, RandomIt last, CompareOp compare_op)
{
  if constexpr (!hipstd::is_offloadable_iterator<RandomIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<RandomIt>::iterator_category>();
  }
  if constexpr (!hipstd::is_offloadable_callable<CompareOp>())
  {
    hipstd::unsupported_callable_type<CompareOp>();
  }

  return std::is_heap_until(std::execution::par, first, last, std::move(compare_op));
}

template <typename RandomIt,
          typename CompareOp,
          enable_if_t<hipstd::is_offloadable_iterator<RandomIt>() && hipstd::is_offloadable_callable<CompareOp>()>* =
            nullptr>
inline RandomIt is_heap_until(execution::parallel_unsequenced_policy, RandomIt first, RandomIt last, CompareOp compare_op)
{
  ::hipstd::__maybe_bind_globals();

  ::hipstd::warn_if_no_xnack();
  return ::thrust::__is_heap_until(::thrust::device, first, last, compare_op);
}

template <typename RandomIt, enable_if_t<!hipstd::is_offloadable_iterator<RandomIt>()>* = nullptr>
inline RandomIt is_heap_until(execution::parallel_unsequenced_policy, RandomIt first, RandomIt last)
{
  if constexpr (!hipstd::is_offloadable_iterator<RandomIt>())
  {
    hipstd::unsupported_iterator_category<typename iterator_traits<RandomIt>::iterator_category>();
  }

  return std::is_heap_until(std::execution::par, first, last);
}

template <typename RandomIt, enable_if_t<hipstd::is_offloadable_iterator<RandomIt>()>* = nullptr>
inline RandomIt is_heap_until(execution::parallel_unsequenced_policy policy, RandomIt first, RandomIt last)
{
  ::hipstd::warn_if_no_xnack();
  using item_type = typename thrust::iterator_value<RandomIt>::type;
  return std::is_heap_until(policy, first, last, thrust::less<item_type>());
}
// END IS_HEAP_UNTIL
} // namespace std
#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
