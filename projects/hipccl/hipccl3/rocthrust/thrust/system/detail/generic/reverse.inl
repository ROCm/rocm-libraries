// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/swap.h>
#include <thrust/system/detail/generic/reverse.h>

#include _THRUST_STD_INCLUDE(__iterator/advance.h)
#include _THRUST_STD_INCLUDE(__iterator/distance.h)
#include _THRUST_STD_INCLUDE(__iterator/reverse_iterator.h)

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename ExecutionPolicy, typename BidirectionalIterator>
THRUST_HOST_DEVICE void
reverse(thrust::execution_policy<ExecutionPolicy>& exec, BidirectionalIterator first, BidirectionalIterator last)
{
  using difference_type = thrust::detail::it_difference_t<BidirectionalIterator>;

  // find the midpoint of [first,last)
  difference_type N = _THRUST_STD::distance(first, last);
  BidirectionalIterator mid(first);
  _THRUST_STD::advance(mid, N / 2);

  // swap elements of [first,mid) with [last - 1, mid)
  thrust::swap_ranges(exec, first, mid, _THRUST_STD::make_reverse_iterator(last));
} // end reverse()

template <typename ExecutionPolicy, typename BidirectionalIterator, typename OutputIterator>
THRUST_HOST_DEVICE OutputIterator reverse_copy(
  thrust::execution_policy<ExecutionPolicy>& exec,
  BidirectionalIterator first,
  BidirectionalIterator last,
  OutputIterator result)
{
  return thrust::copy(exec, _THRUST_STD::reverse_iterator{last}, _THRUST_STD::reverse_iterator{first}, result);
} // end reverse_copy()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
