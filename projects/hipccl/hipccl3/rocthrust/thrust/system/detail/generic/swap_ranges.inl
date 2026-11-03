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
#include <thrust/detail/internal_functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/swap_ranges.h>

#include _THRUST_STD_INCLUDE(__utility/swap.h)
#include _THRUST_STD_INCLUDE(tuple)

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
namespace detail
{
// XXX define this here rather than in internal_functional.h
// to avoid circular dependence between swap.h & internal_functional.h
struct swap_pair_elements
{
  template <typename Tuple>
  THRUST_HOST_DEVICE void operator()(Tuple t)
  {
    // use unqualified swap to allow ADL to catch any user-defined swap
    using _THRUST_STD::swap;
    swap(_THRUST_STD::get<0>(t), _THRUST_STD::get<1>(t));
  }
}; // end swap_pair_elements
} // namespace detail

template <typename DerivedPolicy, typename ForwardIterator1, typename ForwardIterator2>
THRUST_HOST_DEVICE ForwardIterator2 swap_ranges(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator1 first1,
  ForwardIterator1 last1,
  ForwardIterator2 first2)
{
  using IteratorTuple = _THRUST_STD::tuple<ForwardIterator1, ForwardIterator2>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  ZipIterator result = thrust::for_each(
    exec,
    thrust::make_zip_iterator(first1, first2),
    thrust::make_zip_iterator(last1, first2),
    detail::swap_pair_elements());
  return _THRUST_STD::get<1>(result.get_iterator_tuple());
} // end swap_ranges()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
