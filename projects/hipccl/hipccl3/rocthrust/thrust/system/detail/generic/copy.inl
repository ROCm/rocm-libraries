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
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/transform.h>

#include _THRUST_STD_INCLUDE(tuple)

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
THRUST_HOST_DEVICE OutputIterator
copy(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, InputIterator last, OutputIterator result)
{
  return thrust::transform(exec, first, last, result, _THRUST_STD::identity{});
} // end copy()

template <typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
THRUST_HOST_DEVICE OutputIterator
copy_n(thrust::execution_policy<DerivedPolicy>& exec, InputIterator first, Size n, OutputIterator result)
{
  using xfrm_type = _THRUST_STD::identity;

  using functor_type = thrust::detail::unary_transform_functor<xfrm_type>;

  using iterator_tuple = _THRUST_STD::tuple<InputIterator, OutputIterator>;
  using zip_iter       = thrust::zip_iterator<iterator_tuple>;

  zip_iter zipped = thrust::make_zip_iterator(first, result);

  return _THRUST_STD::get<1>(thrust::for_each_n(exec, zipped, n, functor_type{xfrm_type()}).get_iterator_tuple());
} // end copy_n()
} // namespace system::detail::generic
THRUST_NAMESPACE_END
