// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/uninitialized_fill.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/uninitialized_fill.h>
#include <thrust/system/detail/sequential/uninitialized_fill.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(uninitialized_fill.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(uninitialized_fill.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/uninitialized_fill.h>
#  include <thrust/system/cuda/detail/uninitialized_fill.h>
#  include <thrust/system/hip/detail/uninitialized_fill.h>
#  include <thrust/system/omp/detail/uninitialized_fill.h>
#  include <thrust/system/tbb/detail/uninitialized_fill.h>
#endif

THRUST_NAMESPACE_BEGIN

THRUST_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename T>
THRUST_HOST_DEVICE void uninitialized_fill(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const T& x)
{
  using thrust::system::detail::generic::uninitialized_fill;
  return uninitialized_fill(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, x);
} // end uninitialized_fill()

THRUST_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename ForwardIterator, typename Size, typename T>
THRUST_HOST_DEVICE ForwardIterator uninitialized_fill_n(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec, ForwardIterator first, Size n, const T& x)
{
  using thrust::system::detail::generic::uninitialized_fill_n;
  return uninitialized_fill_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, x);
} // end uninitialized_fill_n()

template <typename ForwardIterator, typename T>
void uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x)
{
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  thrust::uninitialized_fill(select_system(system), first, last, x);
} // end uninitialized_fill()

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(ForwardIterator first, Size n, const T& x)
{
  using thrust::system::detail::generic::select_system;

  using System = typename thrust::iterator_system<ForwardIterator>::type;

  System system;

  return thrust::uninitialized_fill_n(select_system(system), first, n, x);
} // end uninitialized_fill_n()

THRUST_NAMESPACE_END
