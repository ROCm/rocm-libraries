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
#include <thrust/mismatch.h>
#include <thrust/system/detail/generic/select_system.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/mismatch.h>
#include <thrust/system/detail/sequential/mismatch.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(mismatch.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(mismatch.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/mismatch.h>
#  include <thrust/system/cuda/detail/mismatch.h>
#  include <thrust/system/hip/detail/mismatch.h>
#  include <thrust/system/omp/detail/mismatch.h>
#  include <thrust/system/tbb/detail/mismatch.h>
#endif

THRUST_NAMESPACE_BEGIN

THRUST_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
THRUST_HOST_DEVICE _THRUST_STD::pair<InputIterator1, InputIterator2>
mismatch(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
         InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2)
{
  using thrust::system::detail::generic::mismatch;
  return mismatch(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2);
} // end mismatch()

THRUST_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
THRUST_HOST_DEVICE _THRUST_STD::pair<InputIterator1, InputIterator2> mismatch(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  BinaryPredicate pred)
{
  using thrust::system::detail::generic::mismatch;
  return mismatch(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, pred);
} // end mismatch()

template <typename InputIterator1, typename InputIterator2>
_THRUST_STD::pair<InputIterator1, InputIterator2> mismatch(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::mismatch(select_system(system1, system2), first1, last1, first2);
} // end mismatch()

template <typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
_THRUST_STD::pair<InputIterator1, InputIterator2>
mismatch(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate pred)
{
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;

  System1 system1;
  System2 system2;

  return thrust::mismatch(select_system(system1, system2), first1, last1, first2, pred);
} // end mismatch()

THRUST_NAMESPACE_END
