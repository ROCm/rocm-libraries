// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file advance.h
 *  \brief Advance an iterator by a given distance.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include _THRUST_STD_INCLUDE(__iterator/advance.h)
#include _THRUST_STD_INCLUDE(__iterator/iterator_traits.h)
#include _THRUST_STD_INCLUDE(__iterator/next.h)
#include _THRUST_STD_INCLUDE(__iterator/prev.h)

THRUST_NAMESPACE_BEGIN

//! deprecated [since 10.1]
template <typename InputIterator, typename Distance>
CCCL_DEPRECATED_BECAUSE("Use ::hip::std::advance instead")
_CCCL_API constexpr void advance(InputIterator& i, Distance n)
{
  _THRUST_STD::advance(i, n);
}

//! deprecated [since 10.1]
template <typename InputIterator>
CCCL_DEPRECATED_BECAUSE("Use ::hip::std::next instead")
_CCCL_API constexpr InputIterator
  next(InputIterator i, typename _THRUST_STD::iterator_traits<InputIterator>::difference_type n = 1)
{
  return _THRUST_STD::next(i, n);
}

//! deprecated [since 10.1]
template <typename InputIterator>
CCCL_DEPRECATED_BECAUSE("Use ::hip::std::prev instead")
_CCCL_API constexpr InputIterator
  prev(InputIterator i, typename _THRUST_STD::iterator_traits<InputIterator>::difference_type n = 1)
{
  return _THRUST_STD::prev(i, n);
}

THRUST_NAMESPACE_END
