/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
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

#include _THRUST_STD_INCLUDE(iterator)

THRUST_NAMESPACE_BEGIN

//! deprecated [since 10.1]
template <typename InputIterator, typename Distance>
CCCL_DEPRECATED_BECAUSE("Use ::hip::std::advance instead")
_LIBCUDACXX_HIDE_FROM_ABI constexpr void advance(InputIterator& i, Distance n)
{
  _THRUST_STD::advance(i, n);
}

//! deprecated [since 10.1]
template <typename InputIterator>
CCCL_DEPRECATED_BECAUSE("Use ::hip::std::next instead")
_LIBCUDACXX_HIDE_FROM_ABI constexpr InputIterator
  next(InputIterator i, typename _THRUST_STD::iterator_traits<InputIterator>::difference_type n = 1)
{
  return _THRUST_STD::next(i, n);
}

//! deprecated [since 10.1]
template <typename InputIterator>
CCCL_DEPRECATED_BECAUSE("Use ::hip::std::prev instead")
_LIBCUDACXX_HIDE_FROM_ABI constexpr InputIterator
  prev(InputIterator i, typename _THRUST_STD::iterator_traits<InputIterator>::difference_type n = 1)
{
  return _THRUST_STD::prev(i, n);
}

THRUST_NAMESPACE_END
