// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef ITERATOR_DETAIL_ITERATOR_TRAITS_H
#define ITERATOR_DETAIL_ITERATOR_TRAITS_H

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _THRUST_HAS_DEVICE_SYSTEM_STD
// clang-format off
#  include _THRUST_STD_INCLUDE(__iterator/iterator_traits.h)
// clang-format on
#else
#  include <type_traits>
#endif

#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_NVRTC
#  if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#    include <xutility> // for ::std::input_iterator_tag
#  else // ^^^ THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC ^^^ / vvv THRUST_HOST_COMPILER !=
        // THRUST_HOST_COMPILER_MSVC vvv
#    include <iterator> // for ::std::input_iterator_tag
#  endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
#endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_NVRTC

THRUST_NAMESPACE_BEGIN

namespace detail
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename T>
using is_cpp17_input_iterator = _THRUST_STD::__is_cpp17_input_iterator<T>;
template <typename T>
using is_cpp17_random_access_iterator = _THRUST_STD::__is_cpp17_random_access_iterator<T>;

#else

using input_iterator_tag         = ::std::input_iterator_tag;
using random_access_iterator_tag = ::std::random_access_iterator_tag;

template <typename T>
struct has_iterator_category
{
private:
  template <typename U>
  inline THRUST_HOST_DEVICE static ::std::false_type test(...);
  template <typename U>
  inline THRUST_HOST_DEVICE static ::std::true_type test(typename U::iterator_category* = nullptr);

public:
  static const bool value = decltype(test<T>(nullptr))::value;
};

template <typename T, typename U, bool = has_iterator_category<::std::iterator_traits<T>>::value>
struct has_iterator_category_convertible_to
    : ::std::is_convertible<typename ::std::iterator_traits<T>::iterator_category, U>
{};

template <typename T, typename U>
struct has_iterator_category_convertible_to<T, U, false> : ::std::false_type
{};

template <typename T>
struct is_cpp17_input_iterator : public has_iterator_category_convertible_to<T, input_iterator_tag>
{};

template <typename T>
struct is_cpp17_random_access_iterator : public has_iterator_category_convertible_to<T, random_access_iterator_tag>
{};

#endif

} // namespace detail

THRUST_NAMESPACE_END

#endif // ITERATOR_DETAIL_ITERATOR_TRAITS_H
