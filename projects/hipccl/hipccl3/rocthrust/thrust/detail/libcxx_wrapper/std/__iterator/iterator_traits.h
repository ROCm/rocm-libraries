// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__ITERATOR_ITERATOR_TRAITS_H
#define LIBCXX_WRAPPER_STD__ITERATOR_ITERATOR_TRAITS_H

// TODO(libhipcxx): remove this file and replace THRUST_NS_QUALIFIER::detail::is* with _THRUST_STD::__is*
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/config/namespace.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__iterator/iterator_traits.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>

#  if !THRUST_COMPILER(NVRTC)
#    if THRUST_COMPILER(MSVC)
#      include <xutility> // for ::std::input_iterator_tag
#    else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv !THRUST_COMPILER(MSVC) vvv
#      include <iterator> // for ::std::input_iterator_tag
#    endif // !THRUST_COMPILER(MSVC)
#  endif // !THRUST_COMPILER(NVRTC)

#  include <type_traits>
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename Tp>
using is_cpp17_input_iterator = _THRUST_STD::__is_cpp17_input_iterator<Tp>;
template <typename Tp>
using is_cpp17_random_access_iterator = _THRUST_STD::__is_cpp17_random_access_iterator<Tp>;

#else

namespace detail
{
using input_iterator_tag         = ::std::input_iterator_tag;
using random_access_iterator_tag = ::std::random_access_iterator_tag;

template <typename Tp>
struct has_iterator_category
{
private:
  template <typename Up>
  inline THRUST_HOST_DEVICE static ::std::false_type test(...);
  template <typename Up>
  inline THRUST_HOST_DEVICE static ::std::true_type test(typename Up::iterator_category* = nullptr);

public:
  static const bool value = decltype(test<Tp>(nullptr))::value;
};

template <typename Tp, typename Up, bool = has_iterator_category<::std::iterator_traits<Tp>>::value>
struct has_iterator_category_convertible_to
    : ::std::is_convertible<typename ::std::iterator_traits<Tp>::iterator_category, Up>
{};

template <typename Tp, typename Up>
struct has_iterator_category_convertible_to<Tp, Up, false> : ::std::false_type
{};
} // namespace detail

template <typename Tp>
struct is_cpp17_input_iterator : public detail::has_iterator_category_convertible_to<Tp, detail::input_iterator_tag>
{};

template <typename Tp>
struct is_cpp17_random_access_iterator
    : public detail::has_iterator_category_convertible_to<Tp, detail::random_access_iterator_tag>
{};

#endif

} // namespace internal

#endif // LIBCXX_WRAPPER_STD__ITERATOR_ITERATOR_TRAITS_H
