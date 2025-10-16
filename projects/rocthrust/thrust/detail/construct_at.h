// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef DETAIL_CONSTRUCT_AT_H
#define DETAIL_CONSTRUCT_AT_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
// clang-format off
#  include _THRUST_STD_INCLUDE(__memory/construct_at.h)
// clang-format on
#else
#  include <thrust/detail/memory_wrapper.h>

#  include <iterator>
#  include <type_traits>
#  include <utility>
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_STD::__construct_at;
using _THRUST_STD::__destroy_at;

#else

namespace detail
{

template <typename To, typename...>
struct is_narrowing_impl : ::std::false_type
{};

template <typename To, typename From>
struct is_narrowing_impl<To, From> : ::std::true_type
{};

template <typename To, typename From>
struct is_narrowing_impl<To, From, ::std::void_t<decltype(To{::std::declval<From>()})>> : ::std::false_type
{};

template <typename Tp, typename... Args>
using is_narrowing =
  ::std::conditional_t<THRUST_TRAIT(::std::is_arithmetic, Tp), is_narrowing_impl<Tp, Args...>, ::std::false_type>;

template <typename Tp, typename... Args>
struct can_optimize_construct_at
    : ::std::integral_constant<
        bool,
        THRUST_TRAIT(::std::is_trivially_constructible, Tp, Args...)
          && THRUST_TRAIT(::std::is_trivially_move_assignable, Tp) && !is_narrowing<Tp, Args...>::value>
{};

} // namespace detail

THRUST_EXEC_CHECK_DISABLE
template <typename Tp, typename... Args>
inline THRUST_HOST_DEVICE
  THRUST_CONSTEXPR_CXX20 ::std::enable_if_t<!detail::can_optimize_construct_at<Tp, Args...>::value, Tp*>
  __construct_at(Tp* location, Args&&... args)
{
  assert(location != nullptr && "null pointer given to construct_at");
#  if THRUST_CPP_DIALECT >= 2020
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (::std::is_constant_evaluated())
  {
    return ::std::construct_at(location, ::std::forward<Args>(args)...);
  }
#  endif // THRUST_CPP_DIALECT >= 2020
  return ::new (::std::addressof(*location)) Tp(::std::forward<Args>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Tp, typename... Args>
inline THRUST_HOST_DEVICE
  THRUST_CONSTEXPR_CXX20 ::std::enable_if_t<detail::can_optimize_construct_at<Tp, Args...>::value, Tp*>
  __construct_at(Tp* location, Args&&... args)
{
  assert(location != nullptr && "null pointer given to construct_at");
#  if THRUST_CPP_DIALECT >= 2020
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (::std::is_constant_evaluated())
  {
    return ::std::construct_at(location, ::std::forward<Args>(args)...);
  }
#  endif // THRUST_CPP_DIALECT >= 2020
  *location = Tp{::std::forward<Args>(args)...};
  return location;
}

template <typename ForwardIterator>
inline THRUST_HOST_DEVICE constexpr ForwardIterator __destroy(ForwardIterator, ForwardIterator);

THRUST_EXEC_CHECK_DISABLE
template <typename Tp,
          ::std::enable_if_t<!THRUST_TRAIT(::std::is_array, Tp), int>                  = 0,
          ::std::enable_if_t<!THRUST_TRAIT(::std::is_trivially_destructible, Tp), int> = 0>
inline THRUST_HOST_DEVICE constexpr void __destroy_at(Tp* loc)
{
  assert(loc != nullptr && "null pointer given to destroy_at");
  loc->~Tp();
}

THRUST_EXEC_CHECK_DISABLE
template <typename Tp,
          ::std::enable_if_t<!THRUST_TRAIT(::std::is_array, Tp), int>                 = 0,
          ::std::enable_if_t<THRUST_TRAIT(::std::is_trivially_destructible, Tp), int> = 0>
inline THRUST_HOST_DEVICE constexpr void __destroy_at(Tp* loc)
{
  assert(loc != nullptr && "null pointer given to destroy_at");
  (void) loc;
}

template <typename Tp, ::std::enable_if_t<THRUST_TRAIT(::std::is_array, Tp), int> = 0>
inline THRUST_HOST_DEVICE constexpr void __destroy_at(Tp* loc)
{
  assert(loc != nullptr && "null pointer given to destroy_at");
  __destroy(::std::begin(*loc), ::std::end(*loc));
}

template <typename ForwardIterator>
inline THRUST_HOST_DEVICE constexpr ForwardIterator __destroy(ForwardIterator first, ForwardIterator last)
{
  for (; first != last; ++first)
  {
    __destroy_at(::std::addressof(*first));
  }
  return first;
}

#endif

} // namespace internal

#endif // DETAIL_CONSTRUCT_AT_H
