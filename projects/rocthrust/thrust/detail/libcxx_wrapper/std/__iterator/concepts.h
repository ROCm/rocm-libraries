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

#ifndef LIBCXX_WRAPPER_STD__ITERATOR_CONCEPTS_H
#define LIBCXX_WRAPPER_STD__ITERATOR_CONCEPTS_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/dialect.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__iterator/concepts.h)
#elif THRUST_STD_VER >= 2020
#  include <iterator>
#else
#  include <type_traits>
#  include <utility>
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD || THRUST_STD_VER >= 2020

using _THRUST_STD::indirectly_readable;

#else

namespace detail
{

template <typename, typename = void>
struct is_indirectly_readable : ::std::false_type
{};

template <typename T>
struct is_indirectly_readable<T, ::std::void_t<decltype(*::std::declval<T&>()), typename T::value_type>>
    : ::std::true_type
{};

template <typename T>
struct is_indirectly_readable<T*, void> : ::std::true_type
{};

} // namespace detail

template <typename T>
inline constexpr bool indirectly_readable = detail::is_indirectly_readable<T>::value;

#endif
} // namespace internal

#endif // LIBCXX_WRAPPER_STD__ITERATOR_CONCEPTS_H
