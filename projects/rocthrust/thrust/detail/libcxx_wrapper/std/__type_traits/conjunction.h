//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__TYPE_TRAITS_CONJUNCTION_H
#define LIBCXX_WRAPPER_STD__TYPE_TRAITS_CONJUNCTION_H

// TODO(libhipcxx): remove this file and replace ::internal* with _THRUST_STD* in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__type_traits/conjunction.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>

#  include <type_traits>
#endif

namespace internal
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename... Pred>
using _And = _THRUST_STD::_And<Pred...>;

#else

namespace detail
{
template <typename...>
using expand_to_true = ::std::true_type;

template <typename... Pred>
THRUST_HOST_DEVICE expand_to_true<::std::enable_if_t<Pred::value>...> and_helper(int);

template <typename...>
THRUST_HOST_DEVICE ::std::false_type and_helper(...);
} // namespace detail

template <typename... Pred>
using _And = decltype(detail::and_helper<Pred...>(0));

#endif
} // namespace internal

#endif // LIBCXX_WRAPPER_STD__TYPE_TRAITS_CONJUNCTION_H
