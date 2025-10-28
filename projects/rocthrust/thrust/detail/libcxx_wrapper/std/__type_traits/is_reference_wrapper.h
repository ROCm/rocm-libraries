//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
#define LIBCXX_WRAPPER_STD__TYPE_TRAITS_IS_REFERENCE_WRAPPER_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if !_THRUST_HAS_DEVICE_SYSTEM_STD

#  include <functional>
#  include <type_traits>

namespace internal
{
namespace detail
{
template <typename Tp>
struct is_reference_wrapper_impl : public ::std::false_type
{};
template <typename Tp>
struct is_reference_wrapper_impl<::std::reference_wrapper<Tp>> : public ::std::true_type
{};
template <typename Tp>
struct is_reference_wrapper : public is_reference_wrapper_impl<::std::remove_cv_t<Tp>>
{};
} // namespace detail
} // namespace internal

#endif

#endif // LIBCXX_WRAPPER_STD__TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
