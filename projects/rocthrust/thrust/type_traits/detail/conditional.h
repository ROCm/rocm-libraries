//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TYPE_TRAITS_DETAIL_CONDITIONAL_H
#define TYPE_TRAITS_DETAIL_CONDITIONAL_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
// clang-format off
#  include _THRUST_STD_INCLUDE(__type_traits/conditional.h)
// clang-format on
#endif

namespace internal
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <bool Cond, typename IfRes, typename ElseRes>
using If = _THRUST_STD::_If<Cond, IfRes, ElseRes>;

#else

namespace detail
{

template <bool>
struct IfImpl;

template <>
struct IfImpl<true>
{
  template <typename IfRes, typename ElseRes>
  using Select = IfRes;
};

template <>
struct IfImpl<false>
{
  template <typename IfRes, typename ElseRes>
  using Select = ElseRes;
};

} // namespace detail

template <bool Cond, typename IfRes, typename ElseRes>
using If = typename detail::IfImpl<Cond>::template Select<IfRes, ElseRes>;

#endif
} // namespace internal

#endif // TYPE_TRAITS_DETAIL_CONDITIONAL_H
