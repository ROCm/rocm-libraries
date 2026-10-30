//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__TYPE_TRAITS_TYPE_IDENTITY_H
#define LIBCXX_WRAPPER_STD__TYPE_TRAITS_TYPE_IDENTITY_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/dialect.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__type_traits/type_identity.h)
#elif THRUST_STD_VER >= 2020
#  include <type_traits>
#endif

namespace internal
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD || THRUST_STD_VER >= 2020

using _THRUST_STD::type_identity;

#else

template <typename Tp>
struct type_identity
{
  using type = Tp;
};

#endif
} // namespace internal

#endif // LIBCXX_WRAPPER_STD__TYPE_TRAITS_TYPE_IDENTITY_H
