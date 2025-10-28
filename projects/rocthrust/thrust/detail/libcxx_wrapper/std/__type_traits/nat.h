//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__TYPE_TRAITS_NAT_H
#define LIBCXX_WRAPPER_STD__TYPE_TRAITS_NAT_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if !_THRUST_HAS_DEVICE_SYSTEM_STD

namespace internal
{
namespace detail
{
struct nat
{
  nat()                      = delete;
  nat(const nat&)            = delete;
  nat& operator=(const nat&) = delete;
  ~nat()                     = delete;
};
} // namespace detail
} // namespace internal

#endif

#endif // LIBCXX_WRAPPER_STD__TYPE_TRAITS_NAT_H
