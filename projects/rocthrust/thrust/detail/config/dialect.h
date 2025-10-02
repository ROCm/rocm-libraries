//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_DIALECT_H
#define CONFIG_DIALECT_H

// TODO(libhipcxx): remove this file and replace THRUST* with _CCCL* in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/dialect.h)
// clang-format on

#  define THRUST_CONSTEXPR_CXX17 _CCCL_CONSTEXPR_CXX17
#  define THRUST_CONSTEXPR_CXX20 _CCCL_CONSTEXPR_CXX20
#  define THRUST_CONSTEXPR_CXX23 _CCCL_CONSTEXPR_CXX23
#  define THRUST_IF_CONSTEXPR    _CCCL_IF_CONSTEXPR

#else

#  include <thrust/detail/config/cpp_dialect.h>

// Constexpr feature macros:
#  if THRUST_CPP_DIALECT >= 2017
#    define THRUST_CONSTEXPR_CXX17 constexpr
#  else
#    define THRUST_CONSTEXPR_CXX17
#  endif

#  if THRUST_CPP_DIALECT >= 2020
#    define THRUST_CONSTEXPR_CXX20 constexpr
#  else
#    define THRUST_CONSTEXPR_CXX20
#  endif

#  if THRUST_CPP_DIALECT >= 2023
#    define THRUST_CONSTEXPR_CXX23 constexpr
#  else
#    define THRUST_CONSTEXPR_CXX23
#  endif

#  if THRUST_CPP_DIALECT <= 2014 || (defined(__cpp_if_constexpr) && __cpp_if_constexpr < 201606L)
#    define THRUST_IF_CONSTEXPR if
#  else // ^^^ _CCCL_NO_IF_CONSTEXPR ^^^ / vvv !_CCCL_NO_IF_CONSTEXPR vvv
#    define THRUST_IF_CONSTEXPR if constexpr
#  endif // !_CCCL_NO_IF_CONSTEXPR

#endif

#endif // CONFIG_DIALECT_H
