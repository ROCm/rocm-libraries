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
#  define THRUST_TRAIT           _CCCL_TRAIT
#  define THRUST_GLOBAL_CONSTANT _CCCL_GLOBAL_CONSTANT

#else

#  include <thrust/detail/config/cpp_dialect.h>
#  include <thrust/detail/config/execution_space.h>

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

#  define THRUST_TRAIT(__TRAIT, ...) __TRAIT##_v<__VA_ARGS__>

#  define THRUST_CONSTEXPR_GLOBAL constexpr

// We need to treat host and device separately
#  if defined(__CUDA_ARCH__)
#    define THRUST_GLOBAL_CONSTANT THRUST_DEVICE THRUST_CONSTEXPR_GLOBAL
#  else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#    define THRUST_GLOBAL_CONSTANT inline constexpr
#  endif // __CUDA_ARCH__

#endif

#endif // CONFIG_DIALECT_H
