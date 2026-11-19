//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_DIALECT_H
#define LIBCXX_WRAPPER_STD__CCCL_DIALECT_H

// TODO(libhipcxx): remove this file and replace THRUST* with _CCCL* in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/dialect.h)

#  define THRUST_STD_VER             _CCCL_STD_VER
#  define THRUST_CONSTEXPR_CXX20     _CCCL_CONSTEXPR_CXX20
#  define THRUST_CONSTEXPR_CXX23     _CCCL_CONSTEXPR_CXX23
#  define THRUST_TRAIT(__TRAIT, ...) _CCCL_TRAIT(__TRAIT, __VA_ARGS__)
#  define THRUST_GLOBAL_CONSTANT     _CCCL_GLOBAL_CONSTANT

#else

#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>

#  if THRUST_COMPILER(MSVC)
#    if _MSVC_LANG <= 201103L
#      define THRUST_STD_VER 2011
#    elif _MSVC_LANG <= 201402L
#      define THRUST_STD_VER 2014
#    elif _MSVC_LANG <= 201703L
#      define THRUST_STD_VER 2017
#    elif _MSVC_LANG <= 202002L
#      define THRUST_STD_VER 2020
#    else
#      define THRUST_STD_VER 2023 // current year, or date of c++2b ratification
#    endif
#  else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv !THRUST_COMPILER(MSVC) vvv
#    if __cplusplus <= 199711L
#      define THRUST_STD_VER 2003
#    elif __cplusplus <= 201103L
#      define THRUST_STD_VER 2011
#    elif __cplusplus <= 201402L
#      define THRUST_STD_VER 2014
#    elif __cplusplus <= 201703L
#      define THRUST_STD_VER 2017
#    elif __cplusplus <= 202002L
#      define THRUST_STD_VER 2020
#    elif __cplusplus <= 202302L
#      define THRUST_STD_VER 2023
#    else
#      define THRUST_STD_VER 2024 // current year, or date of c++2c ratification
#    endif
#  endif // !THRUST_COMPILER(MSVC)

#  if THRUST_STD_VER >= 2020
#    define THRUST_CONSTEXPR_CXX20 constexpr
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
#    define THRUST_CONSTEXPR_CXX20
#  endif // THRUST_STD_VER <= 2017

#  if THRUST_STD_VER >= 2023
#    define THRUST_CONSTEXPR_CXX23 constexpr
#  else // ^^^ C++23 ^^^ / vvv C++20 vvv
#    define THRUST_CONSTEXPR_CXX23
#  endif // THRUST_STD_VER <= 2020

#  define THRUST_TRAIT(__TRAIT, ...) __TRAIT##_v<__VA_ARGS__>

#  if defined(__CUDA_ARCH__)
#    define THRUST_GLOBAL_CONSTANT THRUST_DEVICE constexpr
#  else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#    define THRUST_GLOBAL_CONSTANT inline constexpr
#  endif // __CUDA_ARCH__

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_DIALECT_H
