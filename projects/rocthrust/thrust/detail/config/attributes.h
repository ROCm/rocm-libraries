//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_ATTRIBUTES_H
#define CONFIG_ATTRIBUTES_H

// TODO(libhipcxx): remove this file and replace THRUST_DECLSPEC_EMPTY_BASES, THRUST_NODISCARD_FRIEND and
// THRUST_ALIAS_ATTRIBUTE with _CCCL_DECLSPEC_EMPTY_BASES, _CCCL_NODISCARD_FRIEND and _CCCL_ALIAS_ATTRIBUTE in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/attributes.h)
// clang-format on

#  define THRUST_HAS_ATTRIBUTE(__x)   _CCCL_HAS_ATTRIBUTE(__x)
#  define THRUST_DECLSPEC_EMPTY_BASES _CCCL_DECLSPEC_EMPTY_BASES
#  define THRUST_NODISCARD_FRIEND     _CCCL_NODISCARD_FRIEND
#  define THRUST_ALIAS_ATTRIBUTE(...) _CCCL_ALIAS_ATTRIBUTE(__VA_ARGS__)

#else // !_THRUST_HAS_DEVICE_SYSTEM_STD

#  include <thrust/detail/config/compiler.h>

#  ifdef __has_attribute
#    define THRUST_HAS_ATTRIBUTE(__x) __has_attribute(__x)
#  else // ^^^ __has_attribute ^^^ / vvv !__has_attribute vvv
#    define THRUST_HAS_ATTRIBUTE(__x) 0
#  endif // !__has_attribute

#  ifdef __has_declspec_attribute
#    define THRUST_HAS_DECLSPEC_ATTRIBUTE(__x) __has_declspec_attribute(__x)
#  else // ^^^ __has_declspec_attribute ^^^ / vvv !__has_declspec_attribute vvv
#    define THRUST_HAS_DECLSPEC_ATTRIBUTE(__x) 0
#  endif // !__has_declspec_attribute

// MSVC needs extra help with empty base classes
#  if THRUST_COMPILER(MSVC) || THRUST_HAS_DECLSPEC_ATTRIBUTE(empty_bases)
#    define THRUST_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#  else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv !THRUST_COMPILER(MSVC) vvv
#    define THRUST_DECLSPEC_EMPTY_BASES
#  endif // !THRUST_COMPILER(MSVC)

// NVCC below 11.3 does not support nodiscard on friend operators
// It always fails with clang
#  if THRUST_COMPILER(CLANG) || THRUST_COMPILER(HIP)
#    define THRUST_NODISCARD_FRIEND friend
#  else
#    define THRUST_NODISCARD_FRIEND THRUST_NODISCARD friend
#  endif

#  define THRUST_ALIAS_ATTRIBUTE(...) __VA_ARGS__

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

#endif // CONFIG_ATTRIBUTES_H
