//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef DETAIL_ATTRIBUTES_H
#define DETAIL_ATTRIBUTES_H

#include <thrust/detail/config.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/attributes.h)
// clang-format on

#  define THRUST_DECLSPEC_EMPTY_BASES _CCCL_DECLSPEC_EMPTY_BASES

#else // !_THRUST_HAS_DEVICE_SYSTEM_STD

#  ifdef __has_declspec_attribute
#    define THRUST_HAS_DECLSPEC_ATTRIBUTE(__x) __has_declspec_attribute(__x)
#  else // ^^^ __has_declspec_attribute ^^^ / vvv !__has_declspec_attribute vvv
#    define THRUST_HAS_DECLSPEC_ATTRIBUTE(__x) 0
#  endif // !__has_declspec_attribute

// MSVC needs extra help with empty base classes
#  if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || THRUST_HAS_DECLSPEC_ATTRIBUTE(empty_bases)
#    define THRUST_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#  else // !THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#    define THRUST_DECLSPEC_EMPTY_BASES
#  endif // THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

#endif // DETAIL_ATTRIBUTES_H
