//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_DIAGNOSTIC_H
#define CONFIG_DIAGNOSTIC_H

// TODO(libhipcxx): remove this file and replace all the THRUST* macros with _CCCL* macros in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/diagnostic.h)
// clang-format on

#  define THRUST_DIAG_PUSH                _CCCL_DIAG_PUSH
#  define THRUST_DIAG_POP                 _CCCL_DIAG_POP
#  define THRUST_DIAG_SUPPRESS_CLANG(str) _CCCL_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)   _CCCL_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str) _CCCL_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)  _CCCL_DIAG_SUPPRESS_MSVC(str)
#  define THRUST_SUPPRESS_DEPRECATED_PUSH _CCCL_SUPPRESS_DEPRECATED_PUSH
#  define THRUST_SUPPRESS_DEPRECATED_POP  _CCCL_SUPPRESS_DEPRECATED_POP

#else

#  include <thrust/detail/config/compiler.h>

// Enable us to selectively silence host compiler warnings
#  if THRUST_COMPILER(CLANG) || THRUST_COMPILER(HIP)
#    define THRUST_DIAG_PUSH                THRUST_PRAGMA(clang diagnostic push)
#    define THRUST_DIAG_POP                 THRUST_PRAGMA(clang diagnostic pop)
#    define THRUST_DIAG_SUPPRESS_CLANG(str) THRUST_PRAGMA(clang diagnostic ignored str)
#    define THRUST_DIAG_SUPPRESS_GCC(str)
#    define THRUST_DIAG_SUPPRESS_NVHPC(str)
#    define THRUST_DIAG_SUPPRESS_MSVC(str)
#  elif THRUST_COMPILER(GCC)
#    define THRUST_DIAG_PUSH THRUST_PRAGMA(GCC diagnostic push)
#    define THRUST_DIAG_POP  THRUST_PRAGMA(GCC diagnostic pop)
#    define THRUST_DIAG_SUPPRESS_CLANG(str)
#    define THRUST_DIAG_SUPPRESS_GCC(str) THRUST_PRAGMA(GCC diagnostic ignored str)
#    define THRUST_DIAG_SUPPRESS_NVHPC(str)
#    define THRUST_DIAG_SUPPRESS_MSVC(str)
#  elif THRUST_COMPILER(NVHPC)
#    define THRUST_DIAG_PUSH THRUST_PRAGMA(diagnostic push)
#    define THRUST_DIAG_POP  THRUST_PRAGMA(diagnostic pop)
#    define THRUST_DIAG_SUPPRESS_CLANG(str)
#    define THRUST_DIAG_SUPPRESS_GCC(str)
#    define THRUST_DIAG_SUPPRESS_NVHPC(str) THRUST_PRAGMA(diag_suppress str)
#    define THRUST_DIAG_SUPPRESS_MSVC(str)
#  elif THRUST_COMPILER(MSVC)
#    define THRUST_DIAG_PUSH THRUST_PRAGMA(warning(push))
#    define THRUST_DIAG_POP  THRUST_PRAGMA(warning(pop))
#    define THRUST_DIAG_SUPPRESS_CLANG(str)
#    define THRUST_DIAG_SUPPRESS_GCC(str)
#    define THRUST_DIAG_SUPPRESS_NVHPC(str)
#    define THRUST_DIAG_SUPPRESS_MSVC(str) THRUST_PRAGMA(warning(disable : str))
#  else
#    define THRUST_DIAG_PUSH
#    define THRUST_DIAG_POP
#    define THRUST_DIAG_SUPPRESS_CLANG(str)
#    define THRUST_DIAG_SUPPRESS_GCC(str)
#    define THRUST_DIAG_SUPPRESS_NVHPC(str)
#    define THRUST_DIAG_SUPPRESS_MSVC(str)
#  endif

// Convenient shortcuts to silence common warnings
#  if THRUST_COMPILER(CLANG) || THRUST_COMPILER(HIP)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH      \
      THRUST_DIAG_PUSH                           \
      THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated") \
      THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations")
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#  elif THRUST_COMPILER(GCC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH    \
      THRUST_DIAG_PUSH                         \
      THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated") \
      THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations")
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#  elif THRUST_COMPILER(NVHPC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH         \
      THRUST_DIAG_PUSH                              \
      THRUST_DIAG_SUPPRESS_NVHPC(deprecated_entity) \
      THRUST_DIAG_SUPPRESS_NVHPC(deprecated_entity_with_custom_message)
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#  elif THRUST_COMPILER(MSVC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH \
      THRUST_DIAG_PUSH                      \
      THRUST_DIAG_SUPPRESS_MSVC(4996)
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#  else // !THRUST_COMPILER_CLANG && !THRUST_COMPILER_GCC
#    define THRUST_SUPPRESS_DEPRECATED_PUSH
#    define THRUST_SUPPRESS_DEPRECATED_POP
#  endif // !THRUST_COMPILER_CLANG && !THRUST_COMPILER_GCC

#endif

#endif // CONFIG_DIAGNOSTIC_H
