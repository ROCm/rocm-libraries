//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef THRUST_DETAIL_CONFIG_DIAGNOSTIC_H
#define THRUST_DETAIL_CONFIG_DIAGNOSTIC_H

#include <thrust/detail/config/compiler.h>

// Enable us to selectively silence host compiler warnings
#define THRUST_TOSTRING2(_STR) #_STR
#define THRUST_TOSTRING(_STR)  THRUST_TOSTRING2(_STR)
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
#  define THRUST_DIAG_PUSH                _Pragma("clang diagnostic push")
#  define THRUST_DIAG_POP                 _Pragma("clang diagnostic pop")
#  define THRUST_DIAG_SUPPRESS_CLANG(str) _Pragma(THRUST_TOSTRING(clang diagnostic ignored str))
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#elif (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) || (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_INTEL)
#  define THRUST_DIAG_PUSH _Pragma("GCC diagnostic push")
#  define THRUST_DIAG_POP  _Pragma("GCC diagnostic pop")
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str) _Pragma(THRUST_TOSTRING(GCC diagnostic ignored str))
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVHPC
#  define THRUST_DIAG_PUSH _Pragma("diagnostic push")
#  define THRUST_DIAG_POP  _Pragma("diagnostic pop")
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str) _Pragma(THRUST_TOSTRING(diag_suppress str))
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_DIAG_PUSH __pragma(warning(push))
#  define THRUST_DIAG_POP  __pragma(warning(pop))
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str) __pragma(warning(disable : str))
#else
#  define THRUST_DIAG_PUSH
#  define THRUST_DIAG_POP
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#endif

// Convenient shortcuts to silence common warnings
#if THRUST_HOST_COMPILER == THRUST_DEVICE_COMPILER_CLANG
#  define THRUST_SUPPRESS_DEPRECATED_PUSH \
    THRUST_DIAG_PUSH                           \
    THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated") \
    THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations")
#  define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#elif (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) || (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_ICC)
#  define THRUST_SUPPRESS_DEPRECATED_PUSH \
    THRUST_DIAG_PUSH                           \
    THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated")   \
    THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations")
#  define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_SUPPRESS_DEPRECATED_PUSH \
    THRUST_DIAG_PUSH                           \
    THRUST_DIAG_SUPPRESS_MSVC(4996)
#  define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#else // !THRUST_COMPILER_CLANG && !THRUST_COMPILER_GCC
#  define THRUST_SUPPRESS_DEPRECATED_PUSH
#  define THRUST_SUPPRESS_DEPRECATED_POP
#endif // !THRUST_COMPILER_CLANG && !THRUST_COMPILER_GCC

#endif // THRUST_DETAIL_CONFIG_DIAGNOSTIC_H