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

#ifndef THRUST_DETAIL_CONFIG_DIAGNOSTIC_H
#define THRUST_DETAIL_CONFIG_DIAGNOSTIC_H

#include <thrust/detail/config/compiler.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Enable us to selectively silence host compiler warnings
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
#  define THRUST_DIAG_PUSH                THRUST_PRAGMA(clang diagnostic push)
#  define THRUST_DIAG_POP                 THRUST_PRAGMA(clang diagnostic pop)
#  define THRUST_DIAG_SUPPRESS_CLANG(str) THRUST_PRAGMA(clang diagnostic ignored str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#  define THRUST_DIAG_SUPPRESS_ICC(str)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
#  define THRUST_DIAG_PUSH THRUST_PRAGMA(GCC diagnostic push)
#  define THRUST_DIAG_POP  THRUST_PRAGMA(GCC diagnostic pop)
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str) THRUST_PRAGMA(GCC diagnostic ignored str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#  define THRUST_DIAG_SUPPRESS_ICC(str)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_INTEL
#  define THRUST_DIAG_PUSH THRUST_PRAGMA(GCC diagnostic push)
#  define THRUST_DIAG_POP  THRUST_PRAGMA(GCC diagnostic pop)
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str) THRUST_PRAGMA(GCC diagnostic ignored str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#  define THRUST_DIAG_SUPPRESS_ICC(str) THRUST_PRAGMA(warning disable str)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVHPC
#  define THRUST_DIAG_PUSH THRUST_PRAGMA(diagnostic push)
#  define THRUST_DIAG_POP  THRUST_PRAGMA(diagnostic pop)
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str) THRUST_PRAGMA(diag_suppress str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#  define THRUST_DIAG_SUPPRESS_ICC(str)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_DIAG_PUSH THRUST_PRAGMA(warning(push))
#  define THRUST_DIAG_POP  THRUST_PRAGMA(warning(pop))
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str) THRUST_PRAGMA(warning(disable : str))
#  define THRUST_DIAG_SUPPRESS_ICC(str)
#else
#  define THRUST_DIAG_PUSH
#  define THRUST_DIAG_POP
#  define THRUST_DIAG_SUPPRESS_CLANG(str)
#  define THRUST_DIAG_SUPPRESS_GCC(str)
#  define THRUST_DIAG_SUPPRESS_NVHPC(str)
#  define THRUST_DIAG_SUPPRESS_MSVC(str)
#  define THRUST_DIAG_SUPPRESS_ICC(str)
#endif

// Convenient shortcuts to silence common warnings
#if THRUST_HOST_COMPILER == THRUST_DEVICE_COMPILER_CLANG
#  define THRUST_SUPPRESS_DEPRECATED_PUSH      \
    THRUST_DIAG_PUSH                           \
    THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated") \
    THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations")
#  define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#elif (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC) || (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_INTEL)
#  define THRUST_SUPPRESS_DEPRECATED_PUSH    \
    THRUST_DIAG_PUSH                         \
    THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated") \
    THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations")
#  define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_SUPPRESS_DEPRECATED_PUSH \
    THRUST_DIAG_PUSH                      \
    THRUST_DIAG_SUPPRESS_MSVC(4996)
#  define THRUST_SUPPRESS_DEPRECATED_POP THRUST_DIAG_POP
#else // !THRUST_COMPILER_CLANG && !THRUST_COMPILER_GCC
#  define THRUST_SUPPRESS_DEPRECATED_PUSH
#  define THRUST_SUPPRESS_DEPRECATED_POP
#endif // !THRUST_COMPILER_CLANG && !THRUST_COMPILER_GCC

#endif // THRUST_DETAIL_CONFIG_DIAGNOSTIC_H
