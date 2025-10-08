//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_VISIBILITY_H
#define CONFIG_VISIBILITY_H

// TODO(libhipcxx): remove this file and replace THRUST_FORCEINLINE with _CCCL_FORCEINLINE in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/visibility.h)
// clang-format on

#  define THRUST_FORCEINLINE _CCCL_FORCEINLINE

#else

#  include <thrust/detail/config/compiler.h>

#  if THRUST_COMPILER(MSVC)
#    define THRUST_FORCEINLINE __forceinline
#  else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv THRUST_COMPILER(MSVC) vvv
#    define THRUST_FORCEINLINE __inline__ __attribute__((__always_inline__))
#  endif // !THRUST_COMPILER(MSVC)

#endif

#endif // CONFIG_VISIBILITY_H
