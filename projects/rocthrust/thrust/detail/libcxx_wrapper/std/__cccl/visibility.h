//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_VISIBILITY_H
#define LIBCXX_WRAPPER_STD__CCCL_VISIBILITY_H

// TODO(libhipcxx): remove this file and replace THRUST_FORCEINLINE with _CCCL_FORCEINLINE in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include <thrust/detail/libcxx_wrapper/__cccl_config.h> // _THRUST_STD_INCLUDE(__cccl/visibility.h)

#  define THRUST_FORCEINLINE _CCCL_FORCEINLINE

#else

#  if THRUST_COMPILER(MSVC)
#    define THRUST_FORCEINLINE __forceinline
#  else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv THRUST_COMPILER(MSVC) vvv
#    define THRUST_FORCEINLINE __inline__ __attribute__((__always_inline__))
#  endif // !THRUST_COMPILER(MSVC)

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_VISIBILITY_H
