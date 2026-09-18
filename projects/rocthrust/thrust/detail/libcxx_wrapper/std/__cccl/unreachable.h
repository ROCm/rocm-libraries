//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_UNREACHABLE_H
#define LIBCXX_WRAPPER_STD__CCCL_UNREACHABLE_H

// TODO(libhipcxx): remove this file and replace THRUST* with _CCCL* in rocThrust once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/unreachable.h)

#  define THRUST_UNREACHABLE() _CCCL_UNREACHABLE()

#else

#  if THRUST_CUDA_COMPILER(CLANG)
#    define THRUST_UNREACHABLE() __builtin_unreachable()
#  elif defined(__CUDA_ARCH__)
#    define THRUST_UNREACHABLE() __builtin_unreachable()
#  else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#    if THRUST_COMPILER(MSVC)
#      define THRUST_UNREACHABLE() __assume(0)
#    else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv !THRUST_COMPILER(MSVC) vvv
#      define THRUST_UNREACHABLE() __builtin_unreachable()
#    endif // !THRUST_COMPILER(MSVC)
#  endif // !__CUDA_ARCH__

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_UNREACHABLE_H
