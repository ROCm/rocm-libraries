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

#ifndef CONFIG_CUDA_TOOLKIT_H
#define CONFIG_CUDA_TOOLKIT_H

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/cuda_toolkit.h)
// clang-format on

#  define THRUST_CTK_AT_LEAST _CCCL_CTK_AT_LEAST

#else

#  include <thrust/detail/config/compiler.h>
#  include <thrust/detail/config/preprocessor.h>

#  if THRUST_CUDA_COMPILATION() || THRUST_HAS_INCLUDE(<cuda_runtime_api.h>)
#    define THRUST_HAS_CTK() 1
#  else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#    define THRUST_HAS_CTK() 0
#  endif // ^^^ no cuda toolkit ^^^

// CUDA compilers preinclude cuda_runtime.h, so we need to include it here to get the CUDART_VERSION macro
#  if THRUST_HAS_CTK() && !THRUST_CUDA_COMPILATION()
#    include <cuda_runtime_api.h>
#  endif // THRUST_HAS_CTK() && !THRUST_CUDA_COMPILATION()

// Check compatibility of the CUDA compiler and CUDA toolkit headers
#  if THRUST_CUDA_COMPILATION()
#    if !THRUST_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#      error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
#    endif // !THRUST_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#  endif // THRUST_CUDA_COMPILATION()

#  define THRUST_CTK_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)
#  define THRUST_CTK()                            (CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10)
#  define THRUST_CTK_AT_LEAST(...)                THRUST_HAS_CTK() && THRUST_VERSION_COMPARE(THRUST_CTK_, THRUST_CTK, >=, __VA_ARGS__)

#endif

#endif // CONFIG_CUDA_TOOLKIT_H
