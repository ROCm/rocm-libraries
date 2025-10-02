//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_EXECUTION_SPACE_H
#define CONFIG_EXECUTION_SPACE_H

// TODO(libhipcxx): needs to check this file once libhipcxx gets ready

#include <thrust/detail/config/compiler.h>

// We need to ensure that we not only compile with a cuda or hip compiler but also compile cuda or hip source files
#if (defined(__NVCC__) || defined(_NVHPC_CUDA)                                    \
     || (defined(__CUDA__) && THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG) \
     || THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVRTC)                       \
  || (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP)
#  define THRUST_HOST        __host__
#  define THRUST_DEVICE      __device__
#  define THRUST_HOST_DEVICE __host__ __device__
#else // ^^^ (CUDA_COMPILATION || HIP_COMPILATION) ^^^ / vvv !(CUDA_COMPILATION || HIP_COMPILATION) vvv
#  define THRUST_HOST
#  define THRUST_DEVICE
#  define THRUST_HOST_DEVICE
#endif // !(CUDA_COMPILATION || HIP_COMPILATION)

#if !defined(__HIP__)
#  if !defined(THRUST_EXEC_CHECK_DISABLE)
#    if defined(__NVCC__)
#      define THRUST_EXEC_CHECK_DISABLE THRUST_PRAGMA(nv_exec_check_disable)
#    else
#      define THRUST_EXEC_CHECK_DISABLE
#    endif // __NVCC__
#  endif // !THRUST_EXEC_CHECK_DISABLE
#else
#  define THRUST_EXEC_CHECK_DISABLE
#endif // !HIP

#endif // CONFIG_EXECUTION_SPACE_H
