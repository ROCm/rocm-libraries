//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef THRUST_DETAIL_CONFIG_EXECUTION_SPACE_H
#define THRUST_DETAIL_CONFIG_EXECUTION_SPACE_H

#include <thrust/detail/config/compiler.h>

#if (THRUST_DEVICE_COMPILER != THRUST_DEVICE_COMPILER_NVCC) && (THRUST_DEVICE_COMPILER != THRUST_DEVICE_COMPILER_HIP)
#define THRUST_HOST
#define THRUST_DEVICE
#define THRUST_HOST_DEVICE
#define THRUST_FORCEINLINE
#else
#define THRUST_HOST __host__
#define THRUST_DEVICE __device__
#define THRUST_HOST_DEVICE __host__ __device__
#define THRUST_FORCEINLINE __forceinline__
#endif

#if !defined(__HIP__)
#if !defined(THRUST_EXEC_CHECK_DISABLE)
#  if defined(_CCCL_CUDA_COMPILER_NVCC)
#    if defined(_CCCL_COMPILER_MSVC)
#      define THRUST_EXEC_CHECK_DISABLE __pragma("nv_exec_check_disable")
#    else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#      define THRUST_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#    endif // !_CCCL_COMPILER_MSVC
#  else
#    define THRUST_EXEC_CHECK_DISABLE
#  endif // _CCCL_CUDA_COMPILER_NVCC
#endif // !THRUST_EXEC_CHECK_DISABLE
#else
#define THRUST_EXEC_CHECK_DISABLE
#endif // !HIP

#endif // THRUST_DETAIL_CONFIG_EXECUTION_SPACE_H