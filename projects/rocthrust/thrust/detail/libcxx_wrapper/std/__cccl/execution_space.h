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

#ifndef LIBCXX_WRAPPER_STD__CCCL_EXECUTION_SPACE_H
#define LIBCXX_WRAPPER_STD__CCCL_EXECUTION_SPACE_H

// TODO(libhipcxx): remove this file and replace _CCCL* with THRUST* once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/execution_space.h)

#  if THRUST_HAS_HIP_COMPILER()
#    undef _CCCL_HOST
#    undef _CCCL_DEVICE
#    undef _CCCL_HOST_DEVICE
#    define _CCCL_HOST        __host__
#    define _CCCL_DEVICE      __device__
#    define _CCCL_HOST_DEVICE __host__ __device__
#  endif
#  define THRUST_HOST        _CCCL_HOST
#  define THRUST_DEVICE      _CCCL_DEVICE
#  define THRUST_HOST_DEVICE _CCCL_HOST_DEVICE

#  define THRUST_EXEC_CHECK_DISABLE _CCCL_EXEC_CHECK_DISABLE

#else

#  if THRUST_CUDA_COMPILATION() || THRUST_HAS_HIP_COMPILER()
#    define THRUST_HOST        __host__
#    define THRUST_DEVICE      __device__
#    define THRUST_HOST_DEVICE __host__ __device__
#  else // ^^^ (CUDA_COMPILATION || HIP_COMPILATION) ^^^ / vvv !(CUDA_COMPILATION || HIP_COMPILATION) vvv
#    define THRUST_HOST
#    define THRUST_DEVICE
#    define THRUST_HOST_DEVICE
#  endif // !(CUDA_COMPILATION || HIP_COMPILATION)

#  if !defined(THRUST_EXEC_CHECK_DISABLE)
#    if THRUST_CUDA_COMPILER(NVCC)
#      define THRUST_EXEC_CHECK_DISABLE THRUST_PRAGMA(nv_exec_check_disable)
#    else
#      define THRUST_EXEC_CHECK_DISABLE
#    endif // THRUST_CUDA_COMPILER(NVCC)
#  endif // !THRUST_EXEC_CHECK_DISABLE

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_EXECUTION_SPACE_H
