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

#ifndef LIBCXX_WRAPPER_STD__CCCL_EXCEPTIONS_H
#define LIBCXX_WRAPPER_STD__CCCL_EXCEPTIONS_H

// TODO(libhipcxx): remove this file and replace THRUST_NO_EXCEPTIONS with _CCCL_NO_EXCEPTIONS in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/exceptions.h)

#  ifdef _CCCL_NO_EXCEPTIONS
#    define THRUST_NO_EXCEPTIONS
#  endif

#else

#  ifndef THRUST_NO_EXCEPTIONS
#    if defined(THRUST_DISABLE_EXCEPTIONS) // Escape hatch for users to manually disable exceptions
#      define THRUST_NO_EXCEPTIONS
#    elif THRUST_COMPILER(NVRTC) || (THRUST_COMPILER(MSVC) && _CPPUNWIND == 0) \
      || (!THRUST_COMPILER(MSVC) && !__EXCEPTIONS) // Catches all non msvc based compilers
#      define THRUST_NO_EXCEPTIONS
#    endif
#  endif // !THRUST_NO_EXCEPTIONS

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_EXCEPTIONS_H
