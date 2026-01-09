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

#ifndef LIBCXX_WRAPPER_STD__CCCL_RTTI_H
#define LIBCXX_WRAPPER_STD__CCCL_RTTI_H

// TODO(libhipcxx): remove this file and replace THRUST_NO_RTTI with _CCCL_NO_RTTI in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/rtti.h)

#  if !defined(CCCL_DISABLE_RTTI) && THRUST_HAS_HIP_COMPILER()
#    undef _CCCL_NO_RTTI
#    if !_CCCL_HAS_FEATURE(cxx_rtti)
#      define _CCCL_NO_RTTI
#    endif
#  endif
#  ifdef _CCCL_NO_RTTI
#    define THRUST_NO_RTTI
#  endif // _CCCL_NO_RTTI

#else

#  include <thrust/detail/libcxx_wrapper/std/__cccl/builtin.h>

// NOTE: some compilers support the `typeid` feature but not the `dynamic_cast`
// feature. This is why we have separate macros for each.

#  ifndef THRUST_NO_RTTI
#    if defined(THRUST_DISABLE_RTTI) // Escape hatch for users to manually disable RTTI
#      define THRUST_NO_RTTI
#    elif defined(__CUDA_ARCH__)
#      define THRUST_NO_RTTI // No RTTI in CUDA device code
#    elif THRUST_COMPILER(NVRTC)
#      define THRUST_NO_RTTI
#    elif THRUST_COMPILER(MSVC)
#      if _CPPRTTI == 0
#        define THRUST_NO_RTTI
#      endif
#    elif THRUST_COMPILER(CLANG)
#      if !THRUST_HAS_FEATURE(cxx_rtti)
#        define THRUST_NO_RTTI
#      endif
#    else
#      if __GXX_RTTI == 0 && __cpp_rtti == 0
#        define THRUST_NO_RTTI
#      endif
#    endif
#  endif // !THRUST_NO_RTTI

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

#endif // LIBCXX_WRAPPER_STD__CCCL_RTTI_H
