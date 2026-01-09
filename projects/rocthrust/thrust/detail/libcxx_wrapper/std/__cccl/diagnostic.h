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

#ifndef LIBCXX_WRAPPER_STD__CCCL_DIAGNOSTIC_H
#define LIBCXX_WRAPPER_STD__CCCL_DIAGNOSTIC_H

// TODO(libhipcxx): remove this file and replace all the THRUST* macros with _CCCL* macros in rocThrust
// once libhipcxx v3.0.1 gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD && _LIBCUDACXX_CUDA_API_VERSION_PATCH >= 1

#  include _THRUST_STD_INCLUDE(__cccl/diagnostic.h)

#  define THRUST_DIAG_PUSH                     _CCCL_DIAG_PUSH
#  define THRUST_DIAG_POP                      _CCCL_DIAG_POP
#  define THRUST_DIAG_SUPPRESS_CLANG(_WARNING) _CCCL_DIAG_SUPPRESS_CLANG(_WARNING)
#  define THRUST_DIAG_SUPPRESS_MSVC(_WARNING)  _CCCL_DIAG_SUPPRESS_MSVC(_WARNING)
#  define THRUST_SUPPRESS_DEPRECATED_PUSH      _CCCL_SUPPRESS_DEPRECATED_PUSH
#  define THRUST_SUPPRESS_DEPRECATED_POP       _CCCL_SUPPRESS_DEPRECATED_POP

#else // !_THRUST_HAS_DEVICE_SYSTEM_STD || _LIBCUDACXX_CUDA_API_VERSION_PATCH < 1

#  include <thrust/detail/libcxx_wrapper/std/__cccl/preprocessor.h>

#  if THRUST_COMPILER(CLANG)
#    define THRUST_DIAG_PUSH                     THRUST_PRAGMA(clang diagnostic push)
#    define THRUST_DIAG_POP                      THRUST_PRAGMA(clang diagnostic pop)
#    define THRUST_DIAG_SUPPRESS_CLANG(_WARNING) THRUST_PRAGMA(clang diagnostic ignored _WARNING)
#    define THRUST_DIAG_SUPPRESS_GCC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_NVHPC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_MSVC(_WARNING)
#  elif THRUST_COMPILER(GCC)
#    define THRUST_DIAG_PUSH THRUST_PRAGMA(GCC diagnostic push)
#    define THRUST_DIAG_POP  THRUST_PRAGMA(GCC diagnostic pop)
#    define THRUST_DIAG_SUPPRESS_CLANG(_WARNING)
#    define THRUST_DIAG_SUPPRESS_GCC(_WARNING) THRUST_PRAGMA(GCC diagnostic ignored _WARNING)
#    define THRUST_DIAG_SUPPRESS_NVHPC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_MSVC(_WARNING)
#  elif THRUST_COMPILER(NVHPC)
#    define THRUST_DIAG_PUSH THRUST_PRAGMA(diagnostic push)
#    define THRUST_DIAG_POP  THRUST_PRAGMA(diagnostic pop)
#    define THRUST_DIAG_SUPPRESS_CLANG(_WARNING)
#    define THRUST_DIAG_SUPPRESS_GCC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_NVHPC(_WARNING) THRUST_PRAGMA(diag_suppress _WARNING)
#    define THRUST_DIAG_SUPPRESS_MSVC(_WARNING)
#  elif THRUST_COMPILER(MSVC)
#    define THRUST_DIAG_PUSH THRUST_PRAGMA(warning(push))
#    define THRUST_DIAG_POP  THRUST_PRAGMA(warning(pop))
#    define THRUST_DIAG_SUPPRESS_CLANG(_WARNING)
#    define THRUST_DIAG_SUPPRESS_GCC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_NVHPC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_MSVC(_WARNING) THRUST_PRAGMA(warning(disable : _WARNING))
#  else
#    define THRUST_DIAG_PUSH
#    define THRUST_DIAG_POP
#    define THRUST_DIAG_SUPPRESS_CLANG(_WARNING)
#    define THRUST_DIAG_SUPPRESS_GCC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_NVHPC(_WARNING)
#    define THRUST_DIAG_SUPPRESS_MSVC(_WARNING)
#  endif

#  if THRUST_CUDA_COMPILER(NVCC) || THRUST_COMPILER(NVRTC)
#    if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#      define THRUST_NV_DIAG_PUSH()               THRUST_PRAGMA(nv_diagnostic push)
#      define THRUST_NV_DIAG_POP()                THRUST_PRAGMA(nv_diagnostic pop)
#      define THRUST_DIAG_SUPPRESS_NVCC(_WARNING) THRUST_PRAGMA(nv_diag_suppress _WARNING)
#      define THRUST_BEGIN_NV_DIAG_SUPPRESS(...) \
        THRUST_NV_DIAG_PUSH() THRUST_PP_FOR_EACH(THRUST_DIAG_SUPPRESS_NVCC, __VA_ARGS__)
#      define THRUST_END_NV_DIAG_SUPPRESS() THRUST_NV_DIAG_POP()
#    else // ^^^ __NVCC_DIAG_PRAGMA_SUPPORT__ ^^^ / vvv !__NVCC_DIAG_PRAGMA_SUPPORT__ vvv
#      define THRUST_NV_DIAG_PUSH()               THRUST_PRAGMA(diagnostic push)
#      define THRUST_NV_DIAG_POP()                THRUST_PRAGMA(diagnostic pop)
#      define THRUST_DIAG_SUPPRESS_NVCC(_WARNING) THRUST_PRAGMA(diag_suppress _WARNING)
#      define THRUST_BEGIN_NV_DIAG_SUPPRESS(...) \
        THRUST_NV_DIAG_PUSH() THRUST_PP_FOR_EACH(THRUST_DIAG_SUPPRESS_NVCC, __VA_ARGS__)
#      define THRUST_END_NV_DIAG_SUPPRESS() THRUST_NV_DIAG_POP()
#    endif // !__NVCC_DIAG_PRAGMA_SUPPORT__
#  else // ^^^ THRUST_CUDA_COMPILER(NVCC) ^^^ / vvv !THRUST_CUDA_COMPILER(NVCC) vvv
#    define THRUST_NV_DIAG_PUSH()
#    define THRUST_NV_DIAG_POP()
#    define THRUST_DIAG_SUPPRESS_NVCC(_WARNING)
#    define THRUST_BEGIN_NV_DIAG_SUPPRESS(...)
#    define THRUST_END_NV_DIAG_SUPPRESS()
#  endif // !THRUST_CUDA_COMPILER(NVCC)

#  if THRUST_COMPILER(CLANG)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH                   \
      THRUST_DIAG_PUSH                                        \
      THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated")              \
      THRUST_DIAG_SUPPRESS_CLANG("-Wdeprecated-declarations") \
      THRUST_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_NV_DIAG_POP() THRUST_DIAG_POP
#  elif THRUST_COMPILER(GCC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH                 \
      THRUST_DIAG_PUSH                                      \
      THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated")              \
      THRUST_DIAG_SUPPRESS_GCC("-Wdeprecated-declarations") \
      THRUST_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_NV_DIAG_POP() THRUST_DIAG_POP
#  elif THRUST_COMPILER(NVHPC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH                             \
      THRUST_DIAG_PUSH                                                  \
      THRUST_DIAG_SUPPRESS_NVHPC(deprecated_entity)                     \
      THRUST_DIAG_SUPPRESS_NVHPC(deprecated_entity_with_custom_message) \
      THRUST_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_NV_DIAG_POP() THRUST_DIAG_POP
#  elif THRUST_COMPILER(MSVC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH \
      THRUST_DIAG_PUSH                      \
      THRUST_DIAG_SUPPRESS_MSVC(4996)       \
      THRUST_BEGIN_NV_DIAG_SUPPRESS(1444)
#    define THRUST_SUPPRESS_DEPRECATED_POP THRUST_NV_DIAG_POP() THRUST_DIAG_POP
#  elif THRUST_COMPILER(NVRTC)
#    define THRUST_SUPPRESS_DEPRECATED_PUSH THRUST_BEGIN_NV_DIAG_SUPPRESS(1444, 20199)
#    define THRUST_SUPPRESS_DEPRECATED_POP  THRUST_NV_DIAG_POP()
#  else // unknown compiler
#    define THRUST_SUPPRESS_DEPRECATED_PUSH
#    define THRUST_SUPPRESS_DEPRECATED_POP
#  endif // unknown compiler

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_DIAGNOSTIC_H
