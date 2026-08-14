//===----------------------------------------------------------------------===//Add commentMore actions
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Add commentMore actions
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_DEPRECATED_H
#define LIBCXX_WRAPPER_STD__CCCL_DEPRECATED_H

// TODO(libhipcxx): remove this file and replace THRUST* with CCCL* in rocThrust once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__cccl/deprecated.h)
#endif

// Check for deprecation opt outs
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_DIALECT)
#  if !defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#    define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#  endif
#endif // suppress all dialect deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_14) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#  if !defined(THRUST_IGNORE_DEPRECATED_CPP_14)
#    define THRUST_IGNORE_DEPRECATED_CPP_14
#  endif
#endif // suppress all c++14 dialect deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_11) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT) \
  || defined(THRUST_IGNORE_DEPRECATED_CPP_14)
#  if !defined(THRUST_IGNORE_DEPRECATED_CPP_11)
#    define THRUST_IGNORE_DEPRECATED_CPP_11
#  endif
#endif // suppress all c++11 dialect deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_COMPILER) || defined(CCCL_IGNORE_DEPRECATED_COMPILER) \
  || defined(CUB_IGNORE_DEPRECATED_COMPILER) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)  \
  || defined(THRUST_IGNORE_DEPRECATED_CPP_14) || defined(THRUST_IGNORE_DEPRECATED_CPP_11)
#  if !defined(THRUST_IGNORE_DEPRECATED_COMPILER)
#    define THRUST_IGNORE_DEPRECATED_COMPILER
#  endif
#endif // suppress all compiler deprecation warnings
#if defined(LIBCUDACXX_IGNORE_DEPRECATED_API) || defined(CCCL_IGNORE_DEPRECATED_API) \
  || defined(CUB_IGNORE_DEPRECATED_API)
#  if !defined(THRUST_IGNORE_DEPRECATED_API)
#    define THRUST_IGNORE_DEPRECATED_API
#  endif
#endif // suppress all API deprecation warnings

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  define THRUST_DEPRECATED              CCCL_DEPRECATED
#  define THRUST_DEPRECATED_BECAUSE(MSG) CCCL_DEPRECATED_BECAUSE(MSG)

#else

#  ifdef THRUST_IGNORE_DEPRECATED_API
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG)
#  else // ^^^ THRUST_IGNORE_DEPRECATED_API ^^^ / vvv !THRUST_IGNORE_DEPRECATED_API vvv
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED              [[deprecated]]
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG) [[deprecated(MSG)]]
#  endif // !THRUST_IGNORE_DEPRECATED_API

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_DEPRECATED_H
