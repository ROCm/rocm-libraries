//===----------------------------------------------------------------------===//Add commentMore actions
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
// Add commentMore actions
//===----------------------------------------------------------------------===//

#ifndef THRUST_DETAIL_CONFIG_DEPRECATED_H
#define THRUST_DETAIL_CONFIG_DEPRECATED_H

#include <thrust/detail/config/cpp_dialect.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(LIBCUDACXX_IGNORE_DEPRECATED_API) || defined(CCCL_IGNORE_DEPRECATED_API) \
  || defined(CUB_IGNORE_DEPRECATED_API)
#  if !defined(THRUST_IGNORE_DEPRECATED_API)
#    define THRUST_IGNORE_DEPRECATED_API
#  endif
#endif // suppress all API deprecation warnings

#ifdef THRUST_IGNORE_DEPRECATED_API
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED_BECAUSE(MSG)
#elif THRUST_CPP_DIALECT >= 2017
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED              [[deprecated]]
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED_BECAUSE(MSG) [[deprecated(MSG)]]
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED              __declspec(deprecated)
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED_BECAUSE(MSG) __declspec(deprecated(MSG))
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED              __attribute__((deprecated))
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED_BECAUSE(MSG) __attribute__((deprecated(MSG)))
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED              __attribute__((deprecated))
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED_BECAUSE(MSG) __attribute__((deprecated(MSG)))
#else
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED
//! deprecated [Since 2.8]
#  define THRUST_DEPRECATED_BECAUSE(MSG)
#endif

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 3))
#  define THRUST_ALIAS_ATTRIBUTE(...)
#else
#  define THRUST_ALIAS_ATTRIBUTE(...) __VA_ARGS__
#endif

#endif // THRUST_DETAIL_CONFIG_DEPRECATED_H
