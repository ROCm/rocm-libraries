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

#ifndef CONFIG_DEPRECATED_H
#define CONFIG_DEPRECATED_H

// TODO(libhipcxx): remove this file and replace THRUST_DEPRECATED, THRUST_DEPRECATED_BECAUSE and
// THRUST_DEPRECATED_IN_CXX11 with CCCL_DEPRECATED, CCCL_DEPRECATED_BECAUSE and _LIBCUDACXX_DEPRECATED_IN_CXX11 in
// rocThrust once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/deprecated.h)
#  include _THRUST_STD_INCLUDE(detail/libcxx/include/__config)
// clang-format on

#  define THRUST_DEPRECATED              CCCL_DEPRECATED
#  define THRUST_DEPRECATED_BECAUSE(MSG) CCCL_DEPRECATED_BECAUSE(MSG)
#  define THRUST_DEPRECATED_IN_CXX11     _LIBCUDACXX_DEPRECATED_IN_CXX11

#else

#  include <thrust/detail/config/compiler.h>
#  include <thrust/detail/config/cpp_dialect.h>

#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_API) || defined(CCCL_IGNORE_DEPRECATED_API) \
    || defined(CUB_IGNORE_DEPRECATED_API)
#    if !defined(THRUST_IGNORE_DEPRECATED_API)
#      define THRUST_IGNORE_DEPRECATED_API
#    endif
#  endif // suppress all API deprecation warnings

#  ifdef THRUST_IGNORE_DEPRECATED_API
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG)
#  elif THRUST_CPP_DIALECT >= 2014
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED              [[deprecated]]
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG) [[deprecated(MSG)]]
#  elif THRUST_COMPILER(MSVC)
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED              __declspec(deprecated)
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG) __declspec(deprecated(MSG))
#  elif THRUST_COMPILER(CLANG) || THRUST_COMPILER(HIP)
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED              __attribute__((deprecated))
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG) __attribute__((deprecated(MSG)))
#  elif THRUST_COMPILER(GCC)
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED              __attribute__((deprecated))
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG) __attribute__((deprecated(MSG)))
#  else
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED
//! deprecated [Since 2.8]
#    define THRUST_DEPRECATED_BECAUSE(MSG)
#  endif

#  define THRUST_DEPRECATED_IN_CXX11 THRUST_DEPRECATED

#endif

#endif // CONFIG_DEPRECATED_H
