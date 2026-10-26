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

#ifndef LIBCXX_WRAPPER_STD_DETAIL_LIBCXX_INCLUDE__CONFIG_H
#define LIBCXX_WRAPPER_STD_DETAIL_LIBCXX_INCLUDE__CONFIG_H

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(detail/libcxx/include/__config)

#  define THRUST_CTAD_SUPPORTED_FOR_TYPE(_ClassName) _LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(_ClassName)

#else

#  if (!THRUST_COMPILER(GCC) || THRUST_COMPILER(GCC, >, 6))
#    define THRUST_CTAD_SUPPORTED_FOR_TYPE(_ClassName) \
      template <typename... Tag>                       \
      _ClassName(typename Tag::__allow_ctad...)->_ClassName<Tag...>
#  else
#    define THRUST_CTAD_SUPPORTED_FOR_TYPE(_ClassName) static_assert(true, "")
#  endif

#endif

#endif // LIBCXX_WRAPPER_STD_DETAIL_LIBCXX_INCLUDE__CONFIG_H
