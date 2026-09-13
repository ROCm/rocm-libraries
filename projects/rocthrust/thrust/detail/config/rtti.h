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

#ifndef CONFIG_RTTI_H
#define CONFIG_RTTI_H

#include <thrust/detail/config/device_system.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/rtti.h)
// clang-format on

#  ifdef _CCCL_NO_RTTI
#    define THRUST_NO_RTTI
#  endif // _CCCL_NO_RTTI

#else

#  include <thrust/detail/config/compiler.h>

#  if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#    pragma GCC system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#    pragma clang system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#    pragma system_header
#  endif // no system header

// NOTE: some compilers support the `typeid` feature but not the `dynamic_cast`
// feature. This is why we have separate macros for each.

#  ifndef THRUST_NO_RTTI
#    if defined(THRUST_DISABLE_RTTI) // Escape hatch for users to manually disable RTTI
#      define THRUST_NO_RTTI
#    elif defined(__INTEL_COMPILER)
#      if __RTTI == 0 && __INTEL_RTTI__ == 0 && __GXX_RTTI == 0 && _CPPRTTI == 0
#        define THRUST_NO_RTTI
#      endif
#    elif defined(_MSC_VER)
#      if _CPPRTTI == 0
#        define THRUST_NO_RTTI
#      endif
#    elif defined(__clang__)
#      if !__has_feature(cxx_rtti)
#        define THRUST_NO_RTTI
#      endif
#    else
#      if __GXX_RTTI == 0 && __cpp_rtti == 0
#        define THRUST_NO_RTTI
#      endif
#    endif
#  endif // !THRUST_NO_RTTI

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

#endif // CONFIG_RTTI_H
