//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_NV_DETAIL__TARGET_MACROS_H
#define LIBCXX_WRAPPER_NV_DETAIL__TARGET_MACROS_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/libcxx_wrapper/nv/detail/__preprocessor.h>

#if defined(__GNUC__)
#  pragma GCC system_header
#endif

// General libcudacxx::nv macros.
#if defined(__HIP_DEVICE_COMPILE__)
// HIPCC being used, device dispatches allowed.
#  define NV_IS_HOST   0
#  define NV_IS_DEVICE 1
#else
// HIPCC not being used, only host dispatches allowed.
#  define NV_IS_HOST   1
#  define NV_IS_DEVICE 0
#endif
#define NV_ANY_TARGET NV_IS_HOST || NV_IS_DEVICE

// NV_IF_TARGET invoke mechanisms.
#define _NV_BLOCK_EXPAND(...)       {_NV_REMOVE_PAREN(__VA_ARGS__)}
#define _NV_TARGET_IF(cond, t, ...) _NV_IF(cond, t, __VA_ARGS__)

// NV_IF_TARGET supports a false statement provided as a variadic macro
#define NV_IF_TARGET(cond, ...)       _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, __VA_ARGS__))
#define NV_IF_ELSE_TARGET(cond, t, f) _NV_BLOCK_EXPAND(_NV_TARGET_IF(cond, t, f))

#endif // LIBCXX_WRAPPER_NV_DETAIL__TARGET_MACROS_H
