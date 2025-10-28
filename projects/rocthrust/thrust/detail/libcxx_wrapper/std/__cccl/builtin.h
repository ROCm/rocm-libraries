//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_BUILTIN_H
#define LIBCXX_WRAPPER_STD__CCCL_BUILTIN_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if !_THRUST_HAS_DEVICE_SYSTEM_STD

#  ifdef __has_feature
#    define THRUST_HAS_FEATURE(__x) __has_feature(__x)
#  else // ^^^ __has_feature ^^^ / vvv !__has_feature vvv
#    define THRUST_HAS_FEATURE(__x) 0
#  endif // !__has_feature

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_BUILTIN_H
