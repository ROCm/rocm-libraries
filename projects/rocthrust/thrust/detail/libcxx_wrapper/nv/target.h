//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

// This header contains a preview of a portability system that enables
// CUDA C++ development with NVC++, NVCC, and supported host compilers.
// These interfaces are not guaranteed to be stable.

#ifndef LIBCXX_WRAPPER_NV_TARGET_H
#define LIBCXX_WRAPPER_NV_TARGET_H

// TODO(libhipcxx): remove this file and replace <thrust/detail/libcxx_wrapper/nv/target.h> with <nv/target>
// in rocThrust once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include <nv/target>
#else
#  include <thrust/detail/libcxx_wrapper/nv/detail/__target_macros.h>
#endif

#endif // LIBCXX_WRAPPER_NV_TARGET_H
