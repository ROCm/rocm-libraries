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

#ifndef THRUST_DETAIL_CONFIG_VISIBILITY_H
#define THRUST_DETAIL_CONFIG_VISIBILITY_H

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_FORCEINLINE __forceinline
#else // ^^^ THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC ^^^ / vvv THRUST_HOST_COMPILER !=
      // THRUST_HOST_COMPILER_MSVC vvv
#  define THRUST_FORCEINLINE __inline__ __attribute__((__always_inline__))
#endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC

#endif // THRUST_DETAIL_CONFIG_VISIBILITY_H
