//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_PREPROCESSOR_H
#define CONFIG_PREPROCESSOR_H

#ifdef __has_include
#  define THRUST_HAS_INCLUDE(_X) __has_include(_X)
#else
#  define THRUST_HAS_INCLUDE(_X) 0
#endif

#endif // CONFIG_PREPROCESSOR_H
