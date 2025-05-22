//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef THRUST_SEQUENCE_ACCESS_H
#define THRUST_SEQUENCE_ACCESS_H

#pragma once

#include <thrust/detail/config.h>

// We need to define hidden friends for {cr,r,}{begin,end} of our containers as we will otherwise encounter ambiguities
#define THRUST_SYNTHESIZE_SEQUENCE_ACCESS(_ClassName, _ConstIter)                                                      \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE iterator begin(_ClassName& __sequence) noexcept(                          \
    noexcept(__sequence.begin()))                                                                                      \
  {                                                                                                                    \
    return __sequence.begin();                                                                                         \
  }                                                                                                                    \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstIter begin(const _ClassName& __sequence) noexcept(                  \
    noexcept(__sequence.begin()))                                                                                      \
  {                                                                                                                    \
    return __sequence.begin();                                                                                         \
  }                                                                                                                    \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE iterator end(_ClassName& __sequence) noexcept(noexcept(__sequence.end())) \
  {                                                                                                                    \
    return __sequence.end();                                                                                           \
  }                                                                                                                    \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstIter end(const _ClassName& __sequence) noexcept(                    \
    noexcept(__sequence.end()))                                                                                        \
  {                                                                                                                    \
    return __sequence.end();                                                                                           \
  }                                                                                                                    \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstIter cbegin(const _ClassName& __sequence) noexcept(                 \
    noexcept(__sequence.begin()))                                                                                      \
  {                                                                                                                    \
    return __sequence.begin();                                                                                         \
  }                                                                                                                    \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstIter cend(const _ClassName& __sequence) noexcept(                   \
    noexcept(__sequence.end()))                                                                                        \
  {                                                                                                                    \
    return __sequence.end();                                                                                           \
  }
#define THRUST_SYNTHESIZE_SEQUENCE_REVERSE_ACCESS(_ClassName, _ConstRevIter)                               \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE reverse_iterator rbegin(_ClassName& __sequence) noexcept(     \
    noexcept(__sequence.rbegin()))                                                                         \
  {                                                                                                        \
    return __sequence.rbegin();                                                                            \
  }                                                                                                        \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstRevIter rbegin(const _ClassName& __sequence) noexcept(  \
    noexcept(__sequence.rbegin()))                                                                         \
  {                                                                                                        \
    return __sequence.rbegin();                                                                            \
  }                                                                                                        \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE reverse_iterator rend(_ClassName& __sequence) noexcept(       \
    noexcept(__sequence.rend()))                                                                           \
  {                                                                                                        \
    return __sequence.rend();                                                                              \
  }                                                                                                        \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstRevIter rend(const _ClassName& __sequence) noexcept(    \
    noexcept(__sequence.rend()))                                                                           \
  {                                                                                                        \
    return __sequence.rend();                                                                              \
  }                                                                                                        \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstRevIter crbegin(const _ClassName& __sequence) noexcept( \
    noexcept(__sequence.rbegin()))                                                                         \
  {                                                                                                        \
    return __sequence.rbegin();                                                                            \
  }                                                                                                        \
  THRUST_NODISCARD_FRIEND THRUST_HOST_DEVICE _ConstRevIter crend(const _ClassName& __sequence) noexcept(   \
    noexcept(__sequence.rend()))                                                                           \
  {                                                                                                        \
    return __sequence.rend();                                                                              \
  }

#endif // THRUST_SEQUENCE_ACCESS_H
