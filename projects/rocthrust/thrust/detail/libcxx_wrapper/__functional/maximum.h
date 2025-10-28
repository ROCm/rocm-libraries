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

#ifndef LIBCXX_WRAPPER_FUNCTIONAL_MAXIMUM_H
#define LIBCXX_WRAPPER_FUNCTIONAL_MAXIMUM_H

// TODO(libhipcxx): remove this file and replace ::internal with _THRUST_STD in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(__functional/maximum.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>
#  include <thrust/detail/libcxx_wrapper/std/detail/libcxx/include/__config.h>

#  include <type_traits>
#endif

namespace internal
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_LIBCXX::maximum;

#else

template <typename Tp = void>
struct maximum
{
  THRUST_EXEC_CHECK_DISABLE
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr Tp operator()(const Tp& lhs, const Tp& rhs) const
    noexcept(noexcept((lhs < rhs) ? rhs : lhs))
  {
    return (lhs < rhs) ? rhs : lhs;
  }
};
THRUST_CTAD_SUPPORTED_FOR_TYPE(maximum);

template <>
struct maximum<void>
{
  THRUST_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr ::std::common_type_t<T1, T2>
  operator()(const T1& lhs, const T2& rhs) const noexcept(noexcept((lhs < rhs) ? rhs : lhs))
  {
    return (lhs < rhs) ? rhs : lhs;
  }
};

#endif
} // namespace internal

#endif // LIBCXX_WRAPPER_FUNCTIONAL_MAXIMUM_H
