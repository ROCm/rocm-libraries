// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__FUNCTIONAL_IDENTITY_H
#define LIBCXX_WRAPPER_STD__FUNCTIONAL_IDENTITY_H

// TODO(libhipcxx): remove this file and replace ::internal with _THRUST_STD in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__functional/identity.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>

#  include <utility>
#endif

namespace internal
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD

using identity = _THRUST_STD::__identity;

#else

struct identity
{
  template <typename Tp>
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr Tp&& operator()(Tp&& t) const noexcept
  {
    return ::std::forward<Tp>(t);
  }

  using is_transparent = void;
};

#endif
} // namespace internal

#endif // LIBCXX_WRAPPER_STD__FUNCTIONAL_IDENTITY_H
