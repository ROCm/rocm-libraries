// -*- C++ -*-
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

#ifndef LIBCXX_WRAPPER_STD__EXCEPTION_TERMINATE_H
#define LIBCXX_WRAPPER_STD__EXCEPTION_TERMINATE_H

// TODO(libhipcxx): remove this file and replace ::internal with _THRUST_STD in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__exception/terminate.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/diagnostic.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/unreachable.h>

#  include <cstdlib>
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_STD_NOVERSION::terminate;

#else

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4702) // unreachable code

namespace detail
{

THRUST_NORETURN inline THRUST_HOST_DEVICE void cccl_terminate() noexcept
{
  _THRUST_IF_ELSE_TARGET(_THRUST_IS_HOST, (::std::exit(-1);), (__builtin_trap();))
  THRUST_UNREACHABLE();
}

} // namespace detail

THRUST_NORETURN inline THRUST_HOST_DEVICE void terminate() noexcept
{
  detail::cccl_terminate();
  THRUST_UNREACHABLE();
}

THRUST_DIAG_POP

#endif

} // namespace internal

#endif // LIBCXX_WRAPPER_STD__EXCEPTION_TERMINATE_H
