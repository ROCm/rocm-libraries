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

#ifndef LIBCXX_WRAPPER_STD_DETAIL_LIBCXX_INCLUDE_STDEXCEPT_H
#define LIBCXX_WRAPPER_STD_DETAIL_LIBCXX_INCLUDE_STDEXCEPT_H

// TODO(libhipcxx): remove this file and replace ::internal* with _THRUST_STD* in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(detail/libcxx/include/stdexcept)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/exceptions.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>
#  include <thrust/detail/libcxx_wrapper/std/__exception/terminate.h>

#  ifndef THRUST_NO_EXCEPTIONS
#    include <stdexcept>
#  endif // THRUST_NO_EXCEPTIONS
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_STD::__throw_runtime_error;

#else

THRUST_NORETURN inline THRUST_HOST_DEVICE void __throw_runtime_error(const char* msg)
{
#  ifndef THRUST_NO_EXCEPTIONS
  _THRUST_IF_ELSE_TARGET(_THRUST_IS_HOST, (throw ::std::runtime_error(msg);), ((void) msg; terminate();))
#  else // ^^^ !THRUST_NO_EXCEPTIONS ^^^ / vvv THRUST_NO_EXCEPTIONS vvv
  (void) msg;
  terminate();
#  endif // THRUST_NO_EXCEPTIONS
}

#endif

} // namespace internal

#endif // LIBCXX_WRAPPER_STD_DETAIL_LIBCXX_INCLUDE_STDEXCEPT_H
