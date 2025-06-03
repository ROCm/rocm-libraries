//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef DETAIL_FUNCTIONAL_ADDRESS_STABILITY_H
#define DETAIL_FUNCTIONAL_ADDRESS_STABILITY_H

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/std/__functional/address_stability.h>
#else
#  include <functional>
#  include <type_traits>
#  include <utility>
#endif

THRUST_NAMESPACE_BEGIN

namespace detail
{

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

using ::cuda::proclaim_copyable_arguments;
using ::cuda::proclaims_copyable_arguments;

#else

//! Trait telling whether a function object type F does not rely on the memory addresses of its arguments. The nested
//! value is true when the addresses of the arguments do not matter and arguments can be provided from arbitrary copies
//! of the respective sources. This trait can be specialized for custom function objects types.
//! @see proclaim_copyable_arguments
template <typename F, typename SFINAE = void>
struct proclaims_copyable_arguments : ::std::false_type
{};

template <typename F, typename... Args>
inline constexpr bool proclaims_copyable_arguments_v = proclaims_copyable_arguments<F, Args...>::value;

// Wrapper for a callable to mark it as permitting copied arguments
template <typename F>
struct callable_permitting_copied_arguments : F
{
  using F::operator();
};

template <typename F>
struct proclaims_copyable_arguments<callable_permitting_copied_arguments<F>> : ::std::true_type
{};

//! Creates a new function object from an existing one, which is marked as permitting its arguments to be copies of
//! whatever source they come from. This implies that the addresses of the arguments are irrelevant to the function
//! object.
//! @see proclaims_copyable_arguments
template <typename F>
THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr auto proclaim_copyable_arguments(F f)
  -> callable_permitting_copied_arguments<F>
{
  return callable_permitting_copied_arguments<F>{::std::move(f)};
}

#endif

} // namespace detail

THRUST_NAMESPACE_END

#endif // DETAIL_FUNCTIONAL_ADDRESS_STABILITY_H
