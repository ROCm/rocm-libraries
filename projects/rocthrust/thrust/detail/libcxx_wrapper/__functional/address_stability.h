//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_FUNCTIONAL_ADDRESS_STABILITY_H
#define LIBCXX_WRAPPER_FUNCTIONAL_ADDRESS_STABILITY_H

// TODO(libhipcxx): remove this file, check all the usage of proclaim_copyable_arguments and
// proclaims_copyable_arguments, and replace ::thrust::detail::* with _THRUST_LIBCXX::* in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(__functional/address_stability.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/dialect.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>
#  include <thrust/detail/libcxx_wrapper/std/__functional/not_fn.h>
#  include <thrust/detail/type_traits.h>

#  include <functional>
#  include <type_traits>
#  include <utility>
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_LIBCXX::proclaim_copyable_arguments;
using _THRUST_LIBCXX::proclaims_copyable_arguments;

#else

template <typename F, typename SFINAE = void>
struct proclaims_copyable_arguments : ::std::false_type
{};

namespace detail
{
template <typename F>
struct callable_permitting_copied_arguments : F
{
  using F::operator();
};
} // namespace detail

template <typename F>
struct proclaims_copyable_arguments<detail::callable_permitting_copied_arguments<F>> : ::std::true_type
{};

template <typename F>
THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr auto
proclaim_copyable_arguments(F&& f) -> detail::callable_permitting_copied_arguments<::internal::decay_t<F>>
{
  return {::std::forward<F>(f)};
}

template <typename Fn>
struct proclaims_copyable_arguments<not_fn_t<Fn>> : proclaims_copyable_arguments<Fn>
{};

namespace detail
{
template <typename Tp>
struct has_builtin_operators
    : ::std::bool_constant<!THRUST_TRAIT(::std::is_class, Tp) && !THRUST_TRAIT(::std::is_enum, Tp)
                           && !THRUST_TRAIT(::std::is_void, Tp)>
{};
} // namespace detail

#  define THRUST_MARK_CAN_COPY_ARGUMENTS(functor)                                        \
    template <typename Tp>                                                               \
    struct proclaims_copyable_arguments<functor<Tp>> : detail::has_builtin_operators<Tp> \
    {};                                                                                  \
    template <>                                                                          \
    struct proclaims_copyable_arguments<functor<void>> : ::std::false_type               \
    {};

THRUST_MARK_CAN_COPY_ARGUMENTS(::std::plus);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::minus);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::multiplies);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::divides);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::modulus);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::negate);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::bit_and);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::bit_not);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::bit_or);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::bit_xor);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::equal_to);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::not_equal_to);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::less);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::less_equal);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::greater_equal);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::greater);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::logical_and);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::logical_not);
THRUST_MARK_CAN_COPY_ARGUMENTS(::std::logical_or);

#endif

} // namespace internal

#endif // LIBCXX_WRAPPER_FUNCTIONAL_ADDRESS_STABILITY_H
