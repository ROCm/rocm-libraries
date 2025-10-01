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

#if _THRUST_HAS_DEVICE_SYSTEM_STD
// clang-format off
#  include _THRUST_LIBCXX_INCLUDE(__functional/address_stability.h)
// clang-format on
#else
#  include <functional>
#  include <type_traits>
#  include <utility>
#endif

THRUST_NAMESPACE_BEGIN

namespace detail
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_LIBCXX::__has_builtin_operators;
using _THRUST_LIBCXX::proclaim_copyable_arguments;
using _THRUST_LIBCXX::proclaims_copyable_arguments;
#  define THRUST_MARK_CAN_COPY_ARGUMENTS(functor)                                              \
    /*we know what plus<T> etc. does if T is not a type that could have a weird operatorX() */ \
    template <typename T>                                                                      \
    struct proclaims_copyable_arguments<functor<T>> : __has_builtin_operators<T>               \
    {};                                                                                        \
    /*we do not know what plus<void> etc. does, which depends on the types it is invoked on */ \
    template <>                                                                                \
    struct proclaims_copyable_arguments<functor<void>> : _THRUST_STD::false_type               \
    {};

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
//! object. Some algorithms, like thrust::transform, can benefit from this information and choose a more efficient
//! implementation.
//! @see proclaims_copyable_arguments
template <typename F>
THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr auto proclaim_copyable_arguments(F&& f)
  -> callable_permitting_copied_arguments<::std::decay_t<F>>
{
  return {::std::forward<F>(f)};
}

// Specializations for libcu++ function objects are provided here to not pull this include into `<cuda/std/...>` headers

template <typename T>
struct has_builtin_operators
    : ::std::bool_constant<!::std::is_class_v<T> && !::std::is_enum_v<T> && !::std::is_void_v<T>>
{};

#  define THRUST_MARK_CAN_COPY_ARGUMENTS(functor)                                              \
    /*we know what plus<T> etc. does if T is not a type that could have a weird operatorX() */ \
    template <typename T>                                                                      \
    struct proclaims_copyable_arguments<functor<T>> : has_builtin_operators<T>                 \
    {};                                                                                        \
    /*we do not know what plus<void> etc. does, which depends on the types it is invoked on */ \
    template <>                                                                                \
    struct proclaims_copyable_arguments<functor<void>> : ::std::false_type                     \
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

} // namespace detail

THRUST_NAMESPACE_END

#endif // DETAIL_FUNCTIONAL_ADDRESS_STABILITY_H
