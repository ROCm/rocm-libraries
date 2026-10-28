/*
 *  Copyright 2008-2022 NVIDIA Corporation
 *  Modifications Copyright© 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include _THRUST_STD_INCLUDE(type_traits)

#include <thrust/detail/libcxx_wrapper/std/__type_traits/conjunction.h>

// TODO(libhipcxx): remove this namespace once libhipcxx gets ready
namespace internal
{
using _THRUST_STD::is_arithmetic;
using _THRUST_STD::make_unsigned;

template <typename T>
using make_unsigned_t = _THRUST_STD::make_unsigned_t<T>;

template <typename T>
using decay_t = _THRUST_STD::decay_t<T>;
template <typename T>
using remove_cvref = _THRUST_STD::remove_cvref<T>;
template <typename T>
using remove_cvref_t = _THRUST_STD::remove_cvref_t<T>;
} // namespace internal

THRUST_NAMESPACE_BEGIN

// forward declaration of device_reference
template <typename T>
class device_reference;

namespace detail
{
/// helper classes [4.3].
template <typename T, T v>
using integral_constant = _THRUST_STD::integral_constant<T, v>;
using true_type         = _THRUST_STD::true_type;
using false_type        = _THRUST_STD::false_type;

template <typename T>
struct is_non_bool_integral : public _THRUST_STD::is_integral<T>
{};
template <>
struct is_non_bool_integral<bool> : public false_type
{};

template <typename T>
struct is_non_bool_arithmetic : public ::internal::is_arithmetic<T>
{};
template <>
struct is_non_bool_arithmetic<bool> : public false_type
{};

template <typename T>
inline constexpr bool is_proxy_reference_v = false;

template <typename Boolean>
struct not_ : public _THRUST_STD::integral_constant<bool, !Boolean::value>
{}; // end not_

template <bool, typename Then, typename Else>
struct eval_if
{}; // end eval_if

template <typename Then, typename Else>
struct eval_if<true, Then, Else>
{
  using type = typename Then::type;
}; // end eval_if

template <typename Then, typename Else>
struct eval_if<false, Then, Else>
{
  using type = typename Else::type;
}; // end eval_if

template <bool, typename T>
struct lazy_enable_if
{};
template <typename T>
struct lazy_enable_if<true, T>
{
  using type = typename T::type;
};

template <bool condition, typename T = void>
struct disable_if : _THRUST_STD::enable_if<!condition, T>
{};
template <bool condition, typename T>
struct lazy_disable_if : lazy_enable_if<!condition, T>
{};

template <typename T1, typename T2, typename T = void>
using enable_if_convertible_t = _THRUST_STD::enable_if_t<_THRUST_STD::is_convertible<T1, T2>::value, T>;

template <typename T1, typename T2, typename T = void>
struct disable_if_convertible : disable_if<_THRUST_STD::is_convertible<T1, T2>::value, T>
{};

template <typename T>
struct is_numeric : ::internal::_And<_THRUST_STD::is_convertible<int, T>, _THRUST_STD::is_convertible<T, int>>
{}; // end is_numeric

struct largest_available_float
{
  using type = double;
};

// T1 wins if they are both the same size
template <typename T1, typename T2>
struct larger_type
    : thrust::detail::eval_if<(sizeof(T2) > sizeof(T1)), _THRUST_STD::type_identity<T2>, _THRUST_STD::type_identity<T1>>
{};

template <class F, class... Us>
using invoke_result = _THRUST_STD::invoke_result<F, Us...>;

template <class F, class... Us>
using invoke_result_t = typename invoke_result<F, Us...>::type;

// [NON-CCCL PARITY BEGIN]
template <typename T>
struct is_unbounded_array : public _THRUST_STD::false_type
{};
template <typename T>
struct is_unbounded_array<T[]> : public _THRUST_STD::true_type
{};

template <typename T>
struct is_bounded_array : public _THRUST_STD::false_type
{};
template <typename T, _THRUST_STD::size_t N>
struct is_bounded_array<T[N]> : public _THRUST_STD::true_type
{};

template <class Tp, bool>
struct dependent_type
{
  using type = Tp;
};
// [NON-CCCL PARITY END]

} // namespace detail

using detail::false_type;
using detail::integral_constant;
using detail::true_type;

THRUST_NAMESPACE_END
