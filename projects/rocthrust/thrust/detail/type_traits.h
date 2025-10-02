/*
 *  Copyright 2008-2022 NVIDIA Corporation
 *  Modifications Copyright© 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <rocprim/type_traits.hpp>
#  include <rocprim/type_traits_functions.hpp>
#endif // !_THRUST_HAS_DEVICE_SYSTEM_STD

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

using _THRUST_STD::is_arithmetic;
using _THRUST_STD::is_integral;
using _THRUST_STD::make_unsigned;

template <typename T>
using make_unsigned_t = _THRUST_STD::make_unsigned_t<T>;

template <typename Invokable, typename InputT, typename InitT = InputT>
using accumulator_t = _THRUST_STD::__accumulator_t<Invokable, InputT, InitT>;
template <typename T>
using decay_t = _THRUST_STD::decay_t<T>;
template <typename T>
using remove_cvref = _THRUST_STD::remove_cvref<T>;
template <typename T>
using remove_cvref_t = _THRUST_STD::remove_cvref_t<T>;
template <typename... Pred>
using _And = _THRUST_STD::_And<Pred...>;

#else // !_THRUST_HAS_DEVICE_SYSTEM_STD

// TODO: Use rocprim version of is_arithmetic, is_integral, make_unsigned and make_unsigned_t for consistency.
// However, replacing with rocprim::is_arithmetic currently causes issues.
// Keeping standard versions for now until compatibility is resolved.
using ::std::is_arithmetic;
using ::std::is_integral;
using ::std::make_unsigned;

template <typename T>
using make_unsigned_t = typename ::std::make_unsigned<T>::type;

template <typename Invokable, typename InputT, typename InitT = InputT>
using accumulator_t = ::rocprim::accumulator_t<Invokable, InputT, InitT>;
template <typename T>
// If we're not on Windows and we have libstdc++ >= 10, we can use the __decay_t
// builtin to reduce compilation time.
#  if defined(_WIN32) || (defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 10)
using decay_t = ::std::decay_t<T>;
#  else
using decay_t = ::std::__decay_t<T>;
#  endif
template <typename T>
using remove_cvref = ::std::remove_cv<::std::remove_reference_t<T>>;
template <typename T>
using remove_cvref_t = ::std::remove_cv_t<::std::remove_reference_t<T>>;

namespace detail
{
template <typename...>
using expand_to_true = ::std::true_type;
template <typename... Pred>
THRUST_HOST_DEVICE expand_to_true<::std::enable_if_t<Pred::value>...> and_helper(int);
template <typename...>
THRUST_HOST_DEVICE ::std::false_type and_helper(...);
} // namespace detail

template <typename... Pred>
#  if defined(__CUDA__) && THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG && defined(__has_attribute) \
    && __has_attribute(__nodebug__)
using _And __attribute__((__nodebug__)) = decltype(detail::and_helper<Pred...>(0));
#  else
using _And = decltype(detail::and_helper<Pred...>(0));
#  endif

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

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
struct is_non_bool_integral : public ::internal::is_integral<T>
{};
template <>
struct is_non_bool_integral<bool> : public false_type
{};

template <typename T>
struct is_non_bool_arithmetic : public ::internal::is_arithmetic<T>
{};

template <typename T>
struct is_unbounded_array : public thrust::detail::false_type
{};
template <typename T>
struct is_unbounded_array<T[]> : public thrust::detail::true_type
{};

template <typename T>
struct is_bounded_array : public thrust::detail::false_type
{};
template <typename T, size_t N>
struct is_bounded_array<T[N]> : public thrust::detail::true_type
{};

template <>
struct is_non_bool_arithmetic<bool> : public false_type
{};

template <typename T>
struct is_proxy_reference : public false_type
{};

template <typename Boolean>
struct not_ : public integral_constant<bool, !Boolean::value>
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

template <typename T>
//  struct identity
//  XXX WAR nvcc's confusion with thrust::identity
struct identity_
{
  using type = T;
}; // end identity

template <class Tp, bool>
struct dependent_type
{
  using type = Tp;
}; // end dependent_type

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
    : thrust::detail::eval_if<(sizeof(T2) > sizeof(T1)), thrust::detail::identity_<T2>, thrust::detail::identity_<T1>>
{};

template <class F, class... Us>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
using invoke_result = _THRUST_STD::__invoke_of<F, Us...>;
#else // !_THRUST_HAS_DEVICE_SYSTEM_STD
using invoke_result = ::rocprim::invoke_result<F, Us...>;
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

template <class F, class... Us>
using invoke_result_t = typename invoke_result<F, Us...>::type;
} // namespace detail

using detail::false_type;
using detail::integral_constant;
using detail::true_type;

THRUST_NAMESPACE_END

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename... Tp>
using promoted_numerical_type = _THRUST_STD::common_type<Tp...>;

#else // !_THRUST_HAS_DEVICE_SYSTEM_STD

template <typename T1, typename T2, typename Enable = void>
struct promoted_numerical_type;

template <typename T1, typename T2>
struct promoted_numerical_type<
  T1,
  T2,
  typename _THRUST_STD::enable_if<_And<typename _THRUST_STD::is_floating_point<T1>::type,
                                       typename _THRUST_STD::is_floating_point<T2>::type>::value>::type>
{
  using type = typename ::thrust::detail::larger_type<T1, T2>::type;
};

template <typename T1, typename T2>
struct promoted_numerical_type<
  T1,
  T2,
  typename _THRUST_STD::enable_if<
    _And<typename is_integral<T1>::type, typename _THRUST_STD::is_floating_point<T2>::type>::value>::type>
{
  using type = T2;
};

template <typename T1, typename T2>
struct promoted_numerical_type<
  T1,
  T2,
  typename _THRUST_STD::enable_if<
    _And<typename _THRUST_STD::is_floating_point<T1>::type, typename is_integral<T2>::type>::value>::type>
{
  using type = T1;
};

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

} // namespace internal
