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

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/std/type_traits>
#else
#  include <rocprim/type_traits.hpp>

#  include <type_traits>
#endif // THRUST_DEVICE_SYSTEM

namespace internal
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using ::cuda::std::_And;
#else
template <typename...>
using expand_to_true = ::std::true_type;
template <typename... Pred>
THRUST_HOST_DEVICE expand_to_true<::std::enable_if_t<Pred::value>...> and_helper(int);
template <typename...>
THRUST_HOST_DEVICE ::std::false_type and_helper(...);
template <typename... Pred>
#  if defined(__CUDA__) && THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG && defined(__has_attribute) \
    && __has_attribute(__nodebug__)
using _And __attribute__((__nodebug__)) = decltype(and_helper<Pred...>(0));
#  else
using _And = decltype(and_helper<Pred...>(0));
#  endif
#endif
} // namespace internal

THRUST_NAMESPACE_BEGIN

// forward declaration of device_reference
template <typename T>
class device_reference;

namespace detail
{
/// helper classes [4.3].
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using ::cuda::std::add_const;
using ::cuda::std::add_cv;
using ::cuda::std::add_volatile;
using ::cuda::std::conditional;
using ::cuda::std::enable_if;
using ::cuda::std::false_type;
using ::cuda::std::integral_constant;
using ::cuda::std::is_arithmetic;
using ::cuda::std::is_assignable;
using ::cuda::std::is_base_of;
using ::cuda::std::is_const;
using ::cuda::std::is_convertible;
using ::cuda::std::is_copy_assignable;
using ::cuda::std::is_empty;
using ::cuda::std::is_floating_point;
using ::cuda::std::is_integral;
using ::cuda::std::is_pointer;
using ::cuda::std::is_reference;
using ::cuda::std::is_same;
using ::cuda::std::is_void;
using ::cuda::std::is_volatile;
using ::cuda::std::make_unsigned;
using ::cuda::std::remove_const;
using ::cuda::std::remove_cv;
using ::cuda::std::remove_reference;
using ::cuda::std::remove_volatile;
using ::cuda::std::true_type;
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
using ::std::add_const;
using ::std::add_cv;
using ::std::add_volatile;
using ::std::conditional;
using ::std::enable_if;
using ::std::false_type;
using ::std::integral_constant;
using ::std::is_arithmetic;
using ::std::is_assignable;
using ::std::is_base_of;
using ::std::is_const;
using ::std::is_convertible;
using ::std::is_copy_assignable;
using ::std::is_empty;
using ::std::is_floating_point;
using ::std::is_integral;
using ::std::is_pointer;
using ::std::is_reference;
using ::std::is_same;
using ::std::is_void;
using ::std::is_volatile;
using ::std::make_unsigned;
using ::std::remove_const;
using ::std::remove_cv;
using ::std::remove_reference;
using ::std::remove_volatile;
using ::std::true_type;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

template <typename T>
struct is_device_ptr : public false_type
{};

template <typename T>
struct is_non_bool_integral : public is_integral<T>
{};
template <>
struct is_non_bool_integral<bool> : public false_type
{};

template <typename T>
struct is_non_bool_arithmetic : public is_arithmetic<T>
{};
template <>
struct is_non_bool_arithmetic<bool> : public false_type
{};

template <typename T>
struct is_pod
    : public integral_constant<bool,
                               is_void<T>::value || is_pointer<T>::value
                                 || is_arithmetic<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
                                 // use intrinsic type traits
                                 || __is_pod(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#  if (__GNUC__ * 100 + __GNUC_MINOR__ >= 403)
                                 || __is_pod(T)
#  endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
                               >
{};

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
struct has_trivial_constructor
    : public integral_constant<bool, is_pod<T>::value || ::cuda::std::is_trivially_constructible<T>::value>
{};

template <typename T>
struct has_trivial_copy_constructor
    : public integral_constant<bool, is_pod<T>::value || ::cuda::std::is_trivially_copyable<T>::value>
{};
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
struct has_trivial_constructor
    : public integral_constant<bool,
                               is_pod<T>::value
#  if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
                                 || __is_trivially_constructible(T)
#  elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#    if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
                                 || __is_trivially_constructible(T)
#    endif // GCC VERSION
#  endif // THRUST_HOST_COMPILER
                               >
{};

template <typename T>
struct has_trivial_copy_constructor
    : public integral_constant<bool,
                               is_pod<T>::value
#  if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
                                 || __is_trivially_copyable(T)
#  elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#    if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
                                 || __is_trivially_copyable(T)
#    endif // GCC VERSION
#  endif // THRUST_HOST_COMPILER
                               >
{};
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

template <typename T>
struct has_trivial_destructor : public is_pod<T>
{};

template <typename T>
struct is_proxy_reference : public false_type
{};

template <typename T>
struct is_device_reference : public false_type
{};
template <typename T>
struct is_device_reference<thrust::device_reference<T>> : public true_type
{};

// NB: Careful with reference to void.
template <typename _Tp, bool = (is_void<_Tp>::value || is_reference<_Tp>::value)>
struct __add_reference_helper
{
  using type = _Tp&;
};

template <typename _Tp>
struct __add_reference_helper<_Tp, true>
{
  using type = _Tp;
};

template <typename _Tp>
struct add_reference : public __add_reference_helper<_Tp>
{};

template <typename T1, typename T2>
struct lazy_is_same : is_same<typename T1::type, typename T2::type>
{}; // end lazy_is_same

template <typename T1, typename T2>
struct is_different : public true_type
{}; // end is_different

template <typename T>
struct is_different<T, T> : public false_type
{}; // end is_different

template <typename T1, typename T2>
struct lazy_is_different : is_different<typename T1::type, typename T2::type>
{}; // end lazy_is_different

template <typename T1, typename T2>
struct is_one_convertible_to_the_other
    : public integral_constant<bool, is_convertible<T1, T2>::value || is_convertible<T2, T1>::value>
{};

// mpl stuff
template <typename... Conditions>
struct or_;

template <>
struct or_<>
    : public integral_constant<bool,
                               false_type::value // identity for or_
                               >
{}; // end or_

template <typename Condition, typename... Conditions>
struct or_<Condition, Conditions...> : public integral_constant<bool, Condition::value || or_<Conditions...>::value>
{}; // end or_

template <typename... Conditions>
struct and_;

template <>
struct and_<>
    : public integral_constant<bool,
                               true_type::value // identity for and_
                               >
{}; // end and_

template <typename Condition, typename... Conditions>
struct and_<Condition, Conditions...> : public integral_constant<bool, Condition::value && and_<Conditions...>::value>
{}; // end and_

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

template <bool, typename T>
struct lazy_enable_if
{};
template <typename T>
struct lazy_enable_if<true, T>
{
  using type = typename T::type;
};

template <bool condition, typename T = void>
struct disable_if : enable_if<!condition, T>
{};
template <bool condition, typename T>
struct lazy_disable_if : lazy_enable_if<!condition, T>
{};

template <typename T1, typename T2, typename T = void>
using enable_if_convertible = enable_if<is_convertible<T1, T2>::value, T>;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T1, typename T2, typename T = void>
using enable_if_convertible_t = ::cuda::std::enable_if_t<is_convertible<T1, T2>::value, T>;
#else
template <typename T1, typename T2, typename T = void>
using enable_if_convertible_t = ::std::enable_if_t<is_convertible<T1, T2>::value, T>;
#endif

template <typename T1, typename T2, typename T = void>
struct disable_if_convertible : disable_if<is_convertible<T1, T2>::value, T>
{};

template <typename T1, typename T2, typename Result = void>
struct enable_if_different : enable_if<is_different<T1, T2>::value, Result>
{};

template <typename T>
struct is_numeric : ::internal::_And<is_convertible<int, T>, is_convertible<T, int>>
{}; // end is_numeric

template <typename>
struct is_reference_to_const : false_type
{};
template <typename T>
struct is_reference_to_const<const T&> : true_type
{};

struct largest_available_float
{
  using type = double;
};

// T1 wins if they are both the same size
template <typename T1, typename T2>
struct larger_type
    : thrust::detail::eval_if<(sizeof(T2) > sizeof(T1)), thrust::detail::identity_<T2>, thrust::detail::identity_<T1>>
{};

template <typename Base, typename Derived, typename Result = void>
struct enable_if_base_of : enable_if<is_base_of<Base, Derived>::value, Result>
{};

template <typename T1, typename T2, typename Enable = void>
struct promoted_numerical_type;

template <typename T1, typename T2>
struct promoted_numerical_type<
  T1,
  T2,
  typename enable_if<and_<typename is_floating_point<T1>::type, typename is_floating_point<T2>::type>::value>::type>
{
  using type = typename larger_type<T1, T2>::type;
};

template <typename T1, typename T2>
struct promoted_numerical_type<
  T1,
  T2,
  typename enable_if<and_<typename is_integral<T1>::type, typename is_floating_point<T2>::type>::value>::type>
{
  using type = T2;
};

template <typename T1, typename T2>
struct promoted_numerical_type<
  T1,
  T2,
  typename enable_if<and_<typename is_floating_point<T1>::type, typename is_integral<T2>::type>::value>::type>
{
  using type = T1;
};

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <class F, class... Us>
using invoke_result = ::cuda::std::__invoke_of<F, Us...>;
#else
using ::rocprim::invoke_result;
#endif

template <class F, class... Us>
using invoke_result_t = typename invoke_result<F, Us...>::type;
} // namespace detail

using detail::false_type;
using detail::integral_constant;
using detail::true_type;

THRUST_NAMESPACE_END

#include <thrust/detail/type_traits/has_trivial_assign.h>
