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

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#include <rocprim/type_traits.hpp>
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda/std/type_traits>
#elif defined(__has_include)
#if __has_include(<cuda/std/type_traits>)
#include <cuda/std/type_traits>
#endif // __has_include
#endif // THRUST_DEVICE_SYSTEM

#include <type_traits>

THRUST_NAMESPACE_BEGIN

// forward declaration of device_reference
template<typename T> class device_reference;

namespace detail
{
 /// helper classes [4.3].
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template<typename T, T v>
using integral_constant = ::cuda::std::integral_constant<T, v>;
using true_type  = ::cuda::std::true_type;
using false_type = ::cuda::std::false_type;
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
template<typename T, T v>
  struct integral_constant
  {
    THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT T value = v;

    using value_type = T;
    using type       = integral_constant<T, v>;

    // We don't want to switch to std::integral_constant, because we want access
    // to the C++14 operator(), but we'd like standard traits to interoperate
    // with our version when tag dispatching.
    integral_constant() = default;

    integral_constant(integral_constant const&) = default;

    integral_constant& operator=(integral_constant const&) = default;

    constexpr THRUST_HOST_DEVICE
    integral_constant(std::integral_constant<T, v>) noexcept {}

    constexpr THRUST_HOST_DEVICE operator value_type() const noexcept { return value; }
    constexpr THRUST_HOST_DEVICE value_type operator()() const noexcept { return value; }
  };

/// typedef for true_type
using true_type = integral_constant<bool, true>;

/// typedef for true_type
using false_type = integral_constant<bool, false>;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

//template<typename T> struct is_integral : public std::tr1::is_integral<T> {};
template<typename T> struct is_integral                           : public false_type {};
template<>           struct is_integral<bool>                     : public true_type {};
template<>           struct is_integral<char>                     : public true_type {};
template<>           struct is_integral<signed char>              : public true_type {};
template<>           struct is_integral<unsigned char>            : public true_type {};
template<>           struct is_integral<short>                    : public true_type {};
template<>           struct is_integral<unsigned short>           : public true_type {};
template<>           struct is_integral<int>                      : public true_type {};
template<>           struct is_integral<unsigned int>             : public true_type {};
template<>           struct is_integral<long>                     : public true_type {};
template<>           struct is_integral<unsigned long>            : public true_type {};
template<>           struct is_integral<long long>                : public true_type {};
template<>           struct is_integral<unsigned long long>       : public true_type {};
template<>           struct is_integral<const bool>               : public true_type {};
template<>           struct is_integral<const char>               : public true_type {};
template<>           struct is_integral<const unsigned char>      : public true_type {};
template<>           struct is_integral<const short>              : public true_type {};
template<>           struct is_integral<const unsigned short>     : public true_type {};
template<>           struct is_integral<const int>                : public true_type {};
template<>           struct is_integral<const unsigned int>       : public true_type {};
template<>           struct is_integral<const long>               : public true_type {};
template<>           struct is_integral<const unsigned long>      : public true_type {};
template<>           struct is_integral<const long long>          : public true_type {};
template<>           struct is_integral<const unsigned long long> : public true_type {};

template<typename T> struct is_floating_point              : public false_type {};
template<>           struct is_floating_point<float>       : public true_type {};
template<>           struct is_floating_point<double>      : public true_type {};
template<>           struct is_floating_point<long double> : public true_type {};

template<typename T> struct is_arithmetic               : public is_integral<T> {};
template<>           struct is_arithmetic<float>        : public true_type {};
template<>           struct is_arithmetic<double>       : public true_type {};
template<>           struct is_arithmetic<const float>  : public true_type {};
template<>           struct is_arithmetic<const double> : public true_type {};

template<typename T> struct is_pointer      : public false_type {};
template<typename T> struct is_pointer<T *> : public true_type  {};

template<typename T> struct is_device_ptr  : public false_type {};

template<typename T> struct is_void             : public false_type {};
template<>           struct is_void<void>       : public true_type {};
template<>           struct is_void<const void> : public true_type {};

template<typename T> struct is_non_bool_integral       : public is_integral<T> {};
template<>           struct is_non_bool_integral<bool> : public false_type {};

template<typename T> struct is_non_bool_arithmetic       : public is_arithmetic<T> {};
template<>           struct is_non_bool_arithmetic<bool> : public false_type {};

template<typename T> struct is_pod
   : public integral_constant<
       bool,
       is_void<T>::value || is_pointer<T>::value || is_arithmetic<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
// use intrinsic type traits
       || __is_pod(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ * 100 + __GNUC_MINOR__ >= 403)
       || __is_pod(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
     >
 {};


#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
struct has_trivial_constructor
  : public integral_constant<bool, is_pod<T>::value || ::cuda::std::is_trivially_constructible<T>::value>
{};

template<typename T>
struct has_trivial_copy_constructor
  : public integral_constant<bool, is_pod<T>::value || ::cuda::std::is_trivially_copyable<T>::value>
{};

template<typename T> struct has_trivial_destructor : public is_pod<T> {};
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
template<typename T> struct has_trivial_constructor
  : public integral_constant<
      bool,
      is_pod<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
      || __is_trivially_constructible(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
      || __is_trivially_constructible(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
      >
{};

template <typename T>
struct has_trivial_copy_constructor : public integral_constant<bool,
                                                               is_pod<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
                                                                   || __is_trivially_copyable(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
                                                                    || __is_trivially_copyable(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
                                                               >
{
};

template <typename T>
struct has_trivial_destructor : public is_pod<T>
{
};
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

template<typename T> struct is_const          : public false_type {};
template<typename T> struct is_const<const T> : public true_type {};

template<typename T> struct is_volatile             : public false_type {};
template<typename T> struct is_volatile<volatile T> : public true_type {};

template<typename T>
  struct add_const
{
  using type = T const;
}; // end add_const

template<typename T>
  struct remove_const
{
  using type = T;
}; // end remove_const

template<typename T>
  struct remove_const<const T>
{
  using type = T;
}; // end remove_const

template<typename T>
  struct add_volatile
{
  using type = volatile T;
}; // end add_volatile

template<typename T>
  struct remove_volatile
{
  using type = T;
}; // end remove_volatile

template<typename T>
  struct remove_volatile<volatile T>
{
  using type = T;
}; // end remove_volatile

template<typename T>
  struct add_cv
{
  using type = const volatile T;
}; // end add_cv

template<typename T>
  struct remove_cv
{
  using type = typename remove_const<typename remove_volatile<T>::type>::type;
}; // end remove_cv


template<typename T> struct is_reference     : public false_type {};
template<typename T> struct is_reference<T&> : public true_type {};

template<typename T> struct is_proxy_reference  : public false_type {};

template<typename T> struct is_device_reference                                : public false_type {};
template<typename T> struct is_device_reference< thrust::device_reference<T> > : public true_type {};


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

template<typename _Tp>
  struct add_reference
    : public __add_reference_helper<_Tp>{};

template<typename T>
  struct remove_reference
{
  using type = T;
}; // end remove_reference

template<typename T>
  struct remove_reference<T&>
{
  using type = T;
}; // end remove_reference

template<typename T1, typename T2>
  struct is_same
    : public false_type
{
}; // end is_same

template<typename T>
  struct is_same<T,T>
    : public true_type
{
}; // end is_same

template<typename T1, typename T2>
  struct lazy_is_same
    : is_same<typename T1::type, typename T2::type>
{
}; // end lazy_is_same

template<typename T1, typename T2>
  struct is_different
    : public true_type
{
}; // end is_different

template<typename T>
  struct is_different<T,T>
    : public false_type
{
}; // end is_different

template<typename T1, typename T2>
  struct lazy_is_different
    : is_different<typename T1::type, typename T2::type>
{
}; // end lazy_is_different


#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template<class From, class To>
using is_convertible = ::cuda::std::is_convertible<From, To>;
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
using std::is_convertible;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA



template<typename T1, typename T2>
  struct is_one_convertible_to_the_other
    : public integral_constant<
        bool,
        is_convertible<T1,T2>::value || is_convertible<T2,T1>::value
      >
{};


// mpl stuff
template<typename... Conditions>
  struct or_;

template <>
  struct or_<>
    : public integral_constant<
        bool,
        false_type::value  // identity for or_
      >
{
}; // end or_

template <typename Condition, typename... Conditions>
  struct or_<Condition, Conditions...>
    : public integral_constant<
        bool,
        Condition::value || or_<Conditions...>::value
      >
{
}; // end or_

template <typename... Conditions>
  struct and_;

template<>
  struct and_<>
    : public integral_constant<
        bool,
        true_type::value // identity for and_
      >
{
}; // end and_

template <typename Condition, typename... Conditions>
  struct and_<Condition, Conditions...>
    : public integral_constant<
        bool,
        Condition::value && and_<Conditions...>::value>
{
}; // end and_

template <typename Boolean>
  struct not_
    : public integral_constant<bool, !Boolean::value>
{
}; // end not_

template<bool B, class T, class F>
struct conditional { using type = T; };

template<class T, class F>
struct conditional<false, T, F> { using type = F; };

template <bool, typename Then, typename Else>
  struct eval_if
{
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<true, Then, Else>
{
  using type = typename Then::type;
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<false, Then, Else>
{
  using type = typename Else::type;
}; // end eval_if

template<typename T>
//  struct identity
//  XXX WAR nvcc's confusion with thrust::identity
  struct identity_
{
  using type = T;
}; // end identity

template<bool, typename T = void> struct enable_if {};
template<typename T>              struct enable_if<true, T> {using type = T;};

template<bool, typename T> struct lazy_enable_if {};
template<typename T>       struct lazy_enable_if<true, T> {using type = typename T::type;};

template<bool condition, typename T = void> struct disable_if : enable_if<!condition, T> {};
template<bool condition, typename T>        struct lazy_disable_if : lazy_enable_if<!condition, T> {};


template<typename T1, typename T2, typename T = void>
  struct enable_if_convertible
    : enable_if< is_convertible<T1,T2>::value, T >
{};

template<typename T1, typename T2, typename T = void>
using enable_if_convertible_t = typename enable_if_convertible<T1, T2, T>::type;

template<typename T1, typename T2, typename T = void>
  struct disable_if_convertible
    : disable_if< is_convertible<T1,T2>::value, T >
{};


template<typename T1, typename T2, typename Result = void>
  struct enable_if_different
    : enable_if<is_different<T1,T2>::value, Result>
{};

template<typename T>
  struct is_numeric
    : and_<
        is_convertible<int,T>,
        is_convertible<T,int>
      >
{
}; // end is_numeric


template<typename> struct is_reference_to_const             : false_type {};
template<typename T> struct is_reference_to_const<const T&> : true_type {};


// make_unsigned follows

namespace tt_detail
{

template<typename T> struct make_unsigned_simple;

template<> struct make_unsigned_simple<char>                   { using type = unsigned char;          };
template<> struct make_unsigned_simple<signed char>            { using type = unsigned char;          };
template<> struct make_unsigned_simple<unsigned char>          { using type = unsigned char;          };
template<> struct make_unsigned_simple<short>                  { using type = unsigned short;         };
template<> struct make_unsigned_simple<unsigned short>         { using type = unsigned short;         };
template<> struct make_unsigned_simple<int>                    { using type = unsigned int;           };
template<> struct make_unsigned_simple<unsigned int>           { using type = unsigned int;           };
template<> struct make_unsigned_simple<long int>               { using type = unsigned long int;      };
template<> struct make_unsigned_simple<unsigned long int>      { using type = unsigned long int;      };
template<> struct make_unsigned_simple<long long int>          { using type = unsigned long long int; };
template<> struct make_unsigned_simple<unsigned long long int> { using type = unsigned long long int; };

template<typename T>
  struct make_unsigned_base
{
  // remove cv
  using remove_cv_t = typename remove_cv<T>::type;

  // get the simple unsigned type
  using unsigned_remove_cv_t = typename make_unsigned_simple<remove_cv_t>::type;

  // add back const, volatile, both, or neither to the simple result
  using type = typename eval_if<
    is_const<T>::value && is_volatile<T>::value,
    // add cv back
    add_cv<unsigned_remove_cv_t>,
    // check const & volatile individually
    eval_if<
      is_const<T>::value,
      // add c back
      add_const<unsigned_remove_cv_t>,
      eval_if<
        is_volatile<T>::value,
        // add v back
        add_volatile<unsigned_remove_cv_t>,
        // original type was neither cv, return the simple unsigned result
        identity_<unsigned_remove_cv_t>
      >
    >
  >::type;
};

} // end tt_detail

template<typename T>
  struct make_unsigned
    : tt_detail::make_unsigned_base<T>
{};

struct largest_available_float
{
  using type = double;
};

// T1 wins if they are both the same size
template<typename T1, typename T2>
  struct larger_type
    : thrust::detail::eval_if<
        (sizeof(T2) > sizeof(T1)),
        thrust::detail::identity_<T2>,
        thrust::detail::identity_<T1>
      >
{};


#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template<class Base, class Derived>
using is_base_of = ::cuda::std::is_base_of<Base, Derived>;
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
using std::is_base_of;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA



template<typename Base, typename Derived, typename Result = void>
  struct enable_if_base_of
    : enable_if<
        is_base_of<Base,Derived>::value,
        Result
      >
{};


namespace is_assignable_ns
{

template<typename T1, typename T2>
  class is_assignable
{
  using yes_type = char;
  using no_type  = struct
  {
    char array[2];
  };

  template<typename T> static typename add_reference<T>::type declval();

  template<size_t> struct helper { using type = void *; };

  template<typename U1, typename U2> static yes_type test(typename helper<sizeof(declval<U1>() = declval<U2>())>::type);

  template<typename,typename> static no_type test(...);

  public:
    static const bool value = sizeof(test<T1,T2>(0)) == 1;
}; // end is_assignable

} // end is_assignable_ns


template<typename T1, typename T2>
  struct is_assignable
    : integral_constant<
        bool,
        is_assignable_ns::is_assignable<T1,T2>::value
      >
{};


template<typename T>
  struct is_copy_assignable
    : is_assignable<
        typename add_reference<T>::type,
        typename add_reference<typename add_const<T>::type>::type
      >
{};


template<typename T1, typename T2, typename Enable = void> struct promoted_numerical_type;

template<typename T1, typename T2>
  struct promoted_numerical_type<T1,T2,typename enable_if<and_
  <typename is_floating_point<T1>::type, typename is_floating_point<T2>::type>
  ::value>::type>
  {
  using type = typename larger_type<T1,T2>::type;
  };

template<typename T1, typename T2>
  struct promoted_numerical_type<T1,T2,typename enable_if<and_
  <typename is_integral<T1>::type, typename is_floating_point<T2>::type>
  ::value>::type>
  {
  using type = T2;
  };

template<typename T1, typename T2>
  struct promoted_numerical_type<T1,T2,typename enable_if<and_
  <typename is_floating_point<T1>::type, typename is_integral<T2>::type>
  ::value>::type>
  {
  using type = T1;
  };

template<typename T>
  struct is_empty_helper : public T
  {
  };

struct is_empty_helper_base
{
};

template<typename T>
  struct is_empty : integral_constant<bool,
    sizeof(is_empty_helper_base) == sizeof(is_empty_helper<T>)
  >
  {
  };

template <typename Invokable, typename... Args>
using invoke_result_t =
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  typename ::rocprim::invoke_result<Invokable, Args...>::type;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#if THRUST_CPP_DIALECT < 2017
  typename ::cuda::std::result_of<Invokable(Args...)>::type;
#else // 2017+
  ::cuda::std::invoke_result_t<Invokable, Args...>;
#endif
#else
#if THRUST_CPP_DIALECT < 2017
  std::result_of_t<Invokable(Args...)>;
#else // 2017+
  std::invoke_result_t<Invokable, Args...>;
#endif
#endif

template <class F, class... Us>
struct invoke_result
{
  using type = invoke_result_t<F, Us...>;
};

} // end detail

using detail::integral_constant;
using detail::true_type;
using detail::false_type;

THRUST_NAMESPACE_END

#include <thrust/detail/type_traits/has_trivial_assign.h>
