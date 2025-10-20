/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file functional.h
 *  \brief Function objects and tools for manipulating them
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

#include <thrust/detail/functional/actor.h>

#include _THRUST_STD_INCLUDE(functional)

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(functional)
#else
#  include <thrust/detail/type_traits.h>

#  include <any>
#  include <tuple>
#  include <type_traits>
#  include <utility>
#endif

// TODO(libhipcxx): remove this namespace once libhipcxx gets ready
#ifndef THRUST_DOXYGEN_INVOKED
namespace internal
{

#  if _THRUST_HAS_DEVICE_SYSTEM_STD
using _THRUST_LIBCXX::maximum;
using _THRUST_LIBCXX::minimum;
using _THRUST_STD::not_fn;
using identity = _THRUST_STD::__identity;
#  else
// cuda::maximum or hip::maximum
template <typename T = void>
struct maximum
{
  THRUST_EXEC_CHECK_DISABLE
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr T operator()(const T& lhs, const T& rhs) const
    noexcept(noexcept((lhs < rhs) ? rhs : lhs))
  {
    return (lhs < rhs) ? rhs : lhs;
  }
};

template <>
struct maximum<void>
{
  THRUST_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr ::std::common_type_t<T1, T2>
  operator()(const T1& lhs, const T2& rhs) const noexcept(noexcept((lhs < rhs) ? rhs : lhs))
  {
    return (lhs < rhs) ? rhs : lhs;
  }
};

// cuda::minimum or hip::minimum
template <typename T = void>
struct minimum
{
  THRUST_EXEC_CHECK_DISABLE
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr T operator()(const T& lhs, const T& rhs) const
    noexcept(noexcept((lhs < rhs) ? lhs : rhs))
  {
    return (lhs < rhs) ? lhs : rhs;
  }
};

template <>
struct minimum<void>
{
  THRUST_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr ::std::common_type_t<T1, T2>
  operator()(const T1& lhs, const T2& rhs) const noexcept(noexcept((lhs < rhs) ? lhs : rhs))
  {
    return (lhs < rhs) ? lhs : rhs;
  }
};

// _THRUST_STD::not_fn
namespace detail
{

struct nat
{
  nat()                      = delete;
  nat(const nat&)            = delete;
  nat& operator=(const nat&) = delete;
  ~nat()                     = delete;
};

template <typename DecayedFp>
struct member_pointer_class_type
{};

template <typename Ret, typename ClassType>
struct member_pointer_class_type<Ret ClassType::*>
{
  typedef ClassType type;
};

template <typename Tp>
struct is_reference_wrapper_impl : public ::std::false_type
{};
template <typename Tp>
struct is_reference_wrapper_impl<::std::reference_wrapper<Tp>> : public ::std::true_type
{};
template <typename Tp>
struct is_reference_wrapper : public is_reference_wrapper_impl<::std::remove_cv_t<Tp>>
{};

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet1 =
  ::std::enable_if_t<::std::is_member_function_pointer<DecayFp>::value&& ::std::is_base_of<ClassT, DecayA0>::value>;

template <typename Fp, typename A0, typename DecayFp = ::internal::decay_t<Fp>, typename DecayA0 = ::internal::decay_t<A0>>
using enable_if_bullet2 =
  ::std::enable_if_t<::std::is_member_function_pointer<DecayFp>::value && is_reference_wrapper<DecayA0>::value>;

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet3 =
  ::std::enable_if_t<::std::is_member_function_pointer<DecayFp>::value && !::std::is_base_of<ClassT, DecayA0>::value
                     && !is_reference_wrapper<DecayA0>::value>;

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet4 =
  ::std::enable_if_t<::std::is_member_object_pointer<DecayFp>::value&& ::std::is_base_of<ClassT, DecayA0>::value>;

template <typename Fp, typename A0, typename DecayFp = ::internal::decay_t<Fp>, typename DecayA0 = ::internal::decay_t<A0>>
using enable_if_bullet5 =
  ::std::enable_if_t<::std::is_member_object_pointer<DecayFp>::value && is_reference_wrapper<DecayA0>::value>;

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet6 =
  ::std::enable_if_t<::std::is_member_object_pointer<DecayFp>::value && !::std::is_base_of<ClassT, DecayA0>::value
                     && !is_reference_wrapper<DecayA0>::value>;

template <typename... Args>
inline THRUST_HOST_DEVICE nat __invoke(::std::any, Args&&... args);

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename... Args, typename = enable_if_bullet1<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype((::std::declval<A0>().*::std::declval<Fp>())(::std::declval<Args>()...))
__invoke(Fp&& f, A0&& a0, Args&&... args) noexcept(noexcept((static_cast<A0&&>(a0).*f)(static_cast<Args&&>(args)...)))
{
  return (static_cast<A0&&>(a0).*f)(static_cast<Args&&>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename... Args, typename = enable_if_bullet2<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype((::std::declval<A0>().get().*::std::declval<Fp>())(
  ::std::declval<Args>()...))
__invoke(Fp&& f, A0&& a0, Args&&... args) noexcept(noexcept((a0.get().*f)(static_cast<Args&&>(args)...)))
{
  return (a0.get().*f)(static_cast<Args&&>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename... Args, typename = enable_if_bullet3<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype(((*::std::declval<A0>()).*::std::declval<Fp>())(::std::declval<Args>()...))
__invoke(Fp&& f, A0&& a0, Args&&... args) noexcept(noexcept(((*static_cast<A0&&>(a0)).*f)(static_cast<Args&&>(args)...)))
{
  return ((*static_cast<A0&&>(a0)).*f)(static_cast<Args&&>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename = enable_if_bullet4<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype(::std::declval<A0>().*::std::declval<Fp>())
__invoke(Fp&& f, A0&& a0) noexcept(noexcept(static_cast<A0&&>(a0).*f))
{
  return static_cast<A0&&>(a0).*f;
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename = enable_if_bullet5<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype(::std::declval<A0>().get().*::std::declval<Fp>())
__invoke(Fp&& f, A0&& a0) noexcept(noexcept(a0.get().*f))
{
  return a0.get().*f;
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename = enable_if_bullet6<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype((*::std::declval<A0>()).*::std::declval<Fp>())
__invoke(Fp&& f, A0&& a0) noexcept(noexcept((*static_cast<A0&&>(a0)).*f))
{
  return (*static_cast<A0&&>(a0)).*f;
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename... Args>
inline THRUST_HOST_DEVICE constexpr decltype(::std::declval<Fp>()(::std::declval<Args>()...))
__invoke(Fp&& f, Args&&... args) noexcept(noexcept(static_cast<Fp&&>(f)(static_cast<Args&&>(args)...)))
{
  return static_cast<Fp&&>(f)(static_cast<Args&&>(args)...);
}

template <typename Fn, typename... Args>
inline THRUST_HOST_DEVICE constexpr ::thrust::detail::invoke_result_t<Fn, Args...>
invoke(Fn&& f, Args&&... args) noexcept(::std::is_nothrow_invocable_v<Fn, Args...>)
{
  return ::internal::detail::__invoke(::std::forward<Fn>(f), ::std::forward<Args>(args)...);
}

template <typename Op, typename Indices, typename... BoundArgs>
struct perfect_forward_impl;

template <typename Op, size_t... Idx, typename... BoundArgs>
struct perfect_forward_impl<Op, ::std::index_sequence<Idx...>, BoundArgs...>
{
private:
  ::std::tuple<BoundArgs...> bound_args;

public:
  template <typename... Args,
            bool cccl_true = true,
            ::std::enable_if_t<::std::is_constructible_v<::std::tuple<BoundArgs...>, Args&&...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE explicit constexpr perfect_forward_impl(Args&&... args) noexcept(
    ::std::is_nothrow_constructible_v<::std::tuple<BoundArgs...>, Args&&...>)
      : bound_args(::std::forward<Args>(args)...)
  {}

  inline perfect_forward_impl(perfect_forward_impl const&) = default;
  inline perfect_forward_impl(perfect_forward_impl&&)      = default;

  inline perfect_forward_impl& operator=(perfect_forward_impl const&) = default;
  inline perfect_forward_impl& operator=(perfect_forward_impl&&)      = default;

  template <typename... Args,
            bool cccl_true                                                                          = true,
            ::std::enable_if_t<::std::is_invocable_v<Op, BoundArgs&..., Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto
  operator()(Args&&... args) & noexcept(noexcept(Op()(::std::get<Idx>(bound_args)..., ::std::forward<Args>(args)...)))
    -> decltype(Op()(::std::get<Idx>(bound_args)..., ::std::forward<Args>(args)...))
  {
    return Op()(::std::get<Idx>(bound_args)..., ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                            = true,
            ::std::enable_if_t<(!::std::is_invocable_v<Op, BoundArgs&..., Args...>) &&cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE auto operator()(Args&&...) & = delete;

  template <typename... Args,
            bool cccl_true                                                                                = true,
            ::std::enable_if_t<::std::is_invocable_v<Op, BoundArgs const&..., Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto operator()(Args&&... args) const& noexcept(
    noexcept(Op()(::std::get<Idx>(bound_args)..., ::std::forward<Args>(args)...)))
    -> decltype(Op()(::std::get<Idx>(bound_args)..., ::std::forward<Args>(args)...))
  {
    return Op()(::std::get<Idx>(bound_args)..., ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                                  = true,
            ::std::enable_if_t<(!::std::is_invocable_v<Op, BoundArgs const&..., Args...>) &&cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE auto operator()(Args&&...) const& = delete;

  template <typename... Args,
            bool cccl_true                                                                         = true,
            ::std::enable_if_t<::std::is_invocable_v<Op, BoundArgs..., Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto operator()(Args&&... args) && noexcept(
    noexcept(Op()(::std::get<Idx>(::std::move(bound_args))..., ::std::forward<Args>(args)...)))
    -> decltype(Op()(::std::get<Idx>(::std::move(bound_args))..., ::std::forward<Args>(args)...))
  {
    return Op()(::std::get<Idx>(::std::move(bound_args))..., ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                           = true,
            ::std::enable_if_t<(!::std::is_invocable_v<Op, BoundArgs..., Args...>) &&cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE auto operator()(Args&&...) && = delete;

  template <typename... Args,
            bool cccl_true                                                                               = true,
            ::std::enable_if_t<::std::is_invocable_v<Op, BoundArgs const..., Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto operator()(Args&&... args) const&& noexcept(
    noexcept(Op()(::std::get<Idx>(::std::move(bound_args))..., ::std::forward<Args>(args)...)))
    -> decltype(Op()(::std::get<Idx>(::std::move(bound_args))..., ::std::forward<Args>(args)...))
  {
    return Op()(::std::get<Idx>(::std::move(bound_args))..., ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                                 = true,
            ::std::enable_if_t<(!::std::is_invocable_v<Op, BoundArgs const..., Args...>) &&cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE auto operator()(Args&&...) const&& = delete;
};

template <typename Op, typename... Args>
using perfect_forward = perfect_forward_impl<Op, ::std::index_sequence_for<Args...>, Args...>;

struct not_fn_op
{
  template <typename... Args>
  inline THRUST_HOST_DEVICE THRUST_CONSTEXPR_CXX20 auto operator()(Args&&... args) const
    noexcept(noexcept(!::internal::detail::invoke(::std::forward<Args>(args)...)))
      -> decltype(!::internal::detail::invoke(::std::forward<Args>(args)...))
  {
    return !::internal::detail::invoke(::std::forward<Args>(args)...);
  }
};

template <typename Fn>
struct not_fn_t : perfect_forward<not_fn_op, Fn>
{
  using base = perfect_forward<not_fn_op, Fn>;
  using base::base;
  inline constexpr not_fn_t() noexcept = default;
};

} // namespace detail

template <typename Fn,
          typename =
            ::std::enable_if_t<::std::is_constructible_v<decay_t<Fn>, Fn>&& ::std::is_move_constructible_v<decay_t<Fn>>>>
inline THRUST_HOST_DEVICE THRUST_CONSTEXPR_CXX20 auto not_fn(Fn&& f)
{
  return detail::not_fn_t<decay_t<Fn>>(::std::forward<Fn>(f));
}

// _THRUST_STD::__identity
struct identity
{
  template <typename T>
  THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr T&& operator()(T&& t) const noexcept
  {
    return ::std::forward<T>(t);
  }

  using is_transparent = void;
};
#  endif

} // namespace internal
#endif

THRUST_NAMESPACE_BEGIN

/*! \addtogroup predefined_function_objects Predefined Function Objects
 *  \ingroup function_objects
 */

/*! \addtogroup arithmetic_operations Arithmetic Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

using _THRUST_STD::divides;
using _THRUST_STD::minus;
using _THRUST_STD::modulus;
using _THRUST_STD::multiplies;
using _THRUST_STD::negate;
using _THRUST_STD::plus;

/*! \p square is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>square<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x*x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x*x</tt> must be defined and must have a return type that is
 * convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>square</tt> to square
 *  the elements of a device_vector of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(),
 *                    thrust::square<float>());
 *  // V2 is now {1, 4, 9, ..., 1000000}
 *  \endcode
 */
template <typename T = void>
struct square
{
  /*! Function call operator. The return value is <tt>x*x</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE constexpr T operator()(const T& x) const
  {
    return x * x;
  }
};

/*! \brief Specialization of \p square for type void.
 */
template <>
struct square<void>
{
  /*! This functor is transparent. */
  using is_transparent = void;

  /*! Function call operator - returns the square of its argument*/
  THRUST_EXEC_CHECK_DISABLE
  template <typename T>
  THRUST_HOST_DEVICE constexpr T operator()(const T& x) const noexcept(noexcept(x * x))
  {
    return x * x;
  }
};

/*! \}
 */

/*! \addtogroup comparison_operations Comparison Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

using _THRUST_STD::equal_to;
using _THRUST_STD::greater;
using _THRUST_STD::greater_equal;
using _THRUST_STD::less;
using _THRUST_STD::less_equal;
using _THRUST_STD::not_equal_to;

/*! \}
 */

/*! \addtogroup logical_operations Logical Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

using _THRUST_STD::logical_and;
using _THRUST_STD::logical_not;
using _THRUST_STD::logical_or;

/*! \}
 */

/*! \addtogroup bitwise_operations Bitwise Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

using _THRUST_STD::bit_and;
using _THRUST_STD::bit_or;
using _THRUST_STD::bit_xor;

/*! \}
 */

/*! \addtogroup generalized_identity_operations Generalized Identity Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p identity is a Unary Function that represents the identity function: it takes
 *  a single argument \c x, and returns \c x.
 *
 *  \tparam T No requirements on \p T.
 *
 *  The following code snippet demonstrates that \p identity returns its
 *  argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x = 137;
 *  thrust::identity<int> id;
 *  assert(x == id(x));
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/identity
 */
// TODO(bgruber): this version can also act as a functor casting to T making it not equivalent to _THRUST_STD::identity
template <typename T = void>
struct THRUST_DEPRECATED_BECAUSE("use internal::identity instead") identity
{
  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type THRUST_DEPRECATED_IN_CXX11 = T;

  /*! Function call operator. The return value is <tt>x</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE constexpr const T& operator()(const T& x) const
  {
    return x;
  }

  /*! Function call operator. The return value is <tt>x</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE constexpr T& operator()(T& x) const
  {
    return x;
  }

  // we cannot add an overload for `const T&&` because then calling e.g. `thrust::identity<int>{}(3.14);` is ambiguous
  // on MSVC

  /*! Function call operator. The return value is <tt>move(x)</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE constexpr T&& operator()(T&& x) const
  {
    return _THRUST_STD::move(x);
  }
};

THRUST_SUPPRESS_DEPRECATED_PUSH
template <>
struct THRUST_DEPRECATED_BECAUSE("use internal::identity instead") identity<void> : ::internal::identity
{};
THRUST_SUPPRESS_DEPRECATED_POP

using ::internal::maximum;
using ::internal::minimum;

/*! \p project1st is a function object that takes two arguments and returns
 *  its first argument; the second argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project1st<int> pj1;
 *  assert(x == pj1(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project2nd
 */
template <typename T1 = void, typename T2 = void>
struct project1st
{
  /*! Function call operator. The return value is <tt>lhs</tt>.
   */
  THRUST_HOST_DEVICE constexpr const T1& operator()(const T1& lhs, const T2& /*rhs*/) const
  {
    return lhs;
  }
};

/*! \brief Specialization of \p project1st for two void arguments.
 */
template <>
struct project1st<void, void>
{
  /// Indicate that this functor is transparent: it accepts any argument that can be
  /// converted to the required type, and uses perfect forwarding.
  using is_transparent = void;

  /// \brief Invocation operator - returns its first argument.
  THRUST_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  THRUST_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&&) const noexcept(noexcept(THRUST_FWD(t1)))
    -> decltype(THRUST_FWD(t1))
  {
    return THRUST_FWD(t1);
  }
};

/*! \p project2nd is a function object that takes two arguments and returns
 *  its second argument; the first argument is unused. It is essentially a
 *  generalization of identity to the case of a Binary Function.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::project2nd<int> pj2;
 *  assert(y == pj2(x,y));
 *  \endcode
 *
 *  \see identity
 *  \see project1st
 */
template <typename T1 = void, typename T2 = void>
struct project2nd
{
  /*! Function call operator. The return value is <tt>rhs</tt>.
   */
  THRUST_HOST_DEVICE constexpr const T2& operator()(const T1& /*lhs*/, const T2& rhs) const
  {
    return rhs;
  }
}; // end project2nd

/*! \brief Specialization of \p project2nd for two void arguments.
 */
template <>
struct project2nd<void, void>
{
  /// Indicate that this functor is transparent: it accepts any argument that can be
  /// converted to the required type, and uses perfect forwarding.
  using is_transparent = void;

  /// \brief Invocation operator - returns its second argument.
  THRUST_EXEC_CHECK_DISABLE
  template <typename T1, typename T2>
  THRUST_HOST_DEVICE constexpr auto operator()(T1&&, T2&& t2) const noexcept(noexcept(THRUST_FWD(t2)))
    -> decltype(THRUST_FWD(t2))
  {
    return THRUST_FWD(t2);
  }
};

/*! \}
 */

// odds and ends

/*! \addtogroup function_object_adaptors
 *  \{
 */

using ::internal::not_fn;

/*! \}
 */

/*! \addtogroup placeholder_objects Placeholder Objects
 *  \ingroup function_objects
 *  \{
 */

/*! \namespace thrust::placeholders
 *  \brief Facilities for constructing simple functions inline.
 *
 *  Objects in the \p thrust::placeholders namespace may be used to create simple arithmetic functions inline
 *  in an algorithm invocation. Combining placeholders such as \p _1 and \p _2 with arithmetic operations such as \c +
 *  creates an unnamed function object which applies the operation to their arguments.
 *
 *  The type of placeholder objects is implementation-defined.
 *
 *  The following code snippet demonstrates how to use the placeholders \p _1 and \p _2 with \p thrust::transform
 *  to implement the SAXPY computation:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<float> x(4), y(4);
 *    x[0] = 1;
 *    x[1] = 2;
 *    x[2] = 3;
 *    x[3] = 4;
 *
 *    y[0] = 1;
 *    y[1] = 1;
 *    y[2] = 1;
 *    y[3] = 1;
 *
 *    float a = 2.0f;
 *
 *    using namespace thrust::placeholders;
 *
 *    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(),
 *      a * _1 + _2
 *    );
 *
 *    // y is now {3, 5, 7, 9}
 *  }
 *  \endcode
 */
namespace placeholders
{

/*! \p thrust::placeholders::_1 is the placeholder for the first function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<0>::type _1;

/*! \p thrust::placeholders::_2 is the placeholder for the second function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<1>::type _2;

/*! \p thrust::placeholders::_3 is the placeholder for the third function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<2>::type _3;

/*! \p thrust::placeholders::_4 is the placeholder for the fourth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<3>::type _4;

/*! \p thrust::placeholders::_5 is the placeholder for the fifth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<4>::type _5;

/*! \p thrust::placeholders::_6 is the placeholder for the sixth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<5>::type _6;

/*! \p thrust::placeholders::_7 is the placeholder for the seventh function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<6>::type _7;

/*! \p thrust::placeholders::_8 is the placeholder for the eighth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<7>::type _8;

/*! \p thrust::placeholders::_9 is the placeholder for the ninth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<8>::type _9;

/*! \p thrust::placeholders::_10 is the placeholder for the tenth function parameter.
 */
THRUST_INLINE_CONSTANT thrust::detail::functional::placeholder<9>::type _10;

} // namespace placeholders

/*! \} // placeholder_objects
 */

THRUST_NAMESPACE_END

#include <thrust/detail/functional/operators.h>
#include <thrust/detail/type_traits/is_commutative.h>
