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
#include <thrust/detail/functional/address_stability.h>
#include <thrust/detail/type_traits.h>

#include _THRUST_STD_INCLUDE(functional)

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(functional)
#else
#  include <utility>
#endif

/*! \cond
 */
namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD
using _THRUST_LIBCXX::maximum;
using _THRUST_LIBCXX::minimum;
using identity = _THRUST_STD::__identity;
#else
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
#endif

} // namespace internal
/*! \endcond
 */

THRUST_NAMESPACE_BEGIN

/*! \addtogroup function_objects Function Objects
 */

//! deprecated [Since 2.6]
template <typename Operation>
struct THRUST_DEPRECATED unary_traits;

//! deprecated [Since 2.6]
template <typename Operation>
struct THRUST_DEPRECATED binary_traits;

/*! \addtogroup function_object_adaptors Function Object Adaptors
 *  \ingroup function_objects
 *  \{
 */

/*! \p unary_function is an empty base class: it contains no member functions
 *  or member variables, but only type information. The only reason it exists
 *  is to make it more convenient to define types that are models of the
 *  concept Adaptable Unary Function. Specifically, any model of Adaptable
 *  Unary Function must define nested aliases. Those are
 *  provided by the base class \p unary_function.
 *
 *  deprecated [Since 2.6]
 *
 *  The following code snippet demonstrates how to construct an
 *  Adaptable Unary Function using \p unary_function.
 *
 *  \code
 *  struct sine : public thrust::unary_function<float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x) { return sinf(x); }
 *  };
 *  \endcode
 *
 *  \note Because C++11 language support makes the functionality of
 *        \c unary_function obsolete, its use is optional if C++11 language
 *        features are enabled.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/unary_function
 *  \see binary_function
 */
template <typename Argument, typename Result>
struct THRUST_DEPRECATED unary_function
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  using argument_type = Argument;

  /*! \typedef result_type;
   *  \brief The type of the function object's result.
   */
  using result_type = Result;
}; // end unary_function

/*! \p binary_function is an empty base class: it contains no member functions
 *  or member variables, but only type information. The only reason it exists
 *  is to make it more convenient to define types that are models of the
 *  concept Adaptable Binary Function. Specifically, any model of Adaptable
 *  Binary Function must define nested aliases. Those are
 *  provided by the base class \p binary_function.
 *
 *  deprecated [Since 2.6]
 *
 *  The following code snippet demonstrates how to construct an
 *  Adaptable Binary Function using \p binary_function.
 *
 *  \code
 *  struct exponentiate : public thrust::binary_function<float,float,float>
 *  {
 *    __host__ __device__
 *    float operator()(float x, float y) { return powf(x,y); }
 *  };
 *  \endcode
 *
 *  \note Because C++11 language support makes the functionality of
 *        \c binary_function obsolete, its use is optional if C++11 language
 *        features are enabled.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/binary_function
 *  \see unary_function
 */
template <typename Argument1, typename Argument2, typename Result>
struct THRUST_DEPRECATED binary_function
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  using first_argument_type = Argument1;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  using second_argument_type = Argument2;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type = Result;
}; // end binary_function

/*! \}
 */

/*! \addtogroup predefined_function_objects Predefined Function Objects
 *  \ingroup function_objects
 */

/*! \addtogroup arithmetic_operations Arithmetic Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

#define THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(func, impl)                                                       \
  template <>                                                                                                       \
  struct func<void>                                                                                                 \
  {                                                                                                                 \
    /*! This functor is transparent. */                                                                             \
    using is_transparent = void;                                                                                    \
    /*! Function call operator. */                                                                                  \
    THRUST_EXEC_CHECK_DISABLE                                                                                       \
    template <typename T1, typename T2>                                                                             \
    THRUST_HOST_DEVICE constexpr auto operator()(T1&& t1, T2&& t2) const noexcept(noexcept(impl)) -> decltype(impl) \
    {                                                                                                               \
      return impl;                                                                                                  \
    }                                                                                                               \
  }

/*! \p plus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>plus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x+y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x+y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>plus</tt> to sum two
 *  device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::plus<float>());
 *  // V3 is now {76, 77, 78, ..., 1075}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/plus
 *  \see binary_function
 */
template <typename T = void>
struct plus : public _THRUST_STD::plus<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end plus

/*! \p minus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>minus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x-y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x-y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>minus</tt> to subtract
 *  a device_vector of \c floats from another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::minus<float>());
 *  // V3 is now {-74, -73, -72, ..., 925}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/minus
 *  \see binary_function
 */
template <typename T = void>
struct minus : public _THRUST_STD::minus<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end minus

/*! \p multiplies is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>multiplies<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x*y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x*y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>multiplies</tt> to multiply
 *  two device_vectors of \c floats.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::multiplies<float>());
 *  // V3 is now {75, 150, 225, ..., 75000}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/multiplies
 *  \see binary_function
 */
template <typename T = void>
struct multiplies : public _THRUST_STD::multiplies<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end multiplies

/*! \p divides is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>divides<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x/y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x/y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>divides</tt> to divide
 *  one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::divides<float>());
 *  // V3 is now {1/75, 2/75, 3/75, ..., 1000/75}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/divides
 *  \see binary_function
 */
template <typename T = void>
struct divides : public _THRUST_STD::divides<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end divides

/*! \p modulus is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>modulus<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x \% y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x \% y</tt> must be defined and must have a return
 * type that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>modulus</tt> to take
 *  the modulus of one device_vectors of \c floats by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<float> V1(N);
 *  thrust::device_vector<float> V2(N);
 *  thrust::device_vector<float> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 75);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::modulus<int>());
 *  // V3 is now {1%75, 2%75, 3%75, ..., 1000%75}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/modulus
 *  \see binary_function
 */
template <typename T = void>
struct modulus : public _THRUST_STD::modulus<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end modulus

/*! \p negate is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f is an object of class <tt>negate<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>-x</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>-x</tt> must be defined and must have a return type that is
 * convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>negate</tt> to negate
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
 *                    thrust::negate<float>());
 *  // V2 is now {-1, -2, -3, ..., -1000}
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/negate
 *  \see unary_function
 */
template <typename T = void>
struct negate : _THRUST_STD::negate<T>
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  using argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end negate

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
 *
 *  \see unary_function
 */
template <typename T = void>
struct square
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  using argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! Function call operator. The return value is <tt>x*x</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE constexpr T operator()(const T& x) const
  {
    return x * x;
  }
}; // end square

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

/*! \p equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x == y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/equal_to
 *  \see binary_function
 */
template <typename T = void>
struct equal_to : public _THRUST_STD::equal_to<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end equal_to

/*! \p not_equal_to is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>not_equal_to<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x != y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/not_equal_to
 *  \see binary_function
 */
template <typename T = void>
struct not_equal_to : public _THRUST_STD::not_equal_to<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end not_equal_to

/*! \p greater is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x > y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/greater
 *  \see binary_function
 */
template <typename T = void>
struct greater : public _THRUST_STD::greater<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end greater

/*! \p less is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x < y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/less
 *  \see binary_function
 */
template <typename T = void>
struct less : public _THRUST_STD::less<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end less

/*! \p greater_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>greater_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x >= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/greater_equal
 *  \see binary_function
 */
template <typename T = void>
struct greater_equal : public _THRUST_STD::greater_equal<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end greater_equal

/*! \p less_equal is a function object. Specifically, it is an Adaptable Binary
 *  Predicate, which means it is a function object that tests the truth or falsehood
 *  of some condition. If \c f is an object of class <tt>less_equal<T></tt> and \c x
 *  and \c y are objects of class \c T, then <tt>f(x,y)</tt> returns \c true if
 *  <tt>x <= y</tt> and \c false otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/less_equal
 *  \see binary_function
 */
template <typename T = void>
struct less_equal : public _THRUST_STD::less_equal<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end less_equal

/*! \}
 */

/*! \addtogroup logical_operations Logical Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p logical_and is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_and<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if both \c x and \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_and
 *  \see binary_function
 */
template <typename T = void>
struct logical_and : public _THRUST_STD::logical_and<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end logical_and

/*! \p logical_or is a function object. Specifically, it is an Adaptable Binary Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_or<T></tt> and \c x and \c y are objects of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x,y)</tt> returns \c true
 *  if and only if either \c x or \c y are \c true.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_or
 *  \see binary_function
 */
template <typename T = void>
struct logical_or : public _THRUST_STD::logical_or<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end logical_or

/*! \p logical_not is a function object. Specifically, it is an Adaptable Predicate,
 *  which means it is a function object that tests the truth or falsehood of some condition.
 *  If \c f is an object of class <tt>logical_not<T></tt> and \c x is an object of
 *  class \c T (where \c T is convertible to \c bool) then <tt>f(x)</tt> returns \c true
 *  if and only if \c x is \c false.
 *
 *  \tparam T must be convertible to \c bool.
 *
 *  The following code snippet demonstrates how to use \p logical_not to transform
 *  a device_vector of \c bools into its logical complement.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  ...
 *  thrust::device_vector<bool> V;
 *  ...
 *  thrust::transform(V.begin(), V.end(), V.begin(), thrust::logical_not<bool>());
 *  // The elements of V are now the logical complement of what they were prior
 *  \endcode
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/logical_not
 *  \see unary_function
 */
template <typename T = void>
struct logical_not : public _THRUST_STD::logical_not<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end logical_not

/*! \}
 */

/*! \addtogroup bitwise_operations Bitwise Operations
 *  \ingroup predefined_function_objects
 *  \{
 */

/*! \p bit_and is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x&y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x&y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_and</tt> to take
 *  the bitwise AND of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_and<int>());
 *  // V3 is now {1&13, 2&13, 3&13, ..., 1000%13}
 *  \endcode
 *
 *  \see binary_function
 */
template <typename T = void>
struct bit_and : public _THRUST_STD::bit_and<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end bit_and

/*! \p bit_or is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x|y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x|y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_or</tt> to take
 *  the bitwise OR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_or<int>());
 *  // V3 is now {1|13, 2|13, 3|13, ..., 1000|13}
 *  \endcode
 *
 *  \see binary_function
 */
template <typename T = void>
struct bit_or : public _THRUST_STD::bit_or<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end bit_or

/*! \p bit_xor is a function object. Specifically, it is an Adaptable Binary Function.
 *  If \c f is an object of class <tt>bit_and<T></tt>, and \c x and \c y are objects
 *  of class \c T, then <tt>f(x,y)</tt> returns <tt>x^y</tt>.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>,
 *          and if \c x and \c y are objects of type \p T, then <tt>x^y</tt> must be defined and must have a return type
 * that is convertible to \c T.
 *
 *  The following code snippet demonstrates how to use <tt>bit_xor</tt> to take
 *  the bitwise XOR of one device_vector of \c ints by another.
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/sequence.h>
 *  #include <thrust/fill.h>
 *  #include <thrust/transform.h>
 *  ...
 *  const int N = 1000;
 *  thrust::device_vector<int> V1(N);
 *  thrust::device_vector<int> V2(N);
 *  thrust::device_vector<int> V3(N);
 *
 *  thrust::sequence(V1.begin(), V1.end(), 1);
 *  thrust::fill(V2.begin(), V2.end(), 13);
 *
 *  thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),
 *                    thrust::bit_xor<int>());
 *  // V3 is now {1^13, 2^13, 3^13, ..., 1000^13}
 *  \endcode
 *
 *  \see binary_function
 */
template <typename T = void>
struct bit_xor : public _THRUST_STD::bit_xor<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end bit_xor

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
 *  \see unary_function
 */
// TODO(bgruber): this version can also act as a functor casting to T making it not equivalent to _THRUST_STD::identity
template <typename T = void>
struct THRUST_DEPRECATED_BECAUSE("use ::internal::identity instead") identity
{
  /*! \typedef argument_type
   *  \brief The type of the function object's first argument.
   */
  using argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

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
struct THRUST_DEPRECATED_BECAUSE("use ::internal::identity instead") identity<void> : ::internal::identity
{};
THRUST_SUPPRESS_DEPRECATED_POP

/*! \p maximum is a function object that takes two arguments and returns the greater
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>maximum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x > y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  The following code snippet demonstrates that \p maximum returns its
 *  greater argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::maximum<int> mx;
 *  assert(x == mx(x,y));
 *  \endcode
 *
 *  \see minimum
 *  \see min
 *  \see binary_function
 */
template <typename T = void>
struct maximum : ::internal::maximum<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end maximum

/*! \p minimum is a function object that takes two arguments and returns the lesser
 *  of the two. Specifically, it is an Adaptable Binary Function. If \c f is an
 *  object of class <tt>minimum<T></tt> and \c x and \c y are objects of class \c T
 *  <tt>f(x,y)</tt> returns \c x if <tt>x < y</tt> and \c y, otherwise.
 *
 *  \tparam T is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>.
 *
 *  The following code snippet demonstrates that \p minimum returns its
 *  lesser argument.
 *
 *  \code
 *  #include <thrust/functional.h>
 *  #include <assert.h>
 *  ...
 *  int x =  137;
 *  int y = -137;
 *  thrust::minimum<int> mn;
 *  assert(y == mn(x,y));
 *  \endcode
 *
 *  \see maximum
 *  \see max
 *  \see binary_function
 */
template <typename T = void>
struct minimum : ::internal::minimum<T>
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T;
}; // end minimum

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
 *  \see binary_function
 */
template <typename T1 = void, typename T2 = void>
struct project1st
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T1;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T2;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T1;

  /*! Function call operator. The return value is <tt>lhs</tt>.
   */
  THRUST_HOST_DEVICE constexpr const T1& operator()(const T1& lhs, const T2& /*rhs*/) const
  {
    return lhs;
  }
}; // end project1st

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
 *  \see binary_function
 */
template <typename T1 = void, typename T2 = void>
struct project2nd
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   *  deprecated [Since 2.6]
   */
  using first_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T1;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   *  deprecated [Since 2.6]
   */
  using second_argument_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T2;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   *  deprecated [Since 2.6]
   */
  using result_type THRUST_ALIAS_ATTRIBUTE(THRUST_DEPRECATED) = T2;

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

/*! \p unary_negate is a function object adaptor: it is an Adaptable Predicate
 *  that represents the logical negation of some other Adaptable Predicate.
 *  That is: if \c f is an object of class <tt>unary_negate<AdaptablePredicate></tt>,
 *  then there exists an object \c pred of class \c AdaptablePredicate such
 *  that <tt>f(x)</tt> always returns the same value as <tt>!pred(x)</tt>.
 *  There is rarely any reason to construct a <tt>unary_negate</tt> directly;
 *  it is almost always easier to use the helper function not1.
 *
 *  deprecated [Since 2.6]
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/unary_negate
 *  \see not1
 */
template <typename Predicate>
struct THRUST_DEPRECATED unary_negate
{
  /*! \typedef argument_type
   *  \brief The type of the function object's argument.
   */
  using argument_type = typename Predicate::argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type = bool;

  /*! Constructor takes a \p Predicate object to negate.
   *  \param p The \p Predicate object to negate.
   */
  THRUST_HOST_DEVICE explicit unary_negate(Predicate p)
      : pred(p)
  {}

  /*! Function call operator. The return value is <tt>!pred(x)</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE bool operator()(const typename Predicate::argument_type& x)
  {
    return !pred(x);
  }

  /*! \cond
   */
  Predicate pred;
  /*! \endcond
   */
}; // end unary_negate

THRUST_SUPPRESS_DEPRECATED_PUSH
/*! \p not1 is a helper function to simplify the creation of Adaptable Predicates:
 *  it takes an Adaptable Predicate \p pred as an argument and returns a new Adaptable
 *  Predicate that represents the negation of \p pred. That is: if \c pred is an object
 *  of a type which models Adaptable Predicate, then the the type of the result
 *  \c npred of <tt>not1(pred)</tt> is also a model of Adaptable Predicate and
 *  <tt>npred(x)</tt> always returns the same value as <tt>!pred(x)</tt>.
 *
 *  deprecated [Since 2.6]
 *
 *  \param pred The Adaptable Predicate to negate.
 *  \return A new object, <tt>npred</tt> such that <tt>npred(x)</tt> always returns
 *          the same value as <tt>!pred(x)</tt>.
 *  \tparam Predicate is a model of <a
 * href="https://en.cppreference.com/w/cpp/utility/functional/unary_negate">Adaptable Predicate</a>.
 *  \see unary_negate
 *  \see not2
 */
template <typename Predicate>
THRUST_HOST_DEVICE
  THRUST_DEPRECATED_BECAUSE("Use thrust::not_fn instead") unary_negate<Predicate> not1(const Predicate& pred);
THRUST_SUPPRESS_DEPRECATED_POP

/*! \p binary_negate is a function object adaptor: it is an Adaptable Binary
 *  Predicate that represents the logical negation of some other Adaptable
 *  Binary Predicate. That is: if \c f is an object of class <tt>binary_negate<AdaptablePredicate></tt>,
 *  then there exists an object \c pred of class \c AdaptableBinaryPredicate
 *  such that <tt>f(x,y)</tt> always returns the same value as <tt>!pred(x,y)</tt>.
 *  There is rarely any reason to construct a <tt>binary_negate</tt> directly;
 *  it is almost always easier to use the helper function not2.
 *
 *  deprecated [Since 2.6]
 *
 *  \see https://en.cppreference.com/w/cpp/utility/functional/binary_negate
 */
template <typename Predicate>
struct THRUST_DEPRECATED binary_negate
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  using first_argument_type = typename Predicate::first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  using second_argument_type = typename Predicate::second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  using result_type = bool;

  /*! Constructor takes a \p Predicate object to negate.
   *  \param p The \p Predicate object to negate.
   */
  THRUST_HOST_DEVICE explicit binary_negate(Predicate p)
      : pred(p)
  {}

  /*! Function call operator. The return value is <tt>!pred(x,y)</tt>.
   */
  THRUST_EXEC_CHECK_DISABLE
  THRUST_HOST_DEVICE bool operator()(const first_argument_type& x, const second_argument_type& y)
  {
    return !pred(x, y);
  }

  /*! \cond
   */
  Predicate pred;
  /*! \endcond
   */
}; // end binary_negate

THRUST_SUPPRESS_DEPRECATED_PUSH
/*! \p not2 is a helper function to simplify the creation of Adaptable Binary Predicates:
 *  it takes an Adaptable Binary Predicate \p pred as an argument and returns a new Adaptable
 *  Binary Predicate that represents the negation of \p pred. That is: if \c pred is an object
 *  of a type which models Adaptable Binary Predicate, then the the type of the result
 *  \c npred of <tt>not2(pred)</tt> is also a model of Adaptable Binary Predicate and
 *  <tt>npred(x,y)</tt> always returns the same value as <tt>!pred(x,y)</tt>.
 *
 *  deprecated [Since 2.6]
 *
 *  \param pred The Adaptable Binary Predicate to negate.
 *  \return A new object, <tt>npred</tt> such that <tt>npred(x,y)</tt> always returns
 *          the same value as <tt>!pred(x,y)</tt>.
 *  \tparam Binary Predicate is a model of an Adaptable Binary Predicate.
 *  \see binary_negate
 *  \see not1
 */
template <typename BinaryPredicate>
THRUST_HOST_DEVICE THRUST_DEPRECATED_BECAUSE("Use thrust::not_fn instead")
  binary_negate<BinaryPredicate> not2(const BinaryPredicate& pred);
THRUST_SUPPRESS_DEPRECATED_POP

namespace detail
{
template <typename F>
struct not_fun_t
{
  F f;

  template <typename... Ts>
  THRUST_HOST_DEVICE auto operator()(Ts&&... args) noexcept(noexcept(!f(std::forward<Ts>(args)...)))
    -> decltype(!f(std::forward<Ts>(args)...))
  {
    return !f(std::forward<Ts>(args)...);
  }

  template <typename... Ts>
  THRUST_HOST_DEVICE auto operator()(Ts&&... args) const noexcept(noexcept(!f(std::forward<Ts>(args)...)))
    -> decltype(!f(std::forward<Ts>(args)...))
  {
    return !f(std::forward<Ts>(args)...);
  }
};
} // namespace detail

//! Takes a predicate (a callable returning bool) and returns a new predicate that returns the negated result.
//! \see https://en.cppreference.com/w/cpp/utility/functional/not_fn
// TODO(bgruber): alias to _THRUST_STD::not_fn in C++17
template <class F>
THRUST_HOST_DEVICE auto not_fn(F&& f) -> detail::not_fun_t<::internal::decay_t<F>>
{
  return detail::not_fun_t<::internal::decay_t<F>>{std::forward<F>(f)};
}

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

#undef THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION

THRUST_NAMESPACE_END

#ifndef THRUST_DOXYGEN_INVOKED // Do not document
#  if _THRUST_HAS_DEVICE_SYSTEM_STD
_LIBCUDACXX_BEGIN_NAMESPACE_CUDA
#  else
THRUST_NAMESPACE_BEGIN
namespace detail
{
#  endif
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::plus);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::minus);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::multiplies);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::divides);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::modulus);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::negate);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::bit_and);
// THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::bit_not); // does not exist?
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::bit_or);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::bit_xor);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::equal_to);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::not_equal_to);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::less);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::less_equal);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::greater_equal);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::greater);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::logical_and);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::logical_not);
THRUST_MARK_CAN_COPY_ARGUMENTS(THRUST_NS_QUALIFIER::logical_or);
#  if _THRUST_HAS_DEVICE_SYSTEM_STD
_LIBCUDACXX_END_NAMESPACE_CUDA
#  else
}
THRUST_NAMESPACE_END
#  endif
#endif // THRUST_DOXYGEN_INVOKED

#include <thrust/detail/functional.inl>
#include <thrust/detail/functional/operators.h>
#include <thrust/detail/type_traits/is_commutative.h>
