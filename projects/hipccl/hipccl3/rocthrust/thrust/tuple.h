/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.  All rights reserved.
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

/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements.
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#elif defined(__has_include)
#if __has_include(<cuda/std/tuple>)
#include <cuda/std/tuple>
#endif // __has_include(<cuda/std/tuple>)
#if __has_include(<cuda/std/type_traits>)
#include <cuda/std/type_traits>
#endif // __has_include(<cuda/std/type_traits>)
#if __has_include(<cuda/std/utility>)
#include <cuda/std/utility>
#endif // __has_include(<cuda/std/utility>)
#endif // THRUST_DEVICE_SYSTEM

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

#include <tuple>

THRUST_NAMESPACE_BEGIN

/*! \cond
 */

// define null_type for backwards compatability
struct null_type {};

THRUST_HOST_DEVICE inline
bool operator==(const null_type&, const null_type&) { return true; }

THRUST_HOST_DEVICE inline
bool operator>=(const null_type&, const null_type&) { return true; }

THRUST_HOST_DEVICE inline
bool operator<=(const null_type&, const null_type&) { return true; }

THRUST_HOST_DEVICE inline
bool operator!=(const null_type&, const null_type&) { return false; }

THRUST_HOST_DEVICE inline
bool operator<(const null_type&, const null_type&) { return false; }

THRUST_HOST_DEVICE inline
bool operator>(const null_type&, const null_type&) { return false; }

/*! \endcond
 */

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <size_t N, class T>
using tuple_element = _CUDA_VSTD::tuple_element<N, T>;

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <class T>
using tuple_size = _CUDA_VSTD::tuple_size<T>;

template <class>
struct __is_tuple_of_iterator_references : _CUDA_VSTD::false_type
{};

/*! \brief \p tuple is a class template that can be instantiated with up to ten
 *  arguments. Each template argument specifies the type of element in the \p
 *  tuple. Consequently, tuples are heterogeneous, fixed-size collections of
 *  values. An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class... Ts>
struct tuple : public _CUDA_VSTD::tuple<Ts...>
{
  using super_t = _CUDA_VSTD::tuple<Ts...>;
  using super_t::super_t;

  tuple() = default;

  template <class _TupleOfIteratorReferences,
            _CUDA_VSTD::__enable_if_t<__is_tuple_of_iterator_references<_TupleOfIteratorReferences>::value, int> = 0,
            _CUDA_VSTD::__enable_if_t<(tuple_size<_TupleOfIteratorReferences>::value == sizeof...(Ts)), int>     = 0>
  _CCCL_HOST_DEVICE tuple(_TupleOfIteratorReferences&& tup)
      : tuple(_CUDA_VSTD::forward<_TupleOfIteratorReferences>(tup).template __to_tuple<Ts...>(
        _CUDA_VSTD::__make_tuple_indices_t<sizeof...(Ts)>()))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class TupleLike,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__tuple_assignable<TupleLike, super_t>::value, int> = 0>
  _CCCL_HOST_DEVICE tuple& operator=(TupleLike&& other)
  {
    super_t::operator=(_CUDA_VSTD::forward<TupleLike>(other));
    return *this;
  }

#if defined(_CCCL_COMPILER_MSVC_2017)
  // MSVC2017 needs some help to convert tuples
  template <class... Us,
            _CUDA_VSTD::__enable_if_t<!_CUDA_VSTD::is_same<tuple<Us...>, tuple>::value, int> = 0,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__tuple_convertible<_CUDA_VSTD::tuple<Us...>, super_t>::value, int> = 0>
  _CCCL_HOST_DEVICE constexpr operator tuple<Us...>()
  {
    return __to_tuple<Us...>(typename _CUDA_VSTD::__make_tuple_indices<sizeof...(Ts)>::type{});
  }

  template <class... Us, size_t... Id>
  _CCCL_HOST_DEVICE constexpr tuple<Us...> __to_tuple(_CUDA_VSTD::__tuple_indices<Id...>) const
  {
    return tuple<Us...>{_CUDA_VSTD::get<Id>(*this)...};
  }
#endif // _CCCL_COMPILER_MSVC_2017
};

#if _CCCL_STD_VER >= 2017
template <class... Ts>
_CCCL_HOST_DEVICE tuple(Ts...) -> tuple<Ts...>;

template <class T1, class T2>
struct pair;

template <class T1, class T2>
_CCCL_HOST_DEVICE tuple(pair<T1, T2>) -> tuple<T1, T2>;
#endif // _CCCL_STD_VER >= 2017

template <class... Ts>
inline _CCCL_HOST_DEVICE
  _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__all<_CUDA_VSTD::__is_swappable<Ts>::value...>::value, void>
  swap(tuple<Ts...>& __x,
       tuple<Ts...>& __y) noexcept((_CUDA_VSTD::__all<_CUDA_VSTD::__is_nothrow_swappable<Ts>::value...>::value))
{
  __x.swap(__y);
}

template <class... Ts>
inline _CCCL_HOST_DEVICE tuple<typename _CUDA_VSTD::__unwrap_ref_decay<Ts>::type...> make_tuple(Ts&&... __t)
{
  return tuple<typename _CUDA_VSTD::__unwrap_ref_decay<Ts>::type...>(_CUDA_VSTD::forward<Ts>(__t)...);
}

template <class... Ts>
inline _CCCL_HOST_DEVICE tuple<Ts&...> tie(Ts&... ts) noexcept
{
  return tuple<Ts&...>(ts...);
}

using _CUDA_VSTD::get;

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_size<tuple<Ts...>>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_element<Id, tuple<Ts...>>
{};

template <class... Ts>
struct __tuple_like_ext<THRUST_NS_QUALIFIER::tuple<Ts...>> : true_type
{};

template <>
struct tuple_size<tuple<THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<>>
{};

template <class T0>
struct tuple_size<tuple<T0,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0>>
{};

template <class T0, class T1>
struct tuple_size<tuple<T0,
                        T1,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1>>
{};

template <class T0, class T1, class T2>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2>>
{};

template <class T0, class T1, class T2, class T3>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        T3,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3>>
{};

template <class T0, class T1, class T2, class T3, class T4>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        T3,
                        T4,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5>
struct tuple_size<tuple<T0,
                        T1,
                        T2,
                        T3,
                        T4,
                        T5,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type,
                        THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4, T5>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
struct tuple_size<
  tuple<T0, T1, T2, T3, T4, T5, T6, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>>
    : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>>
    : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7>>
{};

template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, THRUST_NS_QUALIFIER::null_type>>
    : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>>
{};

_LIBCUDACXX_END_NAMESPACE_STD

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion
#if _CCCL_STD_VER >= 2017
namespace std
{
template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_size<tuple<Ts...>>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_element<Id, tuple<Ts...>>
{};
} // namespace std
#endif // _CCCL_STD_VER >= 2017

#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA

#include <thrust/detail/tuple.inl>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! \cond
 */

struct null_type;

/*! \endcond
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <size_t N, class T> struct tuple_element;

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <class T> struct tuple_size;


// get function for non-const cons-lists, returns a reference to the element

/*! The \p get function returns a reference to a \p tuple element of
 *  interest.
 *
 *  \param t A reference to a \p tuple of interest.
 *  \return A reference to \p t's <tt>N</tt>th element.
 *
 *  \tparam N The index of the element of interest.
 *
 *  The following code snippet demonstrates how to use \p get to print
 *  the value of a \p tuple element.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  thrust::tuple<int, const char *> t(13, "thrust");
 *
 *  std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
 *  \endcode
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class HT, class TT>
THRUST_HOST_DEVICE
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::non_const_type
get(detail::cons<HT, TT>& t);


/*! The \p get function returns a \c const reference to a \p tuple element of
 *  interest.
 *
 *  \param t A reference to a \p tuple of interest.
 *  \return A \c const reference to \p t's <tt>N</tt>th element.
 *
 *  \tparam N The index of the element of interest.
 *
 *  The following code snippet demonstrates how to use \p get to print
 *  the value of a \p tuple element.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  thrust::tuple<int, const char *> t(13, "thrust");
 *
 *  std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
 *  \endcode
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class HT, class TT>
THRUST_HOST_DEVICE
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::const_type
get(const detail::cons<HT, TT>& t);


#if THRUST_CPP_DIALECT >= 2017
/*! Constructs a \p tuple from a variadic list of types \p Ts, allowing the \p tuple to deduce 
 *  its type as \p tuple<Ts...> based on the types of the provided arguments.
 *
 *  \tparam Ts... The parameter pack of types that will determine the tuple's type.
 *  \note This deduction guide enables automatic type deduction for variadic arguments 
 *        when constructing a \p tuple.
 *  \see tuple
 */
template <class... Ts>
THRUST_HOST_DEVICE tuple(Ts...) -> tuple<Ts...>;

/*! \cond
 */

/*! A \p pair is a structure template holding two elements of types \p T1 and \p T2.
 *
 *  \tparam T1 The type of the first element in the \p pair.
 *  \tparam T2 The type of the second element in the \p pair.
 *  \note \p pair is used to store two heterogeneous values and can be converted to a \p tuple.
 *  \see tuple
 */
template <class T1, class T2>
struct pair;

/*! \endcond
 */

/*! Constructs a \p tuple from a \p pair<T1,T2>, unpacking its elements to initialize
 *  the tuple as \p tuple<T1,T2>.
 *
 *  \tparam T1 The type of the first element in the \p pair.
 *  \tparam T2 The type of the second element in the \p pair.
 *  \note This deduction guide allows a \p tuple to be created directly from a \p pair,
 *        simplifying the type conversion.
 *  \see pair
 *  \see tuple
 */
template <class T1, class T2>
THRUST_HOST_DEVICE tuple(pair<T1, T2>) -> tuple<T1, T2>;
#endif

/*! \brief \p tuple is a class template that can be instantiated with up to ten
 *  arguments. Each template argument specifies the type of element in the \p
 *  tuple. Consequently, tuples are heterogeneous, fixed-size collections of
 *  values. An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class T0, class T1, class T2, class T3, class T4,
          class T5, class T6, class T7, class T8, class T9>
  class tuple
  /*! \cond
   */
    : public detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
  /*! \endcond
   */
{
  /*! \cond
   */

  private:
  using inherited = typename detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type;

  /*! \endcond
   */

  public:

  /*! \p tuple's no-argument constructor initializes each element.
   */
  inline THRUST_HOST_DEVICE
  tuple(void) {}

  /*! \p tuple's one-argument constructor copy constructs the first element from the given parameter
   *     and intializes all other elements.
   *  \param t0 The value to assign to this \p tuple's first element.
   */
  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0)
    : inherited(t0,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  /*! \p tuple's one-argument constructor copy constructs the first two elements from the given parameters
   *     and intializes all other elements.
   *  \param t0 The value to assign to this \p tuple's first element.
   *  \param t1 The value to assign to this \p tuple's second element.
   *  \note \p tuple's constructor has ten variants of this form, the rest of which are ommitted here for brevity.
   */
  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1)
    : inherited(t0, t1,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  /*! \cond
   */

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2)
    : inherited(t0, t1, t2,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3)
    : inherited(t0, t1, t2, t3,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4)
    : inherited(t0, t1, t2, t3, t4,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5)
    : inherited(t0, t1, t2, t3, t4, t5,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6)
    : inherited(t0, t1, t2, t3, t4, t5, t6,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7,
        typename access_traits<T8>::parameter_type t8)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8,
                static_cast<const null_type&>(null_type())) {}

  inline THRUST_HOST_DEVICE
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7,
        typename access_traits<T8>::parameter_type t8,
        typename access_traits<T9>::parameter_type t9)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9) {}


  template<class U1, class U2>
  inline THRUST_HOST_DEVICE
  tuple(const detail::cons<U1, U2>& p) : inherited(p) {}

  THRUST_EXEC_CHECK_DISABLE
  template <class U1, class U2>
  inline THRUST_HOST_DEVICE
  tuple& operator=(const detail::cons<U1, U2>& k)
  {
    inherited::operator=(k);
    return *this;
  }

  /*! \endcond
   */

  /*! This assignment operator allows assigning the first two elements of this \p tuple from a \p pair.
   *  \param k A \p pair to assign from.
   */
  THRUST_EXEC_CHECK_DISABLE
  template <class U1, class U2>
  THRUST_HOST_DEVICE inline
  tuple& operator=(const thrust::pair<U1, U2>& k) {
    //BOOST_STATIC_ASSERT(length<tuple>::value == 2);// check_length = 2
    this->head = k.first;
    this->tail.head = k.second;
    return *this;
  }

  /*! \p swap swaps the elements of two <tt>tuple</tt>s.
   *
   *  \param t The other <tt>tuple</tt> with which to swap.
   */
  inline THRUST_HOST_DEVICE
  void swap(tuple &t)
  {
    inherited::swap(t);
  }
};

/*! \cond
 */

template <>
class tuple<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>  :
  public null_type
{
public:
  using inherited = null_type;
};

/*! \endcond
 */


/*! This version of \p make_tuple creates a new \c tuple object from a
 *  single object.
 *
 *  \param t0 The object to copy from.
 *  \return A \p tuple object with a single member which is a copy of \p t0.
 */
template<class T0>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0>::type
    make_tuple(const T0& t0);

/*! This version of \p make_tuple creates a new \c tuple object from two
 *  objects.
 *
 *  \param t0 The first object to copy from.
 *  \param t1 The second object to copy from.
 *  \return A \p tuple object with two members which are copies of \p t0
 *          and \p t1.
 *
 *  \note \p make_tuple has ten variants, the rest of which are omitted here
 *        for brevity.
 */
template<class T0, class T1>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1>::type
    make_tuple(const T0& t0, const T1& t1);

/*! This version of \p tie creates a new \c tuple whose single element is
 *  a reference which refers to this function's argument.
 *
 *  \param t0 The object to reference.
 *  \return A \p tuple object with one member which is a reference to \p t0.
 */
template<typename T0>
THRUST_HOST_DEVICE inline
tuple<T0&> tie(T0& t0);

/*! This version of \p tie creates a new \c tuple of references object which
 *  refers to this function's arguments.
 *
 *  \param t0 The first object to reference.
 *  \param t1 The second object to reference.
 *  \return A \p tuple object with two members which are references to \p t0
 *          and \p t1.
 *
 *  \note \p tie has ten variants, the rest of which are omitted here for
 *           brevity.
 */
template<typename T0, typename T1>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&> tie(T0& t0, T1& t1);

/*! \p swap swaps the contents of two <tt>tuple</tt>s.
 *
 *  \param x The first \p tuple to swap.
 *  \param y The second \p tuple to swap.
 */
template<
  typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9,
  typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9
>
inline THRUST_HOST_DEVICE
void swap(tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> &x,
          tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9> &y);



/*! \cond
 */

template<class T0, class T1, class T2>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2);

template<class T0, class T1, class T2, class T3>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3);

template<class T0, class T1, class T2, class T3, class T4>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4);

template<class T0, class T1, class T2, class T3, class T4, class T5>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
THRUST_HOST_DEVICE inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9);

template<typename T0, typename T1, typename T2>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&> tie(T0 &t0, T1 &t1, T2 &t2);

template<typename T0, typename T1, typename T2, typename T3>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3);

template<typename T0, typename T1, typename T2, typename T3, typename T4>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&,T4&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
THRUST_HOST_DEVICE inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9);


THRUST_HOST_DEVICE inline
bool operator==(const null_type&, const null_type&);

THRUST_HOST_DEVICE inline
bool operator>=(const null_type&, const null_type&);

THRUST_HOST_DEVICE inline
bool operator<=(const null_type&, const null_type&);

THRUST_HOST_DEVICE inline
bool operator!=(const null_type&, const null_type&);

THRUST_HOST_DEVICE inline
bool operator<(const null_type&, const null_type&);

THRUST_HOST_DEVICE inline
bool operator>(const null_type&, const null_type&);

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

#if THRUST_CPP_DIALECT >= 2017
namespace std
{
  template <class... Ts>
  struct tuple_size<THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_size<tuple<Ts...>>
  {};

  template <size_t Id, class... Ts>
  struct tuple_element<Id, THRUST_NS_QUALIFIER::tuple<Ts...>> : tuple_element<Id, tuple<Ts...>>
  {};

  template <>
  struct tuple_size<tuple<THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<>>
  {};

  template <class T0>
  struct tuple_size<tuple<T0,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0>>
  {};

  template <class T0, class T1>
  struct tuple_size<tuple<T0,
                          T1,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1>>
  {};

  template <class T0, class T1, class T2>
  struct tuple_size<tuple<T0,
                          T1,
                          T2,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2>>
  {};

  template <class T0, class T1, class T2, class T3>
  struct tuple_size<tuple<T0,
                          T1,
                          T2,
                          T3,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3>>
  {};

  template <class T0, class T1, class T2, class T3, class T4>
  struct tuple_size<tuple<T0,
                          T1,
                          T2,
                          T3,
                          T4,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4>>
  {};

  template <class T0, class T1, class T2, class T3, class T4, class T5>
  struct tuple_size<tuple<T0,
                          T1,
                          T2,
                          T3,
                          T4,
                          T5,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type,
                          THRUST_NS_QUALIFIER::null_type>> : tuple_size<tuple<T0, T1, T2, T3, T4, T5>>
  {};

  template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
  struct tuple_size<
    tuple<T0, T1, T2, T3, T4, T5, T6, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>>
      : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6>>
  {};

  template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
  struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, THRUST_NS_QUALIFIER::null_type, THRUST_NS_QUALIFIER::null_type>>
      : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7>>
  {};

  template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
  struct tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, THRUST_NS_QUALIFIER::null_type>>
      : tuple_size<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8>>
  {};
} // namespace std
#endif // THRUST_CPP_DIALECT >= 2017

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
