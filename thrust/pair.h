/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

/*! \file pair.h
 *  \brief A type encapsulating a heterogeneous pair of elements
 */

#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/std/utility>
#elif defined(__has_include)
#  if __has_include(<cuda/std/utility>)
#    include <cuda/std/utility>
#  endif // __has_include(<cuda/std/utility>)
#endif // THRUST_DEVICE_SYSTEM

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns either the type of a \p pair's
 *  \c first_type or \c second_type in its nested type, \c type.
 *
 *  \tparam N This parameter selects the member of interest.
 *  \tparam T A \c pair type of interest.
 */
template <size_t N, class T>
using tuple_element = _CUDA_VSTD::tuple_element<N, T>;

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns \c 2, the number of elements of a \p pair,
 *  in its nested data member, \c value.
 *
 *  \tparam Pair A \c pair type of interest.
 */
template <class T>
using tuple_size = _CUDA_VSTD::tuple_size<T>;

/*! \p pair is a generic data structure encapsulating a heterogeneous
 *  pair of values.
 *
 *  \tparam T1 The type of \p pair's first object type.  There are no
 *          requirements on the type of \p T1. <tt>T1</tt>'s type is
 *          provided by <tt>pair::first_type</tt>.
 *
 *  \tparam T2 The type of \p pair's second object type.  There are no
 *          requirements on the type of \p T2. <tt>T2</tt>'s type is
 *          provided by <tt>pair::second_type</tt>.
 */
template <class T, class U>
struct pair : public _CUDA_VSTD::pair<T, U>
{
  using super_t = _CUDA_VSTD::pair<T, U>;
  using super_t::super_t;

#  if (defined(_CCCL_COMPILER_GCC) && __GNUC__ < 9) || (defined(_CCCL_COMPILER_CLANG) && __clang_major__ < 12)
  // For whatever reason nvcc complains about that constructor being used before being defined in a constexpr variable
  constexpr pair() = default;

  template <class _U1          = T,
            class _U2          = U,
            class _Constraints = typename _CUDA_VSTD::__pair_constraints<T, U>::template __constructible<_U1, _U2>,
            _CUDA_VSTD::__enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_HOST_DEVICE constexpr pair(_U1&& __u1, _U2&& __u2)
      : super_t(_CUDA_VSTD::forward<_U1>(__u1), _CUDA_VSTD::forward<_U2>(__u2))
  {}
#  endif // _CCCL_COMPILER_GCC < 9 || _CCCL_COMPILER_CLANG < 12
};

#  if _CCCL_STD_VER >= 2017
template <class _T1, class _T2>
_CCCL_HOST_DEVICE pair(_T1, _T2) -> pair<_T1, _T2>;
#  endif // _CCCL_STD_VER >= 2017

template <class T1, class T2>
inline _CCCL_HOST_DEVICE
  _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::__is_swappable<T1>::value && _CUDA_VSTD::__is_swappable<T2>::value, void>
  swap(pair<T1, T2>& lhs, pair<T1, T2>& rhs) noexcept(
    (_CUDA_VSTD::__is_nothrow_swappable<T1>::value && _CUDA_VSTD::__is_nothrow_swappable<T2>::value))
{
  lhs.swap(rhs);
}

template <class T1, class T2>
inline _CCCL_HOST_DEVICE
  pair<typename _CUDA_VSTD::__unwrap_ref_decay<T1>::type, typename _CUDA_VSTD::__unwrap_ref_decay<T2>::type>
  make_pair(T1&& t1, T2&& t2)
{
  return pair<typename _CUDA_VSTD::__unwrap_ref_decay<T1>::type, typename _CUDA_VSTD::__unwrap_ref_decay<T2>::type>(
    _CUDA_VSTD::forward<T1>(t1), _CUDA_VSTD::forward<T2>(t2));
}

using _CUDA_VSTD::get;

/*! \} // pair
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class T1, class T2>
struct tuple_size<THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_size<pair<T1, T2>>
{};

template <size_t Id, class T1, class T2>
struct tuple_element<Id, THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_element<Id, pair<T1, T2>>
{};

template <class T1, class T2>
struct __tuple_like_ext<THRUST_NS_QUALIFIER::pair<T1, T2>> : true_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion
#  if _CCCL_STD_VER >= 2017

#    include <utility>

namespace std
{
template <class T1, class T2>
struct tuple_size<THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_size<pair<T1, T2>>
{};

template <size_t Id, class T1, class T2>
struct tuple_element<Id, THRUST_NS_QUALIFIER::pair<T1, T2>> : tuple_element<Id, pair<T1, T2>>
{};
} // namespace std
#  endif // _CCCL_STD_VER >= 2017

#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA

#  include <utility>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

/*! \p pair is a generic data structure encapsulating a heterogeneous
 *  pair of values.
 *
 *  \tparam T1 The type of \p pair's first object type.  There are no
 *          requirements on the type of \p T1. <tt>T1</tt>'s type is
 *          provided by <tt>pair::first_type</tt>.
 *
 *  \tparam T2 The type of \p pair's second object type.  There are no
 *          requirements on the type of \p T2. <tt>T2</tt>'s type is
 *          provided by <tt>pair::second_type</tt>.
 */
template <typename T1, typename T2>
struct pair
{
  /*! \p first_type is the type of \p pair's first object type.
   */
  using first_type = T1;

  /*! \p second_type is the type of \p pair's second object type.
   */
  using second_type = T2;

  /*! The \p pair's first object.
   */
  first_type first;

  /*! The \p pair's second object.
   */
  second_type second;

  /*! \p pair's default constructor constructs \p first
   *  and \p second using \c first_type & \c second_type's
   *  default constructors, respectively.
   */
  THRUST_HOST_DEVICE pair(void);

  /*! This constructor accepts two objects to copy into this \p pair.
   *
   *  \param x The object to copy into \p first.
   *  \param y The object to copy into \p second.
   */
  inline THRUST_HOST_DEVICE pair(const T1& x, const T2& y);

  /*! This copy constructor copies from a \p pair whose types are
   *  convertible to this \p pair's \c first_type and \c second_type,
   *  respectively.
   *
   *  \param p The \p pair to copy from.
   *
   *  \tparam U1 is convertible to \c first_type.
   *  \tparam U2 is convertible to \c second_type.
   */
  template <typename U1, typename U2>
  inline THRUST_HOST_DEVICE pair(const pair<U1, U2>& p);

  /*! This copy constructor copies from a <tt>std::pair</tt> whose types are
   *  convertible to this \p pair's \c first_type and \c second_type,
   *  respectively.
   *
   *  \param p The <tt>std::pair</tt> to copy from.
   *
   *  \tparam U1 is convertible to \c first_type.
   *  \tparam U2 is convertible to \c second_type.
   */
  template <typename U1, typename U2>
  inline THRUST_HOST_DEVICE pair(const std::pair<U1, U2>& p);

  /*! \p swap swaps the elements of two <tt>pair</tt>s.
   *
   *  \param p The other <tt>pair</tt> with which to swap.
   */
  inline THRUST_HOST_DEVICE void swap(pair& p);
}; // end pair

/*! This operator tests two \p pairs for equality.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>x.first == y.first && x.second == y.second</tt>.
 *
 *  \tparam T1 is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality
 * Comparable</a>. \tparam T2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE bool operator==(const pair<T1, T2>& x, const pair<T1, T2>& y);

/*! This operator tests two pairs for ascending ordering.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>x.first < y.first || (!(y.first < x.first) && x.second < y.second)</tt>.
 *
 *  \tparam T1 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>. \tparam T2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE bool operator<(const pair<T1, T2>& x, const pair<T1, T2>& y);

/*! This operator tests two pairs for inequality.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>!(x == y)</tt>.
 *
 *  \tparam T1 is a model of <a href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality
 * Comparable</a>. \tparam T2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/concepts/equality_comparable">Equality Comparable</a>.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE bool operator!=(const pair<T1, T2>& x, const pair<T1, T2>& y);

/*! This operator tests two pairs for descending ordering.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>y < x</tt>.
 *
 *  \tparam T1 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>. \tparam T2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE bool operator>(const pair<T1, T2>& x, const pair<T1, T2>& y);

/*! This operator tests two pairs for ascending ordering or equivalence.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>!(y < x)</tt>.
 *
 *  \tparam T1 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>. \tparam T2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE bool operator<=(const pair<T1, T2>& x, const pair<T1, T2>& y);

/*! This operator tests two pairs for descending ordering or equivalence.
 *
 *  \param x The first \p pair to compare.
 *  \param y The second \p pair to compare.
 *  \return \c true if and only if <tt>!(x < y)</tt>.
 *
 *  \tparam T1 is a model of <a href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan
 * Comparable</a>. \tparam T2 is a model of <a
 * href="https://en.cppreference.com/w/cpp/named_req/LessThanComparable">LessThan Comparable</a>.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE bool operator>=(const pair<T1, T2>& x, const pair<T1, T2>& y);

/*! \p swap swaps the contents of two <tt>pair</tt>s.
 *
 *  \param x The first \p pair to swap.
 *  \param y The second \p pair to swap.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE void swap(pair<T1, T2>& x, pair<T1, T2>& y);

/*! This convenience function creates a \p pair from two objects.
 *
 *  \param x The first object to copy from.
 *  \param y The second object to copy from.
 *  \return A newly-constructed \p pair copied from \p a and \p b.
 *
 *  \tparam T1 There are no requirements on the type of \p T1.
 *  \tparam T2 There are no requirements on the type of \p T2.
 */
template <typename T1, typename T2>
inline THRUST_HOST_DEVICE pair<T1, T2> make_pair(T1 x, T2 y);

/*! \cond
 */

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns either the type of a \p pair's
 *  \c first_type or \c second_type in its nested type, \c type.
 *
 *  \tparam N This parameter selects the member of interest.
 *  \tparam T A \c pair type of interest.
 */
template <size_t N, class T>
struct tuple_element;

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns \c 2, the number of elements of a \p pair,
 *  in its nested data member, \c value.
 *
 *  \tparam Pair A \c pair type of interest.
 */
template <typename Pair>
struct tuple_size;

/*! \endcond
 */

/*! This convenience function returns a reference to either the first or
 *  second member of a \p pair.
 *
 *  \param p The \p pair of interest.
 *  \return \c p.first or \c p.second, depending on the template
 *          parameter.
 *
 *  \tparam N This parameter selects the member of interest.
 */
// XXX comment out these prototypes as a WAR to a problem on MSVC 2005
// template<unsigned int N, typename T1, typename T2>
//  inline __host__ __device__
//    typename tuple_element<N, pair<T1,T2> >::type &
//      get(pair<T1,T2> &p);

/*! This convenience function returns a const reference to either the
 *  first or second member of a \p pair.
 *
 *  \param p The \p pair of interest.
 *  \return \c p.first or \c p.second, depending on the template
 *          parameter.
 *
 *  \tparam i This parameter selects the member of interest.
 */
// XXX comment out these prototypes as a WAR to a problem on MSVC 2005
// template<int N, typename T1, typename T2>
//  inline __host__ __device__
//    const typename tuple_element<N, pair<T1,T2> >::type &
//      get(const pair<T1,T2> &p);

/*! \} // pair
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

#  include <thrust/detail/pair.inl>

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
