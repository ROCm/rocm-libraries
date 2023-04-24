// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_EQUAL_H
#define __GPU___ALGORITHM_EQUAL_H

#include "gpu/__config"

#include <type_traits>

#include "gpu/__iterator/distance.h"
#include "gpu/__type_traits/is_equality_comparable.h"
#include "gpu/__string/constexpr_c_functions.h"

namespace gpu {

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool __equal_iter_impl(
    _InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate& __pred) {
  for (; __first1 != __last1; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      return false;
  return true;
}

template <
    class _Tp,
    class _Up,
    class _BinaryPredicate,
    std::enable_if_t<__is_trivial_equality_predicate<_BinaryPredicate, _Tp, _Up>::value && !std::is_volatile<_Tp>::value &&
                      !std::is_volatile<_Up>::value && __libcpp_is_trivially_equality_comparable<_Tp, _Up>::value,
                  int> = 0>
_LIBGPU_NODISCARD __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
__equal_iter_impl(_Tp* __first1, _Tp* __last1, _Up* __first2, _BinaryPredicate&) {
  return gpu::__constexpr_memcmp(__first1, __first2, (__last1 - __first1) * sizeof(_Tp)) == 0;
}

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred) {
  return gpu::__equal_iter_impl(
      gpu::__unwrap_iter(__first1), gpu::__unwrap_iter(__last1), gpu::__unwrap_iter(__first2), __pred);
}

template <class _InputIterator1, class _InputIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2) {
  return gpu::equal(__first1, __last1, __first2, __equal_to());
}

#if _LIBGPU_STD_VER >= 14
template <class _BinaryPredicate, class _InputIterator1, class _InputIterator2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
__equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2,
        _BinaryPredicate __pred, std::input_iterator_tag, std::input_iterator_tag) {
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      return false;
  return __first1 == __last1 && __first2 == __last2;
}

template <class _Iter1, class _Sent1, class _Iter2, class _Sent2, class _Pred, class _Proj1, class _Proj2>
_LIBGPU_NODISCARD __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool __equal_impl(
    _Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Sent2 __last2, _Pred& __comp, _Proj1& __proj1, _Proj2& __proj2) {
  while (__first1 != __last1 && __first2 != __last2) {
    if (!std::__invoke(__comp, std::__invoke(__proj1, *__first1), std::__invoke(__proj2, *__first2)))
      return false;
    ++__first1;
    ++__first2;
  }
  return __first1 == __last1 && __first2 == __last2;
}

template <class _Tp,
          class _Up,
          class _Pred,
          class _Proj1,
          class _Proj2,
          std::enable_if_t<__is_trivial_equality_predicate<_Pred, _Tp, _Up>::value && __is_identity<_Proj1>::value &&
                            __is_identity<_Proj2>::value && !std::is_volatile<_Tp>::value && !std::is_volatile<_Up>::value &&
                            __libcpp_is_trivially_equality_comparable<_Tp, _Up>::value,
                        int> = 0>
_LIBGPU_NODISCARD __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool __equal_impl(
    _Tp* __first1, _Tp* __last1, _Up* __first2, _Up*, _Pred&, _Proj1&, _Proj2&) {
  return gpu::__constexpr_memcmp(__first1, __first2, (__last1 - __first1) * sizeof(_Tp)) == 0;
}

template <class _BinaryPredicate, class _RandomAccessIterator1, class _RandomAccessIterator2>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
__equal(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
        _RandomAccessIterator2 __last2, _BinaryPredicate __pred, std::random_access_iterator_tag,
        std::random_access_iterator_tag) {
  if (gpu::distance(__first1, __last1) != gpu::distance(__first2, __last2))
    return false;
  __identity __proj;
  return gpu::__equal_impl(
      gpu::__unwrap_iter(__first1),
      gpu::__unwrap_iter(__last1),
      gpu::__unwrap_iter(__first2),
      gpu::__unwrap_iter(__last2),
      __pred,
      __proj,
      __proj);
}

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2,
      _BinaryPredicate __pred) {
  return gpu::__equal<_BinaryPredicate&>(
      __first1, __last1, __first2, __last2, __pred, typename std::iterator_traits<_InputIterator1>::iterator_category(),
      typename std::iterator_traits<_InputIterator2>::iterator_category());
}

template <class _InputIterator1, class _InputIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return gpu::__equal(
      __first1,
      __last1,
      __first2,
      __last2,
      __equal_to(),
      typename std::iterator_traits<_InputIterator1>::iterator_category(),
      typename std::iterator_traits<_InputIterator2>::iterator_category());
}
#endif

} // namespace gpu

#endif // __GPU___ALGORITHM_EQUAL_H
