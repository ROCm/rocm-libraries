//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SET_DIFFERENCE_H
#define __GPU___ALGORITHM_SET_DIFFERENCE_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Comp, class _InIter1, class _Sent1, class _InIter2, class _Sent2, class _OutIter>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<std::remove_cv_t<std::remove_reference_t<_InIter1>>, std::remove_cv_t<std::remove_reference_t<_OutIter>> >
__set_difference(
    _InIter1&& __first1, _Sent1&& __last1, _InIter2&& __first2, _Sent2&& __last2, _OutIter&& __result, _Comp&& __comp) {
  while (__first1 != __last1 && __first2 != __last2) {
    if (__comp(*__first1, *__first2)) {
      *__result = *__first1;
      ++__first1;
      ++__result;
    } else if (__comp(*__first2, *__first1)) {
      ++__first2;
    } else {
      ++__first1;
      ++__first2;
    }
  }
  return gpu::__copy<_AlgPolicy>(std::move(__first1), std::move(__last1), std::move(__result));
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator set_difference(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp) {
  return gpu::__set_difference<_ClassicAlgPolicy, __comp_ref_type<_Compare> >(
      __first1, __last1, __first2, __last2, __result, __comp)
      .second;
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator set_difference(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result) {
  return gpu::__set_difference<_ClassicAlgPolicy>(
      __first1,
      __last1,
      __first2,
      __last2,
      __result,
      __less<typename std::iterator_traits<_InputIterator1>::value_type,
             typename std::iterator_traits<_InputIterator2>::value_type>()).second;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_SET_DIFFERENCE_H
