//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SET_SYMMETRIC_DIFFERENCE_H
#define __GPU___ALGORITHM_SET_SYMMETRIC_DIFFERENCE_H

#include "gpu/__config"

namespace gpu {

template <class _InIter1, class _InIter2, class _OutIter>
struct __set_symmetric_difference_result {
  _InIter1 __in1_;
  _InIter2 __in2_;
  _OutIter __out_;

  // need a constructor as C++03 aggregate init is hard
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
  __set_symmetric_difference_result(_InIter1&& __in_iter1, _InIter2&& __in_iter2, _OutIter&& __out_iter)
      : __in1_(std::move(__in_iter1)), __in2_(std::move(__in_iter2)), __out_(std::move(__out_iter)) {}
};

template <class _AlgPolicy, class _Compare, class _InIter1, class _Sent1, class _InIter2, class _Sent2, class _OutIter>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 __set_symmetric_difference_result<_InIter1, _InIter2, _OutIter>
__set_symmetric_difference(
    _InIter1 __first1, _Sent1 __last1, _InIter2 __first2, _Sent2 __last2, _OutIter __result, _Compare&& __comp) {
  while (__first1 != __last1) {
    if (__first2 == __last2) {
      auto __ret1 = gpu::__copy<_AlgPolicy>(std::move(__first1), std::move(__last1), std::move(__result));
      return __set_symmetric_difference_result<_InIter1, _InIter2, _OutIter>(
          std::move(__ret1.first), std::move(__first2), std::move((__ret1.second)));
    }
    if (__comp(*__first1, *__first2)) {
      *__result = *__first1;
      ++__result;
      ++__first1;
    } else {
      if (__comp(*__first2, *__first1)) {
        *__result = *__first2;
        ++__result;
      } else {
        ++__first1;
      }
      ++__first2;
    }
  }
  auto __ret2 = gpu::__copy<_AlgPolicy>(std::move(__first2), std::move(__last2), std::move(__result));
  return __set_symmetric_difference_result<_InIter1, _InIter2, _OutIter>(
      std::move(__first1), std::move(__ret2.first), std::move((__ret2.second)));
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator set_symmetric_difference(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp) {
  return gpu::__set_symmetric_difference<_ClassicAlgPolicy, __comp_ref_type<_Compare> >(
             std::move(__first1),
             std::move(__last1),
             std::move(__first2),
             std::move(__last2),
             std::move(__result),
             __comp)
      .__out_;
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator set_symmetric_difference(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result) {
  return gpu::set_symmetric_difference(
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      std::move(__result),
      __less<typename std::iterator_traits<_InputIterator1>::value_type,
             typename std::iterator_traits<_InputIterator2>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_SET_SYMMETRIC_DIFFERENCE_H
