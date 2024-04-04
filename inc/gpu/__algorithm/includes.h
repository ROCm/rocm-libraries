//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_INCLUDES_H
#define __GPU___ALGORITHM_INCLUDES_H

#include "gpu/__config"

namespace gpu {

template <class _Iter1, class _Sent1, class _Iter2, class _Sent2, class _Comp, class _Proj1, class _Proj2>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
__includes(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Sent2 __last2,
           _Comp&& __comp, _Proj1&& __proj1, _Proj2&& __proj2) {
  for (; __first2 != __last2; ++__first1) {
    if (__first1 == __last1 || std::__invoke(
          __comp, gpu::__invoke(__proj2, *__first2), gpu::__invoke(__proj1, *__first1)))
      return false;
    if (!gpu::__invoke(__comp, gpu::__invoke(__proj1, *__first1), gpu::__invoke(__proj2, *__first2)))
      ++__first2;
  }
  return true;
}

template <class _InputIterator1, class _InputIterator2, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool includes(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _Compare __comp) {
  static_assert(__is_callable<_Compare, decltype(*__first1), decltype(*__first2)>::value,
      "Comparator has to be callable");

  return gpu::__includes(
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      static_cast<__comp_ref_type<_Compare> >(__comp),
      __identity(),
      __identity());
}

template <class _InputIterator1, class _InputIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
includes(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return gpu::includes(
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      __less<typename std::iterator_traits<_InputIterator1>::value_type,
             typename std::iterator_traits<_InputIterator2>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_INCLUDES_H
