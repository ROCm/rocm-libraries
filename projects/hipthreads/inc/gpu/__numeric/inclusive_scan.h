// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_INCLUSIVE_SCAN_H
#define __GPU___NUMERIC_INCLUSIVE_SCAN_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
inclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryOp __b, _Tp __init) {
  for (; __first != __last; ++__first, (void)++__result) {
    __init = __b(__init, *__first);
    *__result = __init;
  }
  return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
inclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryOp __b) {
  if (__first != __last) {
    typename std::iterator_traits<_InputIterator>::value_type __init = *__first;
    *__result++ = __init;
    if (++__first != __last)
      return gpu::inclusive_scan(__first, __last, __result, __b, __init);
  }

  return __result;
}

template <class _InputIterator, class _OutputIterator>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator inclusive_scan(_InputIterator __first,
                                                                                       _InputIterator __last,
                                                                                       _OutputIterator __result) {
  return gpu::inclusive_scan(__first, __last, __result, std::plus<>());
}

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___NUMERIC_INCLUSIVE_SCAN_H
