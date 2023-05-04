// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_EXCLUSIVE_SCAN_H
#define __GPU___NUMERIC_EXCLUSIVE_SCAN_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
exclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init, _BinaryOp __b) {
  if (__first != __last) {
    _Tp __tmp(__b(__init, *__first));
    while (true) {
      *__result = std::move(__init);
      ++__result;
      ++__first;
      if (__first == __last)
        break;
      __init = std::move(__tmp);
      __tmp = __b(__init, *__first);
    }
  }
  return __result;
}

template <class _InputIterator, class _OutputIterator, class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
exclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init) {
  return gpu::exclusive_scan(__first, __last, __result, __init, std::plus<>());
}

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___NUMERIC_EXCLUSIVE_SCAN_H
