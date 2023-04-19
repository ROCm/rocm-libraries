// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_FOR_EACH_N_H
#define __GPU___ALGORITHM_FOR_EACH_N_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _InputIterator, class _Size, class _Function>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _InputIterator for_each_n(_InputIterator __first,
                                                                                         _Size __orig_n,
                                                                                         _Function __f) {
  typedef decltype(gpu::__convert_to_integral(__orig_n)) _IntegralSize;
  _IntegralSize __n = __orig_n;
  while (__n > 0) {
    __f(*__first);
    ++__first;
    --__n;
  }
  return __first;
}

#endif

} // namespace gpu

#endif // __GPU___ALGORITHM_FOR_EACH_N_H
