// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_FIND_H
#define __GPU___ALGORITHM_FIND_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _Tp>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _InputIterator
find(_InputIterator __first, _InputIterator __last, const _Tp& __value) {
  for (; __first != __last; ++__first)
    if (*__first == __value)
      break;
  return __first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_FIND_H
