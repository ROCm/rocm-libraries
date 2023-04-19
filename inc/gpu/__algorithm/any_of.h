// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_ANY_OF_H
#define __GPU___ALGORITHM_ANY_OF_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _Predicate>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
any_of(_InputIterator __first, _InputIterator __last, _Predicate __pred) {
  for (; __first != __last; ++__first)
    if (__pred(*__first))
      return true;
  return false;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_ANY_OF_H
