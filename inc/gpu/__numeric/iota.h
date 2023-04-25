// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_IOTA_H
#define __GPU___NUMERIC_IOTA_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
iota(_ForwardIterator __first, _ForwardIterator __last, _Tp __value)
{
    for (; __first != __last; ++__first, (void) ++__value)
        *__first = __value;
}

} // namespace gpu

#endif // __GPU___NUMERIC_IOTA_H
