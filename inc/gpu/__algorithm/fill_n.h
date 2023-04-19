//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_FILL_N_H
#define __GPU___ALGORITHM_FILL_N_H

#include "gpu/__config"
#include "gpu/__utility/convert_to_integral.h"

namespace gpu {

// fill_n isn't specialized for std::memset, because the compiler already optimizes the loop to a call to std::memset.

template <class _OutputIterator, class _Size, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
__fill_n(_OutputIterator __first, _Size __n, const _Tp& __value)
{
    for (; __n > 0; ++__first, (void) --__n)
        *__first = __value;
    return __first;
}

template <class _OutputIterator, class _Size, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
fill_n(_OutputIterator __first, _Size __n, const _Tp& __value)
{
   return gpu::__fill_n(__first, gpu::__convert_to_integral(__n), __value);
}

} // namespace gpu

#endif // __GPU___ALGORITHM_FILL_N_H
