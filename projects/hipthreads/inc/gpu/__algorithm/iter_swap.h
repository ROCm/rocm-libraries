//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_ITER_SWAP_H
#define __GPU___ALGORITHM_ITER_SWAP_H

#include "gpu/__config"
#include "gpu/__utility/declval.h"

namespace gpu {

template <class _ForwardIterator1, class _ForwardIterator2>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 void iter_swap(_ForwardIterator1 __a,
                                                                              _ForwardIterator2 __b)
    //                                  _NOEXCEPT_(_NOEXCEPT_(swap(*__a, *__b)))
    _NOEXCEPT_(_NOEXCEPT_(swap(*gpu::declval<_ForwardIterator1>(), *gpu::declval<_ForwardIterator2>()))) {
  swap(*__a, *__b);
}

} // namespace gpu

#endif // __GPU___ALGORITHM_ITER_SWAP_H
