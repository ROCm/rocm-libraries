//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_ROTATE_COPY_H
#define __GPU___ALGORITHM_ROTATE_COPY_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _OutputIterator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
rotate_copy(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _OutputIterator __result)
{
    return gpu::copy(__first, __middle, gpu::copy(__middle, __last, __result));
}

} // namespace gpu

#endif // __GPU___ALGORITHM_ROTATE_COPY_H
