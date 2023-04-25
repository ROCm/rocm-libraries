// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_ACCUMULATE_H
#define __GPU___NUMERIC_ACCUMULATE_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_Tp
accumulate(_InputIterator __first, _InputIterator __last, _Tp __init)
{
    for (; __first != __last; ++__first)
#if _LIBGPU_STD_VER >= 20
        __init = std::move(__init) + *__first;
#else
        __init = __init + *__first;
#endif
    return __init;
}

template <class _InputIterator, class _Tp, class _BinaryOperation>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_Tp
accumulate(_InputIterator __first, _InputIterator __last, _Tp __init, _BinaryOperation __binary_op)
{
    for (; __first != __last; ++__first)
#if _LIBGPU_STD_VER >= 20
        __init = __binary_op(std::move(__init), *__first);
#else
        __init = __binary_op(__init, *__first);
#endif
    return __init;
}

} // namespace gpu

#endif // __GPU___NUMERIC_ACCUMULATE_H
