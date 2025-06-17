// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __GPU___ALGORITHM_FILL_H
#define __GPU___ALGORITHM_FILL_H

#include "gpu/__config"

namespace gpu {

// fill isn't specialized for std::memset, because the compiler already optimizes the loop to a call to std::memset.

template <class _ForwardIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
__fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, std::forward_iterator_tag)
{
    for (; __first != __last; ++__first)
        *__first = __value;
}

template <class _RandomAccessIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value, std::random_access_iterator_tag)
{
    gpu::fill_n(__first, __last - __first, __value);
}

template <class _ForwardIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    gpu::__fill(__first, __last, __value, typename std::iterator_traits<_ForwardIterator>::iterator_category());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_FILL_H
