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

#ifndef __GPU___ALGORITHM_REVERSE_H
#define __GPU___ALGORITHM_REVERSE_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _BidirectionalIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
__reverse_impl(_BidirectionalIterator __first, _BidirectionalIterator __last, std::bidirectional_iterator_tag)
{
    while (__first != __last)
    {
        if (__first == --__last)
            break;
        _IterOps<_AlgPolicy>::iter_swap(__first, __last);
        ++__first;
    }
}

template <class _AlgPolicy, class _RandomAccessIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
__reverse_impl(_RandomAccessIterator __first, _RandomAccessIterator __last, std::random_access_iterator_tag)
{
    if (__first != __last)
        for (; __first < --__last; ++__first)
            _IterOps<_AlgPolicy>::iter_swap(__first, __last);
}

template <class _AlgPolicy, class _BidirectionalIterator, class _Sentinel>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
void __reverse(_BidirectionalIterator __first, _Sentinel __last) {
  using _IterCategory = typename _IterOps<_AlgPolicy>::template __iterator_category<_BidirectionalIterator>;
  std::__reverse_impl<_AlgPolicy>(std::move(__first), std::move(__last), _IterCategory());
}

template <class _BidirectionalIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
reverse(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  gpu::__reverse<_ClassicAlgPolicy>(std::move(__first), std::move(__last));
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REVERSE_H
