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

#ifndef __GPU___ALGORITHM_PARTITION_COPY_H
#define __GPU___ALGORITHM_PARTITION_COPY_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _OutputIterator1,
          class _OutputIterator2, class _Predicate>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_OutputIterator1, _OutputIterator2>
partition_copy(_InputIterator __first, _InputIterator __last,
               _OutputIterator1 __out_true, _OutputIterator2 __out_false,
               _Predicate __pred)
{
    for (; __first != __last; ++__first)
    {
        if (__pred(*__first))
        {
            *__out_true = *__first;
            ++__out_true;
        }
        else
        {
            *__out_false = *__first;
            ++__out_false;
        }
    }
    return gpu::pair<_OutputIterator1, _OutputIterator2>(__out_true, __out_false);
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTITION_COPY_H
