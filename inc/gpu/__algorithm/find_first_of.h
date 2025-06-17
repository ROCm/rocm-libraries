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

#ifndef __GPU___ALGORITHM_FIND_FIRST_OF_H
#define __GPU___ALGORITHM_FIND_FIRST_OF_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
__device__ _LIBGPU_HIDE_FROM_ABI
_LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator1 __find_first_of_ce(_ForwardIterator1 __first1,
                                                                   _ForwardIterator1 __last1,
                                                                   _ForwardIterator2 __first2,
                                                                   _ForwardIterator2 __last2,
                                                                   _BinaryPredicate&& __pred) {
  for (; __first1 != __last1; ++__first1)
    for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
      if (__pred(*__first1, *__j))
        return __first1;
  return __last1;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator1
find_first_of(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
              _ForwardIterator2 __last2, _BinaryPredicate __pred) {
  return std::__find_first_of_ce(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2) {
  return std::__find_first_of_ce(__first1, __last1, __first2, __last2, __equal_to());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_FIND_FIRST_OF_H
