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

#ifndef __GPU___ITERATOR_PREV_H
#define __GPU___ITERATOR_PREV_H

#include "gpu/__config"

#include "gpu/__iterator/advance.h"
#include "gpu/__iterator/iterator_traits.h"

namespace gpu {

template <class _InputIter>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    typename std::enable_if<__is_cpp17_input_iterator<_InputIter>::value, _InputIter>::type
    prev(_InputIter __x, typename std::iterator_traits<_InputIter>::difference_type __n = 1) {
  _LIBGPU_ASSERT(__n <= 0 || __is_cpp17_bidirectional_iterator<_InputIter>::value,
                 "Attempt to prev(it, n) with a positive n on a non-bidirectional iterator");
  gpu::advance(__x, -__n);
  return __x;
}

#if _LIBGPU_STD_VER >= 20

// [range.iter.op.prev]

namespace ranges {
namespace __prev {

struct __fn {
  template <bidirectional_iterator _Ip>
  __device__ _LIBGPU_HIDE_FROM_ABI
  constexpr _Ip operator()(_Ip __x) const {
    --__x;
    return __x;
  }

  template <bidirectional_iterator _Ip>
  __device__ _LIBGPU_HIDE_FROM_ABI
  constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n) const {
    ranges::advance(__x, -__n);
    return __x;
  }

  template <bidirectional_iterator _Ip>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n, _Ip __bound_iter) const {
    ranges::advance(__x, -__n, __bound_iter);
    return __x;
  }
};

} // namespace __prev

inline namespace __cpo {
  inline constexpr auto prev = __prev::__fn{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_PREV_H
