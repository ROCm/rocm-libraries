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

#ifndef __GPU___ALGORITHM_UPPER_BOUND_H
#define __GPU___ALGORITHM_UPPER_BOUND_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Compare, class _Iter, class _Sent, class _Tp, class _Proj>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _Iter
__upper_bound(_Iter __first, _Sent __last, const _Tp& __value, _Compare&& __comp, _Proj&& __proj) {
  auto __len = _IterOps<_AlgPolicy>::distance(__first, __last);
  while (__len != 0) {
    auto __half_len = std::__half_positive(__len);
    auto __mid      = _IterOps<_AlgPolicy>::next(__first, __half_len);
    if (std::__invoke(__comp, __value, std::__invoke(__proj, *__mid)))
      __len = __half_len;
    else {
      __first = ++__mid;
      __len -= __half_len + 1;
    }
  }
  return __first;
}

template <class _ForwardIterator, class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
upper_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp) {
  static_assert(std::is_copy_constructible<_ForwardIterator>::value,
                "Iterator has to be copy constructible");
  return gpu::__upper_bound<_ClassicAlgPolicy>(
      std::move(__first), std::move(__last), __value, std::move(__comp), std::__identity());
}

template <class _ForwardIterator, class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
upper_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  return gpu::upper_bound(
      std::move(__first),
      std::move(__last),
      __value,
      __less<_Tp, typename std::iterator_traits<_ForwardIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_UPPER_BOUND_H
