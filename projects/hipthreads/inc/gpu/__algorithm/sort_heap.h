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

#ifndef __GPU___ALGORITHM_SORT_HEAP_H
#define __GPU___ALGORITHM_SORT_HEAP_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
void __sort_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp) {
  __comp_ref_type<_Compare> __comp_ref = __comp;

  using difference_type = typename std::iterator_traits<_RandomAccessIterator>::difference_type;
  for (difference_type __n = __last - __first; __n > 1; --__last, (void) --__n)
    gpu::__pop_heap<_AlgPolicy>(__first, __last, __comp_ref, __n);
}

template <class _RandomAccessIterator, class _Compare>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
void sort_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  static_assert(std::is_copy_constructible<_RandomAccessIterator>::value, "Iterators must be copy constructible.");
  static_assert(std::is_copy_assignable<_RandomAccessIterator>::value, "Iterators must be copy assignable.");

  gpu::__sort_heap<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
void sort_heap(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  gpu::sort_heap(std::move(__first), std::move(__last),
      __less<typename std::iterator_traits<_RandomAccessIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_SORT_HEAP_H
