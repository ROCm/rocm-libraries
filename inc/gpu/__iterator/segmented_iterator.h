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

#ifndef __GPU___SEGMENTED_ITERATOR_H
#define __GPU___SEGMENTED_ITERATOR_H

// Segmented iterators are iterators over (not necessarily contiguous) sub-ranges.
//
// For example, std::deque stores its data into multiple blocks of contiguous memory,
// which are not stored contiguously themselves. The concept of segmented iterators
// allows algorithms to operate over these multi-level iterators natively, opening the
// door to various optimizations. See http://lafstern.org/matt/segmented.pdf for details.
//
// If __segmented_iterator_traits can be instantiated, the following functions and associated types must be provided:
// - Traits::__local_iterator
//   The type of iterators used to iterate inside a segment.
//
// - Traits::__segment_iterator
//   The type of iterators used to iterate over segments.
//   Segment iterators can be forward iterators or bidirectional iterators, depending on the
//   underlying data structure.
//
// - static __segment_iterator Traits::__segment(It __it)
//   Returns an iterator to the segment that the provided iterator is in.
//
// - static __local_iterator Traits::__local(It __it)
//   Returns the local iterator pointing to the element that the provided iterator points to.
//
// - static __local_iterator Traits::__begin(__segment_iterator __it)
//   Returns the local iterator to the beginning of the segment that the provided iterator is pointing into.
//
// - static __local_iterator Traits::__end(__segment_iterator __it)
//   Returns the one-past-the-end local iterator to the segment that the provided iterator is pointing into.
//
// - static It Traits::__compose(__segment_iterator, __local_iterator)
//   Returns the iterator composed of the segment iterator and local iterator.

#include "gpu/__config"

#include <cstddef>
#include <type_traits>

namespace gpu {

template <class _Iterator>
struct __segmented_iterator_traits;
/* exposition-only:
{
  using __segment_iterator = ...;
  using __local_iterator   = ...;

  static __segment_iterator __segment(_Iterator);
  static __local_iterator __local(_Iterator);
  static __local_iterator __begin(__segment_iterator);
  static __local_iterator __end(__segment_iterator);
  static _Iterator __compose(__segment_iterator, __local_iterator);
};
*/

template <class _Tp, std::size_t = 0>
struct __has_specialization : std::false_type {};

template <class _Tp>
struct __has_specialization<_Tp, sizeof(_Tp) * 0> : std::true_type {};

template <class _Iterator>
using __is_segmented_iterator = __has_specialization<__segmented_iterator_traits<_Iterator> >;

} // namespace gpu

#endif // __GPU___SEGMENTED_ITERATOR_H
