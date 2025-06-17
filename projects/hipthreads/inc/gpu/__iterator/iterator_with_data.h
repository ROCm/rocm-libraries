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

#ifndef __GPU___ITERATOR_ITERATOR_WITH_DATA_H
#define __GPU___ITERATOR_ITERATOR_WITH_DATA_H

#include "gpu/__config"

namespace gpu {

template <forward_iterator _Iterator, class _Data>
class __iterator_with_data {
  _Iterator __iter_{};
  _Data __data_{};

public:
  using value_type      = iter_value_t<_Iterator>;
  using difference_type = iter_difference_t<_Iterator>;

  __device__ _LIBGPU_HIDE_FROM_ABI __iterator_with_data() = default;

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI __iterator_with_data(_Iterator __iter, _Data __data)
      : __iter_(std::move(__iter)), __data_(std::move(__data)) {}

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI _Iterator __get_iter() const { return __iter_; }

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI _Data __get_data() && { return std::move(__data_); }

  __device__ friend constexpr _LIBGPU_HIDE_FROM_ABI bool
  operator==(const __iterator_with_data& __lhs, const __iterator_with_data& __rhs) {
    return __lhs.__iter_ == __rhs.__iter_;
  }

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI __iterator_with_data& operator++() {
    ++__iter_;
    return *this;
  }

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI __iterator_with_data operator++(int) {
    auto __tmp = *this;
    __iter_++;
    return __tmp;
  }

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI __iterator_with_data& operator--()
    requires bidirectional_iterator<_Iterator>
  {
    --__iter_;
    return *this;
  }

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI __iterator_with_data operator--(int)
    requires bidirectional_iterator<_Iterator>
  {
    auto __tmp = *this;
    --__iter_;
    return __tmp;
  }

  __device__ constexpr _LIBGPU_HIDE_FROM_ABI iter_reference_t<_Iterator> operator*() const { return *__iter_; }

  __device__ _LIBGPU_HIDE_FROM_ABI friend constexpr iter_rvalue_reference_t<_Iterator>
  iter_move(const __iterator_with_data& __iter) noexcept(noexcept(ranges::iter_move(__iter.__iter_))) {
    return ranges::iter_move(__iter.__iter_);
  }

  __device__ _LIBGPU_HIDE_FROM_ABI friend constexpr void
  iter_swap(const __iterator_with_data& __lhs,
            const __iterator_with_data& __rhs) noexcept(noexcept(ranges::iter_swap(__lhs.__iter_, __rhs.__iter_)))
    requires indirectly_swappable<_Iterator>
  {
    return ranges::iter_swap(__lhs.__data_, __rhs.__iter_);
  }
};

} // namespace gpu

#endif // __GPU___ITERATOR_ITERATOR_WITH_DATA_H
