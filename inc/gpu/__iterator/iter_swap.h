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

#ifndef __GPU___ITERATOR_ITER_SWAP_H
#define __GPU___ITERATOR_ITER_SWAP_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

// [iter.cust.swap]

namespace ranges {
namespace __iter_swap {
  template<class _I1, class _I2>
  void iter_swap(_I1, _I2) = delete;

  template<class _T1, class _T2>
  concept __unqualified_iter_swap =
    (__class_or_enum<std::remove_cv_t<std::remove_reference_t<_T1>>> || __class_or_enum<std::remove_cv_t<std::remove_reference_t<_T2>>>) &&
    requires (_T1&& __x, _T2&& __y) {
      // NOLINTNEXTLINE(libcpp-robust-against-adl) iter_swap ADL calls should only be made through ranges::iter_swap
      iter_swap(std::forward<_T1>(__x), std::forward<_T2>(__y));
    };

  template<class _T1, class _T2>
  concept __readable_swappable =
    indirectly_readable<_T1> && indirectly_readable<_T2> &&
    swappable_with<iter_reference_t<_T1>, iter_reference_t<_T2>>;


  struct __fn {
    // NOLINTBEGIN(libcpp-robust-against-adl) iter_swap ADL calls should only be made through ranges::iter_swap
    template <class _T1, class _T2>
      requires __unqualified_iter_swap<_T1, _T2>
    __device__ _LIBGPU_HIDE_FROM_ABI
    constexpr void operator()(_T1&& __x, _T2&& __y) const
      noexcept(noexcept(iter_swap(std::forward<_T1>(__x), std::forward<_T2>(__y))))
    {
      (void)iter_swap(std::forward<_T1>(__x), std::forward<_T2>(__y));
    }
    // NOLINTEND(libcpp-robust-against-adl)

    template <class _T1, class _T2>
      requires (!__unqualified_iter_swap<_T1, _T2>) &&
               __readable_swappable<_T1, _T2>
    __device__ _LIBGPU_HIDE_FROM_ABI
    constexpr void operator()(_T1&& __x, _T2&& __y) const
      noexcept(noexcept(ranges::swap(*std::forward<_T1>(__x), *std::forward<_T2>(__y))))
    {
      ranges::swap(*std::forward<_T1>(__x), *std::forward<_T2>(__y));
    }

    template <class _T1, class _T2>
      requires (!__unqualified_iter_swap<_T1, _T2> &&
                !__readable_swappable<_T1, _T2>) &&
               indirectly_movable_storable<_T1, _T2> &&
               indirectly_movable_storable<_T2, _T1>
    __device__ _LIBGPU_HIDE_FROM_ABI
    constexpr void operator()(_T1&& __x, _T2&& __y) const
      noexcept(noexcept(iter_value_t<_T2>(ranges::iter_move(__y))) &&
               noexcept(*__y = ranges::iter_move(__x)) &&
               noexcept(*std::forward<_T1>(__x) = gpu::declval<iter_value_t<_T2>>()))
    {
      iter_value_t<_T2> __old(ranges::iter_move(__y));
      *__y = ranges::iter_move(__x);
      *std::forward<_T1>(__x) = std::move(__old);
    }
  };
} // namespace __iter_swap

inline namespace __cpo {
  inline constexpr auto iter_swap = __iter_swap::__fn{};
} // namespace __cpo
} // namespace ranges

template<class _I1, class _I2 = _I1>
concept indirectly_swappable =
  indirectly_readable<_I1> && indirectly_readable<_I2> &&
  requires(const _I1 __i1, const _I2 __i2) {
    ranges::iter_swap(__i1, __i1);
    ranges::iter_swap(__i2, __i2);
    ranges::iter_swap(__i1, __i2);
    ranges::iter_swap(__i2, __i1);
  };

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_ITER_SWAP_H
