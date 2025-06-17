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

#ifndef __GPU___ALGORITHM_THREE_WAY_COMP_REF_TYPE_H
#define __GPU___ALGORITHM_THREE_WAY_COMP_REF_TYPE_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

template <class _Comp>
struct __debug_three_way_comp {
  _Comp& __comp_;
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr __debug_three_way_comp(_Comp& __c) : __comp_(__c) {}

  template <class _Tp, class _Up>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr auto operator()(_Tp& __x, _Up& __y) {
    auto __r = __comp_(__x, __y);
    __do_compare_assert(0, __y, __x, __r);
    return __r;
  }

  template <class _LHS, class _RHS, class _Order>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr inline void __do_compare_assert(int, _LHS& __l, _RHS& __r, _Order __o)
    requires __comparison_category<decltype(gpu::declval<_Comp&>()(gpu::declval<_LHS&>(), gpu::declval<_RHS&>()))>
  {
    _Order __expected = __o;
    if (__o == _Order::less)
      __expected = _Order::greater;
    if (__o == _Order::greater)
      __expected = _Order::less;
    _LIBGPU_DEBUG_ASSERT(__comp_(__l, __r) == __expected, "Comparator does not induce a strict weak ordering");
    (void)__l;
    (void)__r;
    (void)__expected;
  }

  template <class _LHS, class _RHS, class _Order>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr inline void __do_compare_assert(long, _LHS&, _RHS&, _Order) {}
};

// Pass the comparator by lvalue reference. Or in debug mode, using a
// debugging wrapper that stores a reference.
#  ifndef _LIBGPU_ENABLE_DEBUG_MODE
template <class _Comp>
using __three_way_comp_ref_type = __debug_three_way_comp<_Comp>;
#  else
template <class _Comp>
using __three_way_comp_ref_type = _Comp&;
#  endif

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_THREE_WAY_COMP_REF_TYPE_H
