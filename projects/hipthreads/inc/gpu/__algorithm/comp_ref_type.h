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

#ifndef __GPU___ALGORITHM_COMP_REF_TYPE_H
#define __GPU___ALGORITHM_COMP_REF_TYPE_H

#include "gpu/__config"

namespace gpu {

template <class _Compare>
struct __debug_less
{
    _Compare &__comp_;
    _LIBGPU_CONSTEXPR_SINCE_CXX14
    __debug_less(_Compare& __c) : __comp_(__c) {}

    template <class _Tp, class _Up>
    _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _Tp& __x,  const _Up& __y)
    {
        bool __r = __comp_(__x, __y);
        if (__r)
            __do_compare_assert(0, __y, __x);
        return __r;
    }

    template <class _Tp, class _Up>
    _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(_Tp& __x,  _Up& __y)
    {
        bool __r = __comp_(__x, __y);
        if (__r)
            __do_compare_assert(0, __y, __x);
        return __r;
    }

    template <class _LHS, class _RHS>
    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX14
    inline _LIBGPU_INLINE_VISIBILITY
    decltype((void)gpu::declval<_Compare&>()(
        gpu::declval<_LHS &>(), gpu::declval<_RHS &>()))
    __do_compare_assert(int, _LHS & __l, _RHS & __r) {
        _LIBGPU_DEBUG_ASSERT(!__comp_(__l, __r),
            "Comparator does not induce a strict weak ordering");
        (void)__l;
        (void)__r;
    }

    template <class _LHS, class _RHS>
    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX14
    inline _LIBGPU_INLINE_VISIBILITY
    void __do_compare_assert(long, _LHS &, _RHS &) {}
};

// Pass the comparator by lvalue reference. Or in debug mode, using a
// debugging wrapper that stores a reference.
#ifdef _LIBGPU_ENABLE_DEBUG_MODE
template <class _Comp>
using __comp_ref_type = __debug_less<_Comp>;
#else
template <class _Comp>
using __comp_ref_type = _Comp&;
#endif

} // namespace gpu

#endif // __GPU___ALGORITHM_COMP_REF_TYPE_H
