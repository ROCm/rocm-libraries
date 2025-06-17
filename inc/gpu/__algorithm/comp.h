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

#ifndef __GPU___ALGORITHM_COMP_H
#define __GPU___ALGORITHM_COMP_H

#include "gpu/__config"
#include "gpu/__type_traits/predicate_traits.h"

namespace gpu {

struct __equal_to {
  template <class _T1, class _T2>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 bool operator()(const _T1& __x, const _T2& __y) const {
    return __x == __y;
  }
};

template <class _Lhs, class _Rhs>
struct __is_trivial_equality_predicate<__equal_to, _Lhs, _Rhs> : std::true_type {};

template <class _T1, class _T2 = _T1>
struct __less
{
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T2& __y) const {return __x < __y;}

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T2& __x, const _T1& __y) const {return __x < __y;}

    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T2& __x, const _T2& __y) const {return __x < __y;}
};

template <class _T1>
struct __less<_T1, _T1>
{
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}
};

template <class _T1>
struct __less<const _T1, _T1>
{
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}
};

template <class _T1>
struct __less<_T1, const _T1>
{
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}
};

} // namespace gpu

#endif // __GPU___ALGORITHM_COMP_H
