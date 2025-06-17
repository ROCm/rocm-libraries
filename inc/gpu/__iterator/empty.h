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

#ifndef __GPU___ITERATOR_EMPTY_H
#define __GPU___ITERATOR_EMPTY_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _Cont>
_LIBGPU_NODISCARD_AFTER_CXX17 __device__ _LIBGPU_INLINE_VISIBILITY
constexpr auto empty(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.empty()))
-> decltype        (__c.empty())
{ return            __c.empty(); }

template <class _Tp, std::size_t _Sz>
_LIBGPU_NODISCARD_AFTER_CXX17 __device__ _LIBGPU_INLINE_VISIBILITY
constexpr bool empty(const _Tp (&)[_Sz]) noexcept { return false; }

template <class _Ep>
_LIBGPU_NODISCARD_AFTER_CXX17 __device__ _LIBGPU_INLINE_VISIBILITY
constexpr bool empty(std::initializer_list<_Ep> __il) noexcept { return __il.size() == 0; }

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___ITERATOR_EMPTY_H
