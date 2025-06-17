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

#ifndef __GPU___ITERATOR_SIZE_H
#define __GPU___ITERATOR_SIZE_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _Cont>
__device__ _LIBGPU_INLINE_VISIBILITY
constexpr auto size(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.size()))
-> decltype        (__c.size())
{ return            __c.size(); }

template <class _Tp, std::size_t _Sz>
__device__ _LIBGPU_INLINE_VISIBILITY
constexpr std::size_t size(const _Tp (&)[_Sz]) noexcept { return _Sz; }

#if _LIBGPU_STD_VER >= 20
template <class _Cont>
__device__ _LIBGPU_INLINE_VISIBILITY
constexpr auto ssize(const _Cont& __c)
_NOEXCEPT_(noexcept(static_cast<std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(__c.size())>>>(__c.size())))
->                              std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(__c.size())>>
{ return            static_cast<std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(__c.size())>>>(__c.size()); }

// GCC complains about the implicit conversion from std::ptrdiff_t to std::size_t in
// the array bound.
_LIBGPU_DIAGNOSTIC_PUSH
_LIBGPU_GCC_DIAGNOSTIC_IGNORED("-Wsign-conversion")
template <class _Tp, std::ptrdiff_t _Sz>
__device__ _LIBGPU_INLINE_VISIBILITY
constexpr std::ptrdiff_t ssize(const _Tp (&)[_Sz]) noexcept { return _Sz; }
_LIBGPU_DIAGNOSTIC_POP
#endif

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___ITERATOR_SIZE_H
