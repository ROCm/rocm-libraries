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

#ifndef __GPU___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H
#define __GPU___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H

#include "gpu/__config"
#include <type_traits>

namespace gpu {

// A type is trivially relocatable if a move construct + destroy of the original object is equivalent to
// `memcpy(dst, src, sizeof(T))`.

#if __has_builtin(__is_trivially_relocatable)
template <class _Tp, class = void>
struct __libcpp_is_trivially_relocatable : std::integral_constant<bool, __is_trivially_relocatable(_Tp)> {};
#else
template <class _Tp, class = void>
struct __libcpp_is_trivially_relocatable : std::is_trivially_copyable<_Tp> {};
#endif

template <class _Tp>
struct __libcpp_is_trivially_relocatable<_Tp,
                                         std::enable_if_t<std::is_same<_Tp, typename _Tp::__trivially_relocatable>::value> >
    : std::true_type {};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H
