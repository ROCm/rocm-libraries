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

#ifndef __GPU___UTILITY_SWAP_H__
#define __GPU___UTILITY_SWAP_H__

#include "gpu/__config"
#include <cstddef>
#include <type_traits>

namespace gpu {

#ifndef _LIBGPU_CXX03_LANG
template <class _Tp>
using __swap_result_t =
    typename std::enable_if<std::is_move_constructible<_Tp>::value && std::is_move_assignable<_Tp>::value>::type;
#else
template <class>
using __swap_result_t = void;
#endif

template <class _Tp>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY __swap_result_t<_Tp> _LIBGPU_CONSTEXPR_SINCE_CXX20 swap(_Tp &__x, _Tp &__y)
    _NOEXCEPT_(std::is_nothrow_move_constructible<_Tp>::value && std::is_nothrow_move_assignable<_Tp>::value) {
    _Tp __t(std::move(__x));
    __x = std::move(__y);
    __y = std::move(__t);
}

template <class _Tp, size_t _Np>
__host__ __device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
    typename std::enable_if<std::is_swappable<_Tp>::value>::type
    swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) _NOEXCEPT_(std::is_nothrow_swappable<_Tp>::value) {
    for (size_t __i = 0; __i != _Np; ++__i) {
        swap(__a[__i], __b[__i]);
    }
}

} // namespace gpu

#endif // __GPU___UTILITY_SWAP_H__
