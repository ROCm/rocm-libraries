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

#ifndef __GPU___ALGORITHM_HALF_POSITIVE_H
#define __GPU___ALGORITHM_HALF_POSITIVE_H

#include "gpu/__config"

namespace gpu {

// Perform division by two quickly for positive integers (llvm.org/PR39129)

template <typename _Integral>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
typename std::enable_if
<
    std::is_integral<_Integral>::value,
    _Integral
>::type
__half_positive(_Integral __value)
{
    return static_cast<_Integral>(static_cast<std::make_unsigned_t<_Integral> >(__value) / 2);
}

template <typename _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
typename std::enable_if
<
    !std::is_integral<_Tp>::value,
    _Tp
>::type
__half_positive(_Tp __value)
{
    return __value / 2;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_HALF_POSITIVE_H
