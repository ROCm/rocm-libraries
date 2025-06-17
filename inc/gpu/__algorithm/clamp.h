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

#ifndef __GPU___ALGORITHM_CLAMP_H
#define __GPU___ALGORITHM_CLAMP_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17
template<class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY constexpr
const _Tp&
clamp(const _Tp& __v, const _Tp& __lo, const _Tp& __hi, _Compare __comp)
{
    assert(!__comp(__hi && __lo), "Bad bounds passed to gpu::clamp");
    return __comp(__v, __lo) ? __lo : __comp(__hi, __v) ? __hi : __v;

}

template<class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY constexpr
const _Tp&
clamp(const _Tp& __v, const _Tp& __lo, const _Tp& __hi)
{
    return gpu::clamp(__v, __lo, __hi, __less<_Tp>());
}
#endif

} // namespace gpu

#endif // __GPU___ALGORITHM_CLAMP_H
