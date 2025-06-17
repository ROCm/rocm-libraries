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

#ifndef __GPU___ALGORITHM_MINMAX_H
#define __GPU___ALGORITHM_MINMAX_H

#include "gpu/__config"

namespace gpu {

template<class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<const _Tp&, const _Tp&>
minmax(_LIBGPU_LIFETIMEBOUND const _Tp& __a, _LIBGPU_LIFETIMEBOUND const _Tp& __b, _Compare __comp)
{
    return __comp(__b, __a) ? gpu::pair<const _Tp&, const _Tp&>(__b, __a) :
                              gpu::pair<const _Tp&, const _Tp&>(__a, __b);
}

template<class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<const _Tp&, const _Tp&>
minmax(_LIBGPU_LIFETIMEBOUND const _Tp& __a, _LIBGPU_LIFETIMEBOUND const _Tp& __b)
{
    return gpu::minmax(__a, __b, __less<_Tp>());
}

#ifndef _LIBGPU_CXX03_LANG

template<class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<_Tp, _Tp> minmax(std::initializer_list<_Tp> __t, _Compare __comp) {
    static_assert(__is_callable<_Compare, _Tp, _Tp>::value, "The comparator has to be callable");
    __identity __proj;
    auto __ret = std::__minmax_element_impl(__t.begin(), __t.end(), __comp, __proj);
    return gpu::pair<_Tp, _Tp>(*__ret.first, *__ret.second);
}

template<class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<_Tp, _Tp>
minmax(std::initializer_list<_Tp> __t)
{
    return gpu::minmax(__t, __less<_Tp>());
}

#endif // _LIBGPU_CXX03_LANG

} // namespace gpu

#endif // __GPU___ALGORITHM_MINMAX_H
