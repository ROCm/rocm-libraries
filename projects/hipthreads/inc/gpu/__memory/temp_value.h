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

#ifndef __GPU___MEMORY_TEMP_VALUE_H__
#define __GPU___MEMORY_TEMP_VALUE_H__

#include "gpu/__config"

#include <type_traits>

#include "gpu/__memory/addressof.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __memory/temp_value.h
//====================================================================================================================//

// TODO: re-add allocator template parameter
template <class _Tp>
struct __temp_value {
#ifdef _LIBGPU_CXX03_LANG
    typename std::aligned_storage<sizeof(_Tp), _LIBGPU_ALIGNOF(_Tp)>::type __v;
#else
    union { _Tp __v; };
#endif

    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp *__addr() {
#ifdef _LIBGPU_CXX03_LANG
        return reinterpret_cast<_Tp*>(gpu::addressof(__v));
#else
        return gpu::addressof(__v);
#endif
    }

    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp &   get() { return *__addr(); }

    template<class... _Args>
    __device__ _LIBGPU_NO_CFI
    _LIBGPU_CONSTEXPR_SINCE_CXX20 __temp_value(_Args&& ... __args) {
        ::new (static_cast<void*>(__addr())) _Tp(std::forward<_Args>(__args)...);
    }

    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX20 ~__temp_value() { __addr()->~_Tp(); }
};


} // namespace gpu

#endif // __GPU___MEMORY_TEMP_VALUE_H__
