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

#ifndef __GPU___UTILITY_DECLVAL_H__
#define __GPU___UTILITY_DECLVAL_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::declval
//====================================================================================================================//

// Suppress deprecation notice for volatile-qualified return type resulting
// from volatile-qualified types _Tp.
_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _Tp>
__host__ __device__ _Tp&& __declval(int);
template <class _Tp>
__host__ __device__ _Tp __declval(long);
_LIBGPU_SUPPRESS_DEPRECATED_POP

template <class _Tp>
__host__ __device__ decltype(std::__declval<_Tp>(0)) declval() _NOEXCEPT;

} // namespace gpu

#endif // __GPU___UTILITY_DECLVAL_H__
