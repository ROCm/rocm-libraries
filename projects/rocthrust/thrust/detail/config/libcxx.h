// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// This is a utility file that helps managing which
// 'std' implementation we're using. The provided
// macros are for internal use only and may change
// in future versions.
//
// Example usage:
//     #include _THRUST_STD_INCLUDE(optional)
//     using optional_int = _THRUST_STD::optional<int>;

// When targeting CUDA, we want to use 'libcudacxx'. This
// should always be available, since we are also dependent
// on CCCL/thrust, which is a sibling of CCCL/libcudacxx.
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  define _THRUST_STD_INCLUDE(LIB)      <cuda/std/LIB>
#  define _THRUST_STD                   _CUDA_VSTD
#  define _THRUST_HAS_DEVICE_SYSTEM_STD 1
#  define _THRUST_STD_NAMESPACE_BEGIN   _LIBCUDACXX_BEGIN_NAMESPACE_STD
#  define _THRUST_STD_NAMESPACE_END     _LIBCUDACXX_END_NAMESPACE_STD

// When targeting HIP, we want to use 'libhipcxx' if
// available. We can do this by checking the existance
// of a known include.
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#  if defined(__has_include) && __has_include(<hip/std/cassert>)
#    define _THRUST_STD_INCLUDE(LIB)      <hip/std/LIB>
#    define _THRUST_STD                   ::hip::std
#    define _THRUST_HAS_DEVICE_SYSTEM_STD 1
#    define _THRUST_STD_NAMESPACE_BEGIN   _LIBCUDACXX_BEGIN_NAMESPACE_STD
#    define _THRUST_STD_NAMESPACE_END     _LIBCUDACXX_END_NAMESPACE_STD
#  endif
#endif

// If not using GPU backend or GPU vendor's STD is not
// found, use fallback.
#ifndef _THRUST_HAS_DEVICE_SYSTEM_STD
#  define _THRUST_STD_INCLUDE(LIB) <LIB>
#  define _THRUST_STD              ::std
#  define _THRUST_STD_NAMESPACE_BEGIN \
    namespace std                     \
    {
#  define _THRUST_STD_NAMESPACE_END }
#endif
