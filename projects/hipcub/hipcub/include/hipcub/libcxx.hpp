// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
//     #include _HIPCUB_STD_INCLUDE(optional)
//     using optional_int = _HIPCUB_STD::optional<int>;

// Version that we depend on. We can ignore patch for now
// since we're only interested in breaking (major) and
// features (minor).
#define _HIPCUB_REQUIRED_LIBCXX_VERSION_MAJOR 2
#define _HIPCUB_REQUIRED_LIBCXX_VERSION_MINOR 8

#ifdef __has_include
    #define HIPCUB_HAS_INCLUDE(_X) __has_include(_X)
#else
    #define HIPCUB_HAS_INCLUDE(_X) 0
#endif

#define _HIPCUB_STRINGIFY_IMPL(x) #x
#define _HIPCUB_STRINGIFY(x) _HIPCUB_STRINGIFY_IMPL(x)

// clang-format off

// If the '::cuda::std' namespace from 'libcudacxx' or 'libhipcxx' is available.
#if HIPCUB_HAS_INCLUDE(<cuda/std/version>)
    #include <cuda/std/version>
    // If version matches and '_CUDA_VSTD' is available.
    #if _LIBCUDACXX_CUDA_API_VERSION_MAJOR == _HIPCUB_REQUIRED_LIBCXX_VERSION_MAJOR    \
        && _LIBCUDACXX_CUDA_API_VERSION_MINOR >= _HIPCUB_REQUIRED_LIBCXX_VERSION_MINOR \
        && defined(_CUDA_VSTD)
        #define _HIPCUB_LIBCXX_INCLUDE(LIB) _HIPCUB_STRINGIFY(cuda/LIB)
        #define _HIPCUB_STD_INCLUDE(LIB) _HIPCUB_STRINGIFY(cuda/std/LIB)
        #define _HIPCUB_LIBCXX ::cuda
        #define _HIPCUB_STD _CUDA_VSTD
        #define _HIPCUB_HAS_DEVICE_SYSTEM_STD 1
        #define _HIPCUB_STD_NAMESPACE_BEGIN _LIBCUDACXX_BEGIN_NAMESPACE_STD
        #define _HIPCUB_STD_NAMESPACE_END _LIBCUDACXX_END_NAMESPACE_STD
    #endif

// Otherwise, if the '::hip::std' namespace from 'libhipcxx' is available.
#elif HIPCUB_HAS_INCLUDE(<hip/std/version>)
    #include <hip/std/version>
    // If version matches and '_CUDA_VSTD' is available.
    #if _LIBCUDACXX_CUDA_API_VERSION_MAJOR == _HIPCUB_REQUIRED_LIBCXX_VERSION_MINOR    \
        && _LIBCUDACXX_CUDA_API_VERSION_MINOR >= _HIPCUB_REQUIRED_LIBCXX_VERSION_MINOR \
        && defined(_CUDA_VSTD)
        #define _HIPCUB_LIBCXX_INCLUDE(LIB) _HIPCUB_STRINGIFY(hip/LIB)
        #define _HIPCUB_STD_INCLUDE(LIB) _HIPCUB_STRINGIFY(hip/std/LIB)
        // In 'libhipcxx' the '::hip' namespace is synonymous with '::cuda'.
        #define _HIPCUB_LIBCXX ::hip
        // In 'libhipcxx' the macro '_CUDA_VSTD' is also defined.
        #define _HIPCUB_STD _CUDA_VSTD
        #define _HIPCUB_HAS_DEVICE_SYSTEM_STD 1
        #define _HIPCUB_STD_NAMESPACE_BEGIN _LIBCUDACXX_BEGIN_NAMESPACE_STD
        #define _HIPCUB_STD_NAMESPACE_END _LIBCUDACXX_END_NAMESPACE_STD
    #endif
#endif

// If 'libcudacxx' or 'libhipcxx' is not found, use fallback.
#ifndef _HIPCUB_HAS_DEVICE_SYSTEM_STD
    #define _HIPCUB_LIBCXX_INCLUDE(LIB) _HIPCUB_STRINGIFY(LIB)
    #define _HIPCUB_STD_INCLUDE(LIB) _HIPCUB_STRINGIFY(LIB)
    #define _HIPCUB_LIBCXX
    #define _HIPCUB_STD ::std
    #define _HIPCUB_HAS_DEVICE_SYSTEM_STD 0
    #define _HIPCUB_STD_NAMESPACE_BEGIN \
        namespace std                   \
        {
    #define _HIPCUB_STD_NAMESPACE_END }
#endif

// clang-format on
