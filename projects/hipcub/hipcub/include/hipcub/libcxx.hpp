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

#ifndef HIPCUB_LIBCXX_HPP_
#define HIPCUB_LIBCXX_HPP_

#pragma once

// This is a utility file that helps managing which
// 'std' implementation we're using. The provided
// macros are for internal use only and may change
// in future versions.
//
// Example usage:
//     #include _HIPCUB_STD_INCLUDE(optional)
//     using optional_int = _HIPCUB_STD::optional<int>;

// Minimum version that we depend on.
#define _HIPCUB_REQUIRED_LIBCXX_VERSION_MAJOR 3
#define _HIPCUB_REQUIRED_LIBCXX_VERSION_MINOR 0
#define _HIPCUB_REQUIRED_LIBCXX_VERSION_PATCH 0

#define _HIPCUB_REQUIRED_LIBCXX_VERSION                                                            \
    _HIPCUB_REQUIRED_LIBCXX_VERSION_MAJOR * 1000000 + _HIPCUB_REQUIRED_LIBCXX_VERSION_MINOR * 1000 \
        + _HIPCUB_REQUIRED_LIBCXX_VERSION_PATCH

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
    #if defined(_LIBCUDACXX_CUDA_API_VERSION) && (_LIBCUDACXX_CUDA_API_VERSION >= _HIPCUB_REQUIRED_LIBCXX_VERSION) && defined(_CUDA_VSTD)
        #define _HIPCUB_LIBCXX_INCLUDE(LIB) _HIPCUB_STRINGIFY(cuda/LIB)
        #define _HIPCUB_STD_INCLUDE(LIB) _HIPCUB_STRINGIFY(cuda/std/LIB)
        #define _HIPCUB_LIBCXX ::cuda
        #define _HIPCUB_STD _CUDA_VSTD
        #define _HIPCUB_HAS_DEVICE_SYSTEM_STD 1
        #define _HIPCUB_STD_NAMESPACE_BEGIN _LIBCUDACXX_BEGIN_NAMESPACE_STD
        #define _HIPCUB_STD_NAMESPACE_END _LIBCUDACXX_END_NAMESPACE_STD
    #endif
#endif
// Otherwise, if the '::hip::std' namespace from 'libhipcxx' is available.
#if !defined(_HIPCUB_HAS_DEVICE_SYSTEM_STD) && HIPCUB_HAS_INCLUDE(<hip/std/version>)
    #include <hip/std/version>
    // If version matches and '_CUDA_VSTD' is available.
    #if defined(_LIBCUDACXX_CUDA_API_VERSION) && (_LIBCUDACXX_CUDA_API_VERSION >= _HIPCUB_REQUIRED_LIBCXX_VERSION) && defined(_CUDA_VSTD)
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

// If 'libcudacxx' or 'libhipcxx' is not found:
#ifndef _HIPCUB_HAS_DEVICE_SYSTEM_STD
    #if defined(__CUDACC__) || defined(_NVHPC_CUDA)
        #error "libcudacxx could not be found"
    #else
        // libhipcxx does not support Windows, use fallback
        #ifdef _WIN32
            #define _HIPCUB_LIBCXX_INCLUDE(LIB) _HIPCUB_STRINGIFY(LIB)
            #define _HIPCUB_STD_INCLUDE(LIB) _HIPCUB_STRINGIFY(LIB)
            #define _HIPCUB_LIBCXX
            #define _HIPCUB_STD ::std
            #define _HIPCUB_HAS_DEVICE_SYSTEM_STD 0
            #define _HIPCUB_STD_NAMESPACE_BEGIN \
                namespace std                   \
                {
            #define _HIPCUB_STD_NAMESPACE_END }
        #else
            #error "libhipcxx could not be found"
        #endif // _WIN32
    #endif  // defined(__CUDACC__) || defined(_NVHPC_CUDA)
#endif // _HIPCUB_HAS_DEVICE_SYSTEM_STD

#if _HIPCUB_HAS_DEVICE_SYSTEM_STD

#include _HIPCUB_STD_INCLUDE(__cccl/cuda_toolkit.h)
#define HIPCUB_CTK_AT_LEAST(...) _CCCL_CTK_AT_LEAST(__VA_ARGS__)

#else

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define HIPCUB_CUDA_COMPILATION() 1
#else 
#define HIPCUB_CUDA_COMPILATION() 0
#endif // defined(__CUDACC__) || defined(_NVHPC_CUDA)

#if HIPCUB_CUDA_COMPILATION() || HIPCUB_HAS_INCLUDE(<cuda_runtime_api.h>)
#define HIPCUB_HAS_CTK() 1
#else 
#define HIPCUB_HAS_CTK() 0
#endif // HIPCUB_CUDA_COMPILATION() || HIPCUB_HAS_INCLUDE(<cuda_runtime_api.h>)

// CUDA compilers preinclude cuda_runtime.h, so we need to include it here to get the CUDART_VERSION macro
#if HIPCUB_HAS_CTK() && !HIPCUB_CUDA_COMPILATION()
#include <cuda_runtime_api.h>
#endif // HIPCUB_HAS_CTK() && !HIPCUB_CUDA_COMPILATION()

// Check compatibility of the CUDA compiler and CUDA toolkit headers
#define HIPCUB_VERSION_MAJOR_INVALID -1
#define HIPCUB_MAKE_VERSION(_PREFIX, _PAIR) \
    (HIPCUB_PP_EVAL(HIPCUB_PP_CAT(_PREFIX, MAKE_VERSION), HIPCUB_PP_EXPAND _PAIR))
#define HIPCUB_VERSION_IS_INVALID(_PAIR) \
    (HIPCUB_VERSION_MAJOR == HIPCUB_VERSION_MAJOR_INVALID)
#define HIPCUB_VERSION_COMPARE_1(_PREFIX, _VER) (!HIPCUB_VERSION_IS_INVALID(_VER()))
#define HIPCUB_VERSION_COMPARE_3(_PREFIX, _VER, _OP, _MAJOR) \
    (!HIPCUB_VERSION_IS_INVALID(_VER()) && (HIPCUB_VERSION_MAJOR(_VER()) _OP _MAJOR))
#define HIPCUB_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, _MINOR) \
    (!HIPCUB_VERSION_IS_INVALID(_VER())                                \
     && (HIPCUB_MAKE_VERSION(_PREFIX, _VER()) _OP HIPCUB_MAKE_VERSION(_PREFIX, (_MAJOR, _MINOR))))
#define HIPCUB_VERSION_SELECT_COUNT(_ARG1, _ARG2, _ARG3, _ARG4, _ARG5, ...) _ARG5
#define HIPCUB_VERSION_SELECT2(_ARGS)                                       HIPCUB_VERSION_SELECT_COUNT _ARGS
#define HIPCUB_VERSION_SELECT(...)         \
    HIPCUB_VERSION_SELECT2(                  \
      (__VA_ARGS__,                          \
       HIPCUB_VERSION_COMPARE_4,             \
       HIPCUB_VERSION_COMPARE_3,             \
       HIPCUB_VERSION_COMPARE_BAD_ARG_COUNT, \
       HIPCUB_VERSION_COMPARE_1,             \
       HIPCUB_VERSION_COMPARE_BAD_ARG_COUNT))
#define HIPCUB_VERSION_COMPARE(_PREFIX, ...) HIPCUB_VERSION_SELECT(__VA_ARGS__)(_PREFIX, __VA_ARGS__)
#define HIPCUB_CUDACC_EQUAL(...) HIPCUB_VERSION_COMPARE(HIPCUB_CUDACC_, HIPCUB_CUDACC, ==, __VA_ARGS__)

#if HIPCUB_CUDA_COMPILATION()
#if !HIPCUB_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
#endif // !HIPCUB_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#endif // HIPCUB_CUDA_COMPILATION()

#define HIPCUB_PP_CAT(a, b)         _HIPCUB_PP_CAT_IMPL(a, b)
#define _HIPCUB_PP_CAT_IMPL(a, b)   a ## b
#define HIPCUB_PP_EVAL(expand, arg) expand(arg)
#define HIPCUB_PP_EXPAND(...)       __VA_ARGS__
#define HIPCUB_VERSION_COMPARE_BAD_ARG_COUNT(...) 0

#define HIPCUB_CTK_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)
#define HIPCUB_CTK()                            (CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10)
#define HIPCUB_CTK_AT_LEAST(...)                HIPCUB_HAS_CTK() && HIPCUB_VERSION_COMPARE(HIPCUB_CTK_, HIPCUB_CTK, >=, __VA_ARGS__)

#endif // _HIPCUB_HAS_DEVICE_SYSTEM_STD

// clang-format on

#endif // HIPCUB_LIBCXX_HPP_
