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

#include <thrust/detail/config/preprocessor.h>

// This is a utility file that helps managing which
// 'std' implementation we're using. The provided
// macros are for internal use only and may change
// in future versions.
//
// Example usage:
//     #include _THRUST_STD_INCLUDE(optional)
//     using optional_int = _THRUST_STD::optional<int>;

// Version that we depend on. We can ignore patch for now
// since we're only interested in breaking (major) and
// features (minor).
#define _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR 2
#define _THRUST_REQUIRED_LIBCXX_VERSION_MINOR 8

// Set default values if not explictly set
#ifndef USE_LIBCUDACXX
    #define USE_LIBCUDACXX 1
#endif
#ifndef USE_LIBHIPCXX
    #define USE_LIBHIPCXX 1
#endif

// If the '::cuda::std' namespace from 'libcudacxx' is available.
// Not for HIP, where this may be NVIDIA's libcu++. libhipcxx is found via <hip/std/version>.
#if !defined(__HIP__) && THRUST_HAS_INCLUDE(<cuda/std/version>) \
	&& USE_LIBCUDACXX
#  include <cuda/std/version>
// If version matches and '_CUDA_VSTD' is available.
#  if _LIBCUDACXX_CUDA_API_VERSION_MAJOR == _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR \
    && _LIBCUDACXX_CUDA_API_VERSION_MINOR >= _THRUST_REQUIRED_LIBCXX_VERSION_MINOR \
    && defined (_CUDA_VSTD)
#    define _THRUST_LIBCXX_INCLUDE(LIB)   <cuda/LIB>
#    define _THRUST_STD_INCLUDE(LIB)      <cuda/std/LIB>
#    define _THRUST_LIBCXX                ::cuda
#    define _THRUST_STD                   _CUDA_VSTD
#    define _THRUST_HAS_DEVICE_SYSTEM_STD 1
#    define _THRUST_STD_NAMESPACE_BEGIN   _LIBCUDACXX_BEGIN_NAMESPACE_STD
#    define _THRUST_STD_NAMESPACE_END     _LIBCUDACXX_END_NAMESPACE_STD
#    define _THRUST_USE_ROCPRIM           0
#  endif

// Otherwise, if the '::hip::std' namespace from 'libhipcxx' is available.
#elif THRUST_HAS_INCLUDE(<hip/std/version>) && USE_LIBHIPCXX
#  include <hip/std/version>
// If version matches and '_CUDA_VSTD' is available.
#  if _LIBCUDACXX_CUDA_API_VERSION_MAJOR == _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR \
    && _LIBCUDACXX_CUDA_API_VERSION_MINOR >= _THRUST_REQUIRED_LIBCXX_VERSION_MINOR \
    && defined(_CUDA_VSTD)
#    define _THRUST_LIBCXX_INCLUDE(LIB)   <hip/LIB>
#    define _THRUST_STD_INCLUDE(LIB)      <hip/std/LIB>
// In 'libhipcxx' the '::hip' namespace is synonymous with '::cuda'. 
#    define _THRUST_LIBCXX                ::hip
// In 'libhipcxx' the macro '_CUDA_VSTD' is also defined.
#    define _THRUST_STD                   _CUDA_VSTD
#    define _THRUST_HAS_DEVICE_SYSTEM_STD 1
#    define _THRUST_STD_NAMESPACE_BEGIN   _LIBCUDACXX_BEGIN_NAMESPACE_STD
#    define _THRUST_STD_NAMESPACE_END     _LIBCUDACXX_END_NAMESPACE_STD
#    define _THRUST_USE_ROCPRIM           0
#  endif
#endif

// If 'libcudacxx' or 'libhipcxx' is not found, or the
// feature flag ROCTHRUST_USE_LIBHIPCXX is not ON, use fallback.
#ifndef _THRUST_HAS_DEVICE_SYSTEM_STD 
#  define _THRUST_LIBCXX_INCLUDE(LIB)
#  define _THRUST_STD_INCLUDE(LIB) <LIB>
#  define _THRUST_LIBCXX
#  define _THRUST_STD                   ::std
#  define _THRUST_HAS_DEVICE_SYSTEM_STD 0
#  define _THRUST_STD_NAMESPACE_BEGIN \
    namespace std                     \
    {
#  define _THRUST_STD_NAMESPACE_END }
#  define _THRUST_USE_ROCPRIM (THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CPP)
#endif

// NVIDIA's <nv/target> shares our HIP include guard but always selects host code.
#if defined(__HIP__) && defined(_NV_COMPILER_NVCC) && !defined(__NVCC__)
#  error "<nv/target> was included in a HIP compilation, which is not HIP-aware and breaks rocThrust's host/device dispatch."
#endif
