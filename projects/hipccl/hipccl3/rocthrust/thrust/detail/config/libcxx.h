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

#include <thrust/detail/libcxx_wrapper/std/__cccl/preprocessor.h>

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
#define _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR 3
#define _THRUST_REQUIRED_LIBCXX_VERSION_MINOR 4

// Set default values if not explictly set
#ifndef USE_LIBCUDACXX
    #define USE_LIBCUDACXX 1
#endif
#ifndef USE_LIBHIPCXX
    #define USE_LIBHIPCXX 1
#endif

// If the '::cuda::std' namespace from 'libcudacxx' or 'libhipcxx' is available.
#if THRUST_HAS_INCLUDE(<cuda/std/version>) && (USE_LIBCUDACXX || USE_LIBHIPCXX)
#  include <cuda/std/version>
// If version matches and '_CUDA_VSTD' is available.
#  if _LIBCUDACXX_CUDA_API_VERSION_MAJOR == _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR \
    && _LIBCUDACXX_CUDA_API_VERSION_MINOR >= _THRUST_REQUIRED_LIBCXX_VERSION_MINOR
#    define _THRUST_LIBCXX_INCLUDE(LIB)    <cuda/LIB>
#    define _THRUST_STD_INCLUDE(LIB)       <cuda/std/LIB>
#    define _THRUST_LIBCXX                 ::cuda
#    define _THRUST_STD_NOVERSION          ::cuda::std
#    define _THRUST_STD                    ::cuda::std
#    define _THRUST_HAS_DEVICE_SYSTEM_STD  1
#    define _THRUST_STD_NAMESPACE_BEGIN    _THRUST_LIBCXX_BEGIN_NAMESPACE_CUDA_STD
#    define _THRUST_STD_NAMESPACE_END      _THRUST_LIBCXX_END_NAMESPACE_CUDA_STD
#    define _THRUST_USE_ROCPRIM            0
#  endif

// Otherwise, if the '::hip::std' namespace from 'libhipcxx' is available.
#elif THRUST_HAS_INCLUDE(<hip/std/version>) && USE_LIBHIPCXX
#  include <hip/std/version>
// If version matches and '_CUDA_VSTD' is available.
#  if _LIBCUDACXX_CUDA_API_VERSION_MAJOR == _THRUST_REQUIRED_LIBCXX_VERSION_MAJOR \
    && _LIBCUDACXX_CUDA_API_VERSION_MINOR >= _THRUST_REQUIRED_LIBCXX_VERSION_MINOR && defined(_CUDA_VSTD)
#    define _THRUST_LIBCXX_INCLUDE(LIB)   <hip/LIB>
#    define _THRUST_STD_INCLUDE(LIB)      <hip/std/LIB>
// In 'libhipcxx' the '::hip' namespace is synonymous with '::cuda'.
#    define _THRUST_LIBCXX                ::hip
// In 'libhipcxx' the macro '_CUDA_VSTD' is also defined.
#    define _THRUST_STD_NOVERSION         ::hip::std
#    define _THRUST_STD                   ::hip::std
#    define _THRUST_HAS_DEVICE_SYSTEM_STD 1
#    define _THRUST_STD_NAMESPACE_BEGIN   _THRUST_LIBCXX_BEGIN_NAMESPACE_CUDA_STD
#    define _THRUST_STD_NAMESPACE_END     _THRUST_LIBCXX_END_NAMESPACE_CUDA_STD
#    define _THRUST_USE_ROCPRIM           0
#  endif
#endif

// CCCL 3.2 renamed the libcu++ namespace macros
#if defined(_THRUST_HAS_DEVICE_SYSTEM_STD)
#  if defined(_CCCL_BEGIN_NAMESPACE_CUDA_STD)
#    define _THRUST_LIBCXX_BEGIN_NAMESPACE_CUDA_STD _CCCL_BEGIN_NAMESPACE_CUDA_STD
#    define _THRUST_LIBCXX_END_NAMESPACE_CUDA_STD   _CCCL_END_NAMESPACE_CUDA_STD
#    define _THRUST_LIBCXX_BEGIN_NAMESPACE_CUDA     _CCCL_BEGIN_NAMESPACE_CUDA
#    define _THRUST_LIBCXX_END_NAMESPACE_CUDA       _CCCL_END_NAMESPACE_CUDA
#  else
#    define _THRUST_LIBCXX_BEGIN_NAMESPACE_CUDA_STD _LIBCUDACXX_BEGIN_NAMESPACE_STD
#    define _THRUST_LIBCXX_END_NAMESPACE_CUDA_STD   _LIBCUDACXX_END_NAMESPACE_STD
#    define _THRUST_LIBCXX_BEGIN_NAMESPACE_CUDA     _LIBCUDACXX_BEGIN_NAMESPACE_CUDA
#    define _THRUST_LIBCXX_END_NAMESPACE_CUDA       _LIBCUDACXX_END_NAMESPACE_CUDA
#  endif
#else
#  error Could not find libhipcxx!
#endif

// In case libhipcxx is not available, load a basic version from our libcxx wrapper.
#if _THRUST_HAS_DEVICE_SYSTEM_STD
// libhipcxx exposes <nv/target>, but does not expose <hip/target>.
#  include <nv/target>
#else
#  include <thrust/detail/libcxx_wrapper/nv/detail/__target_macros.h>
#endif

// NV_* macros provided by 'libhipcxx' are suffixed with _LIBHIPCXX.
#if defined(NV_IF_TARGET)
#  define _THRUST_IF_TARGET NV_IF_TARGET
#elif defined(NV_IF_TARGET_LIBHIPCXX)
#  define _THRUST_IF_TARGET NV_IF_TARGET_LIBHIPCXX
#else
#  error Could not find definition for '_THRUST_IF_TARGET'!
#endif

#if defined(NV_IF_ELSE_TARGET)
#  define _THRUST_IF_ELSE_TARGET NV_IF_ELSE_TARGET
#elif defined(NV_IF_ELSE_TARGET_LIBHIPCXX)
#  define _THRUST_IF_ELSE_TARGET NV_IF_ELSE_TARGET_LIBHIPCXX
#else
#  error Could not find definition for '_THRUST_IF_ELSE_TARGET'!
#endif

#if defined(NV_IS_HOST)
#  define _THRUST_IS_HOST NV_IS_HOST
#elif defined(NV_IS_HOST_LIBHIPCXX)
#  define _THRUST_IS_HOST NV_IS_HOST_LIBHIPCXX
#else
#  error Could not find definition for '_THRUST_IS_HOST'!
#endif

#if defined(NV_IS_DEVICE)
#  define _THRUST_IS_DEVICE NV_IS_DEVICE
#elif defined(NV_IS_DEVICE_LIBHIPCXX)
#  define _THRUST_IS_DEVICE NV_IS_DEVICE_LIBHIPCXX
#else
#  error Could not find definition for '_THRUST_IS_DEVICE'!
#endif

#if defined(NV_ANY_TARGET)
#  define _THRUST_ANY_TARGET NV_ANY_TARGET
#elif defined(NV_ANY_TARGET_LIBHIPCXX)
#  define _THRUST_ANY_TARGET NV_ANY_TARGET_LIBHIPCXX
#else
#  error Could not find definition for '_THRUST_ANY_TARGET'!
#endif
