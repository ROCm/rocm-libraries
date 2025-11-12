/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file compiler.h
 *  \brief Compiler-specific configuration
 */

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// enumerate host compilers we know about
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_UNKNOWN 0
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_MSVC 1
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_GCC 2
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_CLANG 3
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_INTEL 4
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_NVHPC 5
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_NVRTC 6

// enumerate device compilers we know about
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_UNKNOWN 0
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_MSVC 1
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_GCC 2
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_CLANG 3
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_NVCC 4
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_HIP 5

// figure out which host compiler we're using
#if defined(_MSC_VER)
#  if defined(__clang__)
//! deprecated [Since 2.7]
#    define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_CLANG
//! deprecated [Since 2.7]
#    define THRUST_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#  else
//! deprecated [Since 2.7]
#    define THRUST_HOST_COMPILER     THRUST_HOST_COMPILER_MSVC
//! deprecated [Since 2.7]
#    define THRUST_MSVC_VERSION      _MSC_VER
//! deprecated [Since 2.7]
#    define THRUST_MSVC_VERSION_FULL _MSC_FULL_VER
#  endif
#elif defined(__INTEL_COMPILER)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_INTEL
#elif defined(__clang__)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_CLANG
//! deprecated [Since 2.7]
#  define THRUST_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_GCC
//! deprecated [Since 2.7]
#  define THRUST_GCC_VERSION   (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#  if (THRUST_GCC_VERSION >= 50000)
//! deprecated [Since 2.7]
#    define THRUST_MODERN_GCC
#  else
//! deprecated [Since 2.7]
#    define THRUST_LEGACY_GCC
#  endif
#elif defined(__NVCOMPILER)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_NVHPC
#elif defined(__CUDACC_RTC__)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_NVRTC
#else
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_UNKNOWN
#endif // THRUST_HOST_COMPILER

// figure out which device compiler we're using
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_MSVC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_GCC
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
// CUDA-capable clang should behave similar to NVCC.
#  if defined(__CUDA__)
//! deprecated [Since 2.7]
#    define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#  elif defined(__HIP__)
//! deprecated [Since 2.7]
#    define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_HIP
#  else
//! deprecated [Since 2.7]
#    define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_CLANG
#  endif
#else
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_UNKNOWN
#endif

// is the device compiler capable of compiling omp?
#if defined(_OPENMP) || defined(_NVHPC_STDPAR_OPENMP)
#  define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#else
#  define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // _OPENMP

// Convert parameter to string
#define THRUST_TO_STRING2(_STR) #_STR
#define THRUST_TO_STRING(_STR)  THRUST_TO_STRING2(_STR)

// Define the pragma for the host compiler
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_PRAGMA(x) __pragma(x)
#else
#  define THRUST_PRAGMA(x) _Pragma(THRUST_TO_STRING(x))
#endif // defined(_CCCL_COMPILER_MSVC)
