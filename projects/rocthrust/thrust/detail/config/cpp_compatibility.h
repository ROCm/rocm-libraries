/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright© 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <thrust/detail/config/cpp_dialect.h> // IWYU pragma: export
#  include <thrust/detail/config/execution_space.h> // IWYU pragma: export
#endif

// deprecated [Since 2.8.0]
#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  define THRUST_NODISCARD _CCCL_NODISCARD
#else
#  ifdef __has_cpp_attribute
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#  else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) 0
#  endif // !__has_cpp_attribute
#  if THRUST_HAS_CPP_ATTRIBUTE(nodiscard) \
    || ((THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && THRUST_CPP_DIALECT >= 2017)
#    define THRUST_NODISCARD [[nodiscard]]
#  else // ^^^ has nodiscard ^^^ / vvv no nodiscard vvv
#    define THRUST_NODISCARD
#  endif // no nodiscard
#  undef THRUST_HAS_CPP_ATTRIBUTE
#endif
// deprecated [Since 2.8.0]
#if defined(__CUDA_ARCH__)
#  if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 3))
#    define THRUST_INLINE_CONSTANT THRUST_DEVICE const
#  else
#    define THRUST_INLINE_CONSTANT THRUST_DEVICE constexpr
#  endif
#elif defined(__HIP__)
#  define THRUST_INLINE_CONSTANT THRUST_DEVICE constexpr
#else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#  define THRUST_INLINE_CONSTANT inline constexpr
#endif // __CUDA_ARCH__
// deprecated [Since 2.8.0]
#define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
// libcu++ still needs to be ported to HIP, so for HIP backend these definitions
// are still in use.
#  if defined(__HIP_DEVICE_COMPILE__)
#    define THRUST_IS_DEVICE_CODE      1
#    define THRUST_IS_HOST_CODE        0
#    define THRUST_INCLUDE_DEVICE_CODE 1
#    define THRUST_INCLUDE_HOST_CODE   0
#  else
#    define THRUST_IS_DEVICE_CODE      0
#    define THRUST_IS_HOST_CODE        1
#    define THRUST_INCLUDE_DEVICE_CODE 0
#    define THRUST_INCLUDE_HOST_CODE   1
#  endif
#else
// These definitions were intended for internal use only and are now obsolete.
// If you relied on them, consider porting your code to use the functionality
// in libcu++'s <nv/target> header.
// For a temporary workaround, define THRUST_PROVIDE_LEGACY_ARCH_MACROS to make
// them available again. These should be considered deprecated and will be
// fully removed in a future version.
#  ifdef THRUST_PROVIDE_LEGACY_ARCH_MACROS
#    ifndef THRUST_IS_DEVICE_CODE
#      if defined(_NVHPC_CUDA)
#        define THRUST_IS_DEVICE_CODE      __builtin_is_device_code()
#        define THRUST_IS_HOST_CODE        (!__builtin_is_device_code())
#        define THRUST_INCLUDE_DEVICE_CODE 1
#        define THRUST_INCLUDE_HOST_CODE   1
#      elif defined(__CUDA_ARCH__)
#        define THRUST_IS_DEVICE_CODE      1
#        define THRUST_IS_HOST_CODE        0
#        define THRUST_INCLUDE_DEVICE_CODE 1
#        define THRUST_INCLUDE_HOST_CODE   0
#      else
#        define THRUST_IS_DEVICE_CODE      0
#        define THRUST_IS_HOST_CODE        1
#        define THRUST_INCLUDE_DEVICE_CODE 0
#        define THRUST_INCLUDE_HOST_CODE   1
#      endif
#    endif
#  endif // THRUST_PROVIDE_LEGACY_ARCH_MACROS
#endif

// NVCC below 11.3 does not support nodiscard on friend operators
// It always fails with clang
#if (defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 3))) \
  || THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
#  define THRUST_NODISCARD_FRIEND friend
#else
#  define THRUST_NODISCARD_FRIEND THRUST_NODISCARD friend
#endif

#if THRUST_CPP_DIALECT <= 2014 || (defined(__cpp_if_constexpr) && __cpp_if_constexpr < 201606L)
#  define THRUST_IF_CONSTEXPR if
#else // ^^^ C++14 ^^^ / vvv C++17 vvv
#  define THRUST_IF_CONSTEXPR if constexpr
#endif // THRUST_CPP_DIALECT > 2014
