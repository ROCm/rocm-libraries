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

#include <thrust/detail/config/cpp_dialect.h> // IWYU pragma: export
#include <thrust/detail/config/device_system.h> // IWYU pragma: export
#include <thrust/detail/config/execution_space.h> // IWYU pragma: export
#include <thrust/detail/config/libcxx.h> // IWYU pragma: export

#include _THRUST_STD_INCLUDE(cstddef)

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  define THRUST_HAS_CPP_ATTRIBUTE(__x) _CCCL_HAS_CPP_ATTRIBUTE(__x)
#  define THRUST_NODISCARD              _CCCL_NODISCARD

#else // TODO(libhipcxx): remove the code in this path and replace THRUST_NODISCARD with _CCCL_NODISCARD in rocThrust
      // once libhipcxx gets ready

#  ifdef __has_cpp_attribute
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#  else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) 0
#  endif // !__has_cpp_attribute

#  if THRUST_HAS_CPP_ATTRIBUTE(nodiscard) || (THRUST_COMPILER(MSVC) && THRUST_CPP_DIALECT >= 2017)
#    define THRUST_NODISCARD [[nodiscard]]
#  else // ^^^ has nodiscard ^^^ / vvv no nodiscard vvv
#    define THRUST_NODISCARD
#  endif // no nodiscard
#endif

// FIXME: Combine THRUST_INLINE_CONSTANT and
// THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT into one macro when NVCC properly
// supports `constexpr` globals in host and device code.
#if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
// FIXME: Add this when NVCC supports inline variables.
// #  if   THRUST_CPP_DIALECT >= 2017
// #    define THRUST_INLINE_CONSTANT                 inline constexpr
// #    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT inline constexpr
#  define THRUST_INLINE_CONSTANT                 static const THRUST_DEVICE
#  define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr

#else
// FIXME: Add this when NVCC supports inline variables.
// #  if   THRUST_CPP_DIALECT >= 2017
// #    define THRUST_INLINE_CONSTANT                 inline constexpr
// #    define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT inline constexpr
#  define THRUST_INLINE_CONSTANT                 static constexpr
#  define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr

#endif
