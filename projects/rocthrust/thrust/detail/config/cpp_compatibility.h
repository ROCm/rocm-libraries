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

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(__cccl_config)
#endif

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  define THRUST_HAS_CPP_ATTRIBUTE(__x) _CCCL_HAS_CPP_ATTRIBUTE(__x)
// deprecated [Since 2.8.0]
#  define THRUST_NODISCARD _CCCL_NODISCARD
// deprecated [Since 2.8.0]
#  define THRUST_INLINE_CONSTANT _CCCL_GLOBAL_CONSTANT
#else // TODO(libhipcxx): remove this path of code once libhipcxx gets ready

#  include <thrust/detail/config/compiler.h>
#  include <thrust/detail/config/cpp_dialect.h>
#  include <thrust/detail/config/execution_space.h>

#  ifdef __has_cpp_attribute
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#  else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) 0
#  endif // !__has_cpp_attribute

// deprecated [Since 2.8.0]
#  if THRUST_HAS_CPP_ATTRIBUTE(nodiscard) || (THRUST_COMPILER(MSVC) && THRUST_CPP_DIALECT >= 2017)
#    define THRUST_NODISCARD [[nodiscard]]
#  else // ^^^ has nodiscard ^^^ / vvv no nodiscard vvv
#    define THRUST_NODISCARD
#  endif // no nodiscard

#  if THRUST_CUDACC_BELOW(11, 3)
#    define THRUST_CONSTEXPR_GLOBAL const
#  else // ^^^ THRUST_CUDACC_BELOW(11, 3) ^^^ / vvv THRUST_CUDACC_BELOW(11, 3) vvv
#    define THRUST_CONSTEXPR_GLOBAL constexpr
#  endif // !THRUST_CUDACC_BELOW(11, 3)

// deprecated [Since 2.8.0]
#  if defined(__CUDA_ARCH__)
#    define THRUST_INLINE_CONSTANT THRUST_DEVICE THRUST_CONSTEXPR_GLOBAL
#  else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#    define THRUST_INLINE_CONSTANT inline constexpr
#  endif // __CUDA_ARCH__
#endif
// deprecated [Since 2.8.0]
#define THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT static constexpr
