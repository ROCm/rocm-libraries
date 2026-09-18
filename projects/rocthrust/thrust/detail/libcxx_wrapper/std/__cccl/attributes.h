//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_ATTRIBUTES_H
#define LIBCXX_WRAPPER_STD__CCCL_ATTRIBUTES_H

// TODO(libhipcxx): remove this file and replace THRUST_DECLSPEC_EMPTY_BASES, THRUST_NODISCARD* and
// THRUST_ALIAS_ATTRIBUTE with _CCCL_DECLSPEC_EMPTY_BASES, _CCCL_NODISCARD* and _CCCL_ALIAS_ATTRIBUTE in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#include <thrust/detail/libcxx_wrapper/std/__cccl/compiler.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/attributes.h)

#  define THRUST_HAS_ATTRIBUTE(__x)     _CCCL_HAS_ATTRIBUTE(__x)
#  define THRUST_HAS_CPP_ATTRIBUTE(__x) _CCCL_HAS_CPP_ATTRIBUTE(__x)
#  define THRUST_DECLSPEC_EMPTY_BASES   _CCCL_DECLSPEC_EMPTY_BASES
#  define THRUST_NODISCARD              _CCCL_NODISCARD
#  define THRUST_NODISCARD_FRIEND       _CCCL_NODISCARD_FRIEND
#  define THRUST_NORETURN               _CCCL_NORETURN

#else // !_THRUST_HAS_DEVICE_SYSTEM_STD

#  ifdef __has_attribute
#    define THRUST_HAS_ATTRIBUTE(__x) __has_attribute(__x)
#  else // ^^^ __has_attribute ^^^ / vvv !__has_attribute vvv
#    define THRUST_HAS_ATTRIBUTE(__x) 0
#  endif // !__has_attribute

#  ifdef __has_cpp_attribute
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) __has_cpp_attribute(__x)
#  else // ^^^ __has_cpp_attribute ^^^ / vvv !__has_cpp_attribute vvv
#    define THRUST_HAS_CPP_ATTRIBUTE(__x) 0
#  endif // !__has_cpp_attribute

#  ifdef __has_declspec_attribute
#    define THRUST_HAS_DECLSPEC_ATTRIBUTE(__x) __has_declspec_attribute(__x)
#  else // ^^^ __has_declspec_attribute ^^^ / vvv !__has_declspec_attribute vvv
#    define THRUST_HAS_DECLSPEC_ATTRIBUTE(__x) 0
#  endif // !__has_declspec_attribute

#  if THRUST_COMPILER(MSVC) || THRUST_HAS_DECLSPEC_ATTRIBUTE(empty_bases)
#    define THRUST_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#  else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv !THRUST_COMPILER(MSVC) vvv
#    define THRUST_DECLSPEC_EMPTY_BASES
#  endif // !THRUST_COMPILER(MSVC)

#  if THRUST_HAS_CPP_ATTRIBUTE(nodiscard) || THRUST_COMPILER(MSVC)
#    define THRUST_NODISCARD [[nodiscard]]
#  else // ^^^ has nodiscard ^^^ / vvv no nodiscard vvv
#    define THRUST_NODISCARD
#  endif // no nodiscard

#  if THRUST_COMPILER(CLANG)
#    define THRUST_NODISCARD_FRIEND friend
#  else
#    define THRUST_NODISCARD_FRIEND THRUST_NODISCARD friend
#  endif

#  if THRUST_COMPILER(MSVC)
#    define THRUST_NORETURN __declspec(noreturn)
#  elif THRUST_HAS_CPP_ATTRIBUTE(noreturn)
#    define THRUST_NORETURN [[noreturn]]
#  else
#    define THRUST_NORETURN __attribute__((noreturn))
#  endif

#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

#endif // LIBCXX_WRAPPER_STD__CCCL_ATTRIBUTES_H
