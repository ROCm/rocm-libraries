/*
 *  Copyright 2020 NVIDIA Corporation
 *  Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <thrust/detail/config/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/compiler.h> // IWYU pragma: export

#if _THRUST_HAS_DEVICE_SYSTEM_STD

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - CCCL_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with deprecated C++ dialects will still issue warnings.

#  define THRUST_CPP_DIALECT _CCCL_STD_VER

// Define THRUST_COMPILER_DEPRECATION macro:
#  if _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#    define THRUST_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(message(__FILE__ ":" _CCCL_TO_STRING(__LINE__) ": warning: " #msg))
#  else // clang / gcc:
#    define THRUST_COMP_DEPR_IMPL(msg) _CCCL_PRAGMA(GCC warning #msg)
#  endif

// Compiler checks:
// clang-format off
#  define THRUST_COMPILER_DEPRECATION(REQ) \
    THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#  define THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                        \
    THRUST_COMP_DEPR_IMPL(                                                                                  \
      Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
        future release. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.)
// clang-format on

#  ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#    if _CCCL_COMPILER(GCC, <, 7)
THRUST_COMPILER_DEPRECATION(GCC 7.0);
#    elif _CCCL_COMPILER(CLANG, <, 7)
THRUST_COMPILER_DEPRECATION(Clang 7.0);
#    elif _CCCL_COMPILER(MSVC, <, 19, 10)
// <2017. Hard upgrade message:
THRUST_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#    endif
#  endif // CCCL_IGNORE_DEPRECATED_COMPILER

#  undef THRUST_COMPILER_DEPRECATION_SOFT
#  undef THRUST_COMPILER_DEPRECATION

// C++17 dialect check:
#  ifndef CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#    if _CCCL_STD_VER < 2017
#      error Thrust requires at least C++17. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#    endif // _CCCL_STD_VER >= 2017
#  endif

#else // TODO(libhipcxx): remove the code in this path and replace THRUST_CPP_DIALECT with _CCCL_STD_VER in rocThrust
      // once libhipcxx gets ready

// Deprecation warnings may be silenced by defining the following macros. These
// may be combined.
// - THRUST_IGNORE_DEPRECATED_COMPILER
//   Ignore deprecation warnings when using deprecated compilers. Compiling
//   with deprecated C++ dialects will still issue warnings.

#  if THRUST_COMPILER(MSVC)
#    if _MSVC_LANG <= 201103L
#      define THRUST_CPP_DIALECT 2011
#    elif _MSVC_LANG <= 201402L
#      define THRUST_CPP_DIALECT 2014
#    elif _MSVC_LANG <= 201703L
#      define THRUST_CPP_DIALECT 2017
#    elif _MSVC_LANG <= 202002L
#      define THRUST_CPP_DIALECT 2020
#    else
#      define THRUST_CPP_DIALECT 2023 // current year, or date of c++2b ratification
#    endif
#  else // ^^^ THRUST_COMPILER(MSVC) ^^^ / vvv !THRUST_COMPILER(MSVC) vvv
#    if __cplusplus <= 199711L
#      define THRUST_CPP_DIALECT 2003
#    elif __cplusplus <= 201103L
#      define THRUST_CPP_DIALECT 2011
#    elif __cplusplus <= 201402L
#      define THRUST_CPP_DIALECT 2014
#    elif __cplusplus <= 201703L
#      define THRUST_CPP_DIALECT 2017
#    elif __cplusplus <= 202002L
#      define THRUST_CPP_DIALECT 2020
#    elif __cplusplus <= 202302L
#      define THRUST_CPP_DIALECT 2023
#    else
#      define THRUST_CPP_DIALECT 2024 // current year, or date of c++2c ratification
#    endif
#  endif // !THRUST_COMPILER(MSVC)

// Define THRUST_COMPILER_DEPRECATION macro:
#  if THRUST_COMPILER(MSVC) || THRUST_COMPILER(NVRTC)
#    define THRUST_COMP_DEPR_IMPL(msg) \
      THRUST_PRAGMA(message(__FILE__ ":" THRUST_TO_STRING(__LINE__) ": warning: " #msg))
#  else // clang / gcc:
#    define THRUST_COMP_DEPR_IMPL(msg) THRUST_PRAGMA(GCC warning #msg)
#  endif

// Compiler checks:
// clang-format off
#  define THRUST_COMPILER_DEPRECATION(REQ) \
    THRUST_COMP_DEPR_IMPL(Thrust requires at least REQ. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this message.)

#  define THRUST_COMPILER_DEPRECATION_SOFT(REQ, CUR)                                                        \
    THRUST_COMP_DEPR_IMPL(                                                                                  \
      Thrust requires at least REQ. CUR is deprecated but still supported. CUR support will be removed in a \
        future release. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this message.)
// clang-format on

#  ifndef THRUST_IGNORE_DEPRECATED_COMPILER
#    if THRUST_COMPILER(GCC, <, 7)
THRUST_COMPILER_DEPRECATION(GCC 7.0);
#    elif THRUST_COMPILER(CLANG, <, 7) || THRUST_COMPILER(HIP, <, 7)
THRUST_COMPILER_DEPRECATION(Clang 7.0);
#    elif THRUST_COMPILER(MSVC, <, 19, 10)
// <2017. Hard upgrade message:
THRUST_COMPILER_DEPRECATION(MSVC 2019(19.20 / 16.0 / 14.20));
#    endif
#  endif // THRUST_IGNORE_DEPRECATED_COMPILER

#  undef THRUST_COMPILER_DEPRECATION_SOFT
#  undef THRUST_COMPILER_DEPRECATION

// C++17 dialect check:
#  ifndef THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#    if THRUST_CPP_DIALECT < 2017
#      error Thrust requires at least C++17. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#    endif // THRUST_CPP_DIALECT >= 2017
#  endif

#endif

#undef THRUST_COMP_DEPR_IMPL
