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

#include <thrust/detail/config/libcxx.h> // TODO(libhipcxx): remove this once not needed

// is the device compiler capable of compiling omp?
#if defined(_OPENMP) || defined(_NVHPC_STDPAR_OPENMP)
#  define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#else
#  define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // _OPENMP

// TODO(libhipcxx): check if libhipcxx supports THRUST_COMPILER_HIP, remove the below code,
// and replace all THRUST* with _CCCL* in rocThrust once libhipcxx gets ready
#if _THRUST_HAS_DEVICE_SYSTEM_STD

// clang-format off
#  include _THRUST_STD_INCLUDE(__cccl/compiler.h)
// clang-format on

#  define THRUST_COMPILER _CCCL_COMPILER

#  if _CCCL_COMPILER(NVHPC)
#    define THRUST_COMPILER_NVHPC _CCCL_COMPILER_NVHPC
#  elif defined(__HIP__)
#    define THRUST_COMPILER_HIP _CCCL_COMPILER_MAKE_VERSION(__clang_major__, __clang_minor__)
#  elif _CCCL_COMPILER(CLANG)
#    define THRUST_COMPILER_CLANG _CCCL_COMPILER_CLANG
#  elif _CCCL_COMPILER(GCC)
#    define THRUST_COMPILER_GCC _CCCL_COMPILER_GCC
#  elif _CCCL_COMPILER(MSVC)
#    define THRUST_COMPILER_MSVC     _CCCL_COMPILER_MSVC
#    define THRUST_COMPILER_MSVC2017 _CCCL_COMPILER_MSVC2017
#    define THRUST_COMPILER_MSVC2019 _CCCL_COMPILER_MSVC2019
#    define THRUST_COMPILER_MSVC2022 _CCCL_COMPILER_MSVC2022
#  elif _CCCL_COMPILER(NVRTC)
#    define THRUST_COMPILER_NVRTC _CCCL_COMPILER_NVRTC
#  endif

#  define THRUST_CUDA_COMPILER _CCCL_CUDA_COMPILER

#  if _CCCL_CUDA_COMPILER(NVCC)
#    define THRUST_CUDA_COMPILER_NVCC _CCCL_CUDA_COMPILER_NVCC
#  elif _CCCL_CUDA_COMPILER(NVHPC)
#    define THRUST_CUDA_COMPILER_NVHPC _CCCL_CUDA_COMPILER_NVHPC
#  elif _CCCL_CUDA_COMPILER(CLANG)
#    define THRUST_CUDA_COMPILER_CLANG _CCCL_CUDA_COMPILER_CLANG
#  elif _CCCL_CUDA_COMPILER(NVRTC)
#    define THRUST_CUDA_COMPILER_NVRTC _CCCL_CUDA_COMPILER_NVRTC
#  endif

#  if defined(_CCCL_HAS_CUDA_COMPILER)
#    define THRUST_HAS_CUDA_COMPILER _CCCL_HAS_CUDA_COMPILER
#  endif

#  define THRUST_TO_STRING _CCCL_TO_STRING

#  define THRUST_PRAGMA _CCCL_PRAGMA

#else // TODO(libhipcxx): remove this path once libhipcxx gets ready

// Macros to suppress deprecation compiler warnings, from "deprecated.h"
// Check for deprecation opt-outs
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_DIALECT) || defined(CCCL_IGNORE_DEPRECATED_CPP_DIALECT) \
    || defined(CUB_IGNORE_DEPRECATED_CPP_DIALECT)
#    if !defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#      define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#    endif
#  endif // suppress all dialect deprecation warnings
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_14) || defined(CCCL_IGNORE_DEPRECATED_CPP_14) \
    || defined(CUB_IGNORE_DEPRECATED_CPP_14) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#    if !defined(THRUST_IGNORE_DEPRECATED_CPP_14)
#      define THRUST_IGNORE_DEPRECATED_CPP_14
#    endif
#  endif // suppress all c++14 dialect deprecation warnings
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_11) || defined(CCCL_IGNORE_DEPRECATED_CPP_11)  \
    || defined(CUB_IGNORE_DEPRECATED_CPP_11) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT) \
    || defined(THRUST_IGNORE_DEPRECATED_CPP_14)
#    if !defined(THRUST_IGNORE_DEPRECATED_CPP_11)
#      define THRUST_IGNORE_DEPRECATED_CPP_11
#    endif
#  endif // suppress all c++11 dialect deprecation warnings
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_COMPILER) || defined(CCCL_IGNORE_DEPRECATED_COMPILER) \
    || defined(CUB_IGNORE_DEPRECATED_COMPILER) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)  \
    || defined(THRUST_IGNORE_DEPRECATED_CPP_14) || defined(THRUST_IGNORE_DEPRECATED_CPP_11)
#    if !defined(THRUST_IGNORE_DEPRECATED_COMPILER)
#      define THRUST_IGNORE_DEPRECATED_COMPILER
#    endif
#  endif // suppress all compiler deprecation warnings

// Utility to compare version numbers. To use:
// 1) Define a macro that makes a version number from major and minor numbers, e. g.:
//    #define MYPRODUCT_MAKE_VERSION(_MAJOR, _MINOR) (_MAJOR * 100 + _MINOR)
// 2) Define a macro that you will use to compare versions, e. g.:
//    #define MYPRODUCT(...) THRUST_VERSION_COMPARE(MYPRODUCT, MYPRODUCT_##__VA_ARGS__)
//    Signatures:
//       MYPRODUCT(_PROD)                      - is the product _PROD version non-zero?
//       MYPRODUCT(_PROD, _OP, _MAJOR)         - compare the product _PROD version to _MAJOR using operator _OP
//       MYPRODUCT(_PROD, _OP, _MAJOR, _MINOR) - compare the product _PROD version to _MAJOR._MINOR using operator _OP
#  define THRUST_VERSION_COMPARE_1(_PREFIX, _VER)              (_VER != 0)
#  define THRUST_VERSION_COMPARE_3(_PREFIX, _VER, _OP, _MAJOR) THRUST_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, 0)
#  define THRUST_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, _MINOR) \
    (THRUST_VERSION_COMPARE_1(_PREFIX, _VER) && (_VER _OP _PREFIX##MAKE_VERSION(_MAJOR, _MINOR)))
#  define THRUST_VERSION_SELECT_COUNT(_ARG1, _ARG2, _ARG3, _ARG4, _ARG5, ...) _ARG5
#  define THRUST_VERSION_SELECT2(_ARGS)                                       THRUST_VERSION_SELECT_COUNT _ARGS
// MSVC traditonal preprocessor requires an extra level of indirection
#  define THRUST_VERSION_SELECT(...)         \
    THRUST_VERSION_SELECT2(                  \
      (__VA_ARGS__,                          \
       THRUST_VERSION_COMPARE_4,             \
       THRUST_VERSION_COMPARE_3,             \
       THRUST_VERSION_COMPARE_BAD_ARG_COUNT, \
       THRUST_VERSION_COMPARE_1,             \
       THRUST_VERSION_COMPARE_BAD_ARG_COUNT))
#  define THRUST_VERSION_COMPARE(_PREFIX, ...) THRUST_VERSION_SELECT(__VA_ARGS__)(_PREFIX, __VA_ARGS__)

#  define THRUST_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 100 + (_MINOR))
#  define THRUST_COMPILER(...)                         THRUST_VERSION_COMPARE(THRUST_COMPILER_, THRUST_COMPILER_##__VA_ARGS__)

// Determine the host compiler and its version
#  if defined(__INTEL_COMPILER)
#    ifndef THRUST_IGNORE_DEPRECATED_COMPILER
#      warning \
        "The Intel C++ Compiler Classic (icc/icpc) is not supported by CCCL. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this message."
#    endif // !THRUST_IGNORE_DEPRECATED_COMPILER
#  elif defined(__NVCOMPILER)
#    define THRUST_COMPILER_NVHPC THRUST_COMPILER_MAKE_VERSION(__NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__)
#  elif defined(__HIP__)
#    define THRUST_COMPILER_HIP THRUST_COMPILER_MAKE_VERSION(__clang_major__, __clang_minor__)
#  elif defined(__clang__)
#    define THRUST_COMPILER_CLANG THRUST_COMPILER_MAKE_VERSION(__clang_major__, __clang_minor__)
#  elif defined(__GNUC__)
#    define THRUST_COMPILER_GCC THRUST_COMPILER_MAKE_VERSION(__GNUC__, __GNUC_MINOR__)
#  elif defined(_MSC_VER)
#    define THRUST_COMPILER_MSVC     THRUST_COMPILER_MAKE_VERSION(_MSC_VER / 100, _MSC_VER % 100)
#    define THRUST_COMPILER_MSVC2017 (THRUST_COMPILER_MSVC < THRUST_COMPILER_MAKE_VERSION(19, 20))
#    if THRUST_COMPILER_MSVC2017 && !defined(THRUST_SUPPRESS_MSVC2017_DEPRECATION_WARNING)
#      pragma message( \
        "Support for the Visual Studio 2017 (MSC_VER < 1920) is deprecated and will eventually be removed. Define CCCL_SUPPRESS_MSVC2017_DEPRECATION_WARNING to suppress this warning")
#    endif
#    define THRUST_COMPILER_MSVC2019                                \
      (THRUST_COMPILER_MSVC >= THRUST_COMPILER_MAKE_VERSION(19, 20) \
       && THRUST_COMPILER_MSVC < THRUST_COMPILER_MAKE_VERSION(19, 30))
#    define THRUST_COMPILER_MSVC2022                                \
      (THRUST_COMPILER_MSVC >= THRUST_COMPILER_MAKE_VERSION(19, 30) \
       && THRUST_COMPILER_MSVC < THRUST_COMPILER_MAKE_VERSION(19, 40))
#  elif defined(__CUDACC_RTC__)
#    define THRUST_COMPILER_NVRTC THRUST_COMPILER_MAKE_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#  endif

// The CUDA compiler version shares the implementation with the C++ compiler
#  define THRUST_CUDA_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) THRUST_COMPILER_MAKE_VERSION(_MAJOR, _MINOR)
#  define THRUST_CUDA_COMPILER(...)                         THRUST_VERSION_COMPARE(THRUST_CUDA_COMPILER_, THRUST_CUDA_COMPILER_##__VA_ARGS__)

// Determine the cuda compiler
#  if defined(__NVCC__)
#    define THRUST_CUDA_COMPILER_NVCC THRUST_CUDA_COMPILER_MAKE_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#  elif defined(_NVHPC_CUDA)
#    define THRUST_CUDA_COMPILER_NVHPC THRUST_COMPILER_NVHPC
#  elif defined(__CUDA__) && THRUST_COMPILER(CLANG)
#    define THRUST_CUDA_COMPILER_CLANG THRUST_COMPILER_CLANG
#  elif THRUST_COMPILER(NVRTC)
#    define THRUST_CUDA_COMPILER_NVRTC THRUST_COMPILER_NVRTC
#  endif

#  define THRUST_CUDACC_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)

// clang-cuda does not define __CUDACC_VER_MAJOR__ and friends. They are instead retrieved from the CUDA_VERSION macro
// defined in "cuda.h". clang-cuda automatically pre-includes "__clang_cuda_runtime_wrapper.h" which includes "cuda.h"
#  if THRUST_CUDA_COMPILER(NVCC) || THRUST_CUDA_COMPILER(NVHPC) || THRUST_CUDA_COMPILER(NVRTC)
#    define THRUST_CUDACC THRUST_CUDACC_MAKE_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#  elif THRUST_CUDA_COMPILER(CLANG)
#    define THRUST_CUDACC THRUST_CUDACC_MAKE_VERSION(CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10)
#  endif

#  define THRUST_CUDACC_BELOW(...) THRUST_VERSION_COMPARE(THRUST_CUDACC_, THRUST_CUDACC, <, __VA_ARGS__)

#  if defined(THRUST_CUDACC)
#    define THRUST_HAS_CUDA_COMPILER 1
#  endif

// Convert parameter to string
#  define THRUST_TO_STRING2(_STR) #_STR
#  define THRUST_TO_STRING(_STR)  THRUST_TO_STRING2(_STR)

// Define the pragma for the host compiler
#  if THRUST_COMPILER(MSVC)
#    define THRUST_PRAGMA(x) __pragma(x)
#  else
#    define THRUST_PRAGMA(x) _Pragma(THRUST_TO_STRING(x))
#  endif // THRUST_COMPILER(MSVC)

#endif
