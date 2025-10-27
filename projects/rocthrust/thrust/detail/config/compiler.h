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

#  if defined(__HIP__)
#    define _CCCL_COMPILER_HIP() _CCCL_COMPILER_CLANG()
#    undef _CCCL_COMPILER_CLANG
#    define _CCCL_COMPILER_CLANG() _CCCL_VERSION_INVALID()
#  else
#    define _CCCL_COMPILER_HIP() _CCCL_VERSION_INVALID()
#  endif

#  define THRUST_CUDA_COMPILER       _CCCL_CUDA_COMPILER
#  define THRUST_HAS_CUDA_COMPILER() _CCCL_HAS_CUDA_COMPILER()
#  define THRUST_CUDA_COMPILATION()  _CCCL_CUDA_COMPILATION()

#  undef _CCCL_HOST_COMPILATION
#  if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#    define _CCCL_HOST_COMPILATION() 1
#  else // ^^^ compiling host code ^^^ / vvv not compiling host code vvv
#    define _CCCL_HOST_COMPILATION() 0
#  endif // ^^^ not compiling host code ^^^

#  undef _CCCL_DEVICE_COMPILATION
#  if (_CCCL_CUDA_COMPILATION() && defined(__CUDA_ARCH__)) || _CCCL_CUDA_COMPILER(NVHPC) \
    || (_CCCL_COMPILER(HIP) && defined(__HIP_DEVICE_COMPILE__))
#    define _CCCL_DEVICE_COMPILATION() 1
#  else // ^^^ compiling device code ^^^ / vvv not compiling device code vvv
#    define _CCCL_DEVICE_COMPILATION() 0
#  endif // ^^^ not compiling device code ^^^

#  define THRUST_PRAGMA _CCCL_PRAGMA

#  undef _CCCL_PRAGMA_UNROLL_FULL
#  if _CCCL_DEVICE_COMPILATION()
#    define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA(unroll)
#  elif _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(NVRTC) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(HIP)
#    define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA(unroll)
#  elif _CCCL_COMPILER(GCC, >=, 8)
#    define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA_UNROLL(65534)
#  else // ^^^ has pragma unroll support ^^^ / vvv no pragma unroll support vvv
#    define _CCCL_PRAGMA_UNROLL_FULL()
#  endif // ^^^ no pragma unroll support ^^^
#  define THRUST_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA_UNROLL_FULL()

#else // TODO(libhipcxx): remove this path once libhipcxx gets ready

#  include <thrust/detail/config/preprocessor.h>

// Macros to suppress deprecation compiler warnings, from "deprecated.h"
// Check for deprecation opt-outs
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_DIALECT)
#    if !defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#      define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#    endif
#  endif // suppress all dialect deprecation warnings
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_14) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
#    if !defined(THRUST_IGNORE_DEPRECATED_CPP_14)
#      define THRUST_IGNORE_DEPRECATED_CPP_14
#    endif
#  endif // suppress all c++14 dialect deprecation warnings
#  if defined(LIBCUDACXX_IGNORE_DEPRECATED_CPP_11) || defined(THRUST_IGNORE_DEPRECATED_CPP_DIALECT) \
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
// 1) Define a macro that makes a pair of (major, minor) numbers:
//    #define MYPRODUCT_MAKE_VERSION(_MAJOR, _MINOR) (_MAJOR * 100 + _MINOR)
// 2) Define a macro that you will use to compare versions, e.g.:
//    #define MYPRODUCT(...) THRUST_VERSION_COMPARE(MYPRODUCT, MYPRODUCT_##__VA_ARGS__)
//    Signatures:
//       MYPRODUCT(_PROD)                      - is the product _PROD version non-zero?
//       MYPRODUCT(_PROD, _OP, _MAJOR)         - compare the product _PROD major version to _MAJOR using operator _OP
//       MYPRODUCT(_PROD, _OP, _MAJOR, _MINOR) - compare the product _PROD version to _MAJOR._MINOR using operator _OP
// 3) Define the product version macros as a function-like macro that returns the version number or
//    THRUST_VERSION_INVALID() if the version cannot be determined, e. g.:
//    #define MYPRODUCT_<_PROD>() (1, 2)
//      or
//    #define MYPRODUCT_<_PROD>() THRUST_VERSION_INVALID()
#  define THRUST_VERSION_MAJOR_(_MAJOR, _MINOR) _MAJOR
#  define THRUST_VERSION_MAJOR(_PAIR)           THRUST_VERSION_MAJOR_ _PAIR
#  define THRUST_VERSION_INVALID()              (-1, -1)
#  define THRUST_MAKE_VERSION(_PREFIX, _PAIR) \
    (THRUST_PP_EVAL(THRUST_PP_CAT(_PREFIX, MAKE_VERSION), THRUST_PP_EXPAND _PAIR))
#  define THRUST_VERSION_IS_INVALID(_PAIR) \
    (THRUST_VERSION_MAJOR(_PAIR) == THRUST_VERSION_MAJOR(THRUST_VERSION_INVALID()))
#  define THRUST_VERSION_COMPARE_1(_PREFIX, _VER) (!THRUST_VERSION_IS_INVALID(_VER()))
#  define THRUST_VERSION_COMPARE_3(_PREFIX, _VER, _OP, _MAJOR) \
    (!THRUST_VERSION_IS_INVALID(_VER()) && (THRUST_VERSION_MAJOR(_VER()) _OP _MAJOR))
#  define THRUST_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, _MINOR) \
    (!THRUST_VERSION_IS_INVALID(_VER())                                \
     && (THRUST_MAKE_VERSION(_PREFIX, _VER()) _OP THRUST_MAKE_VERSION(_PREFIX, (_MAJOR, _MINOR))))
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

#  define THRUST_COMPILER_NVHPC()    THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_HIP()      THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_CLANG()    THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_GCC()      THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_MSVC()     THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_MSVC2019() THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_MSVC2022() THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_NVRTC()    THRUST_VERSION_INVALID()

// Determine the host compiler and its version
#  if defined(__INTEL_COMPILER)
#    ifndef THRUST_IGNORE_DEPRECATED_COMPILER
#      warning \
        "The Intel C++ Compiler Classic (icc/icpc) is not supported by CCCL. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this message."
#    endif // !THRUST_IGNORE_DEPRECATED_COMPILER
#  elif defined(__NVCOMPILER)
#    undef THRUST_COMPILER_NVHPC
#    define THRUST_COMPILER_NVHPC() (__NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__)
#  elif defined(__HIP__)
#    undef THRUST_COMPILER_HIP
#    define THRUST_COMPILER_HIP() (__clang_major__, __clang_minor__)
#  elif defined(__clang__)
#    undef THRUST_COMPILER_CLANG
#    define THRUST_COMPILER_CLANG() (__clang_major__, __clang_minor__)
#  elif defined(__GNUC__)
#    undef THRUST_COMPILER_GCC
#    define THRUST_COMPILER_GCC() (__GNUC__, __GNUC_MINOR__)
#  elif defined(_MSC_VER)
#    undef THRUST_COMPILER_MSVC
#    define THRUST_COMPILER_MSVC() (_MSC_VER / 100, _MSC_VER % 100)
#    if THRUST_COMPILER(MSVC, <, 19, 20)
#      ifndef THRUST_IGNORE_DEPRECATED_COMPILER
#        error \
          "Visual Studio 2017 (MSC_VER < 1920) and older are not supported by rocThrust. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this error."
#      endif
#    endif // THRUST_COMPILER(MSVC, <, 19, 20)
#    if THRUST_COMPILER(MSVC, >=, 19, 20) && THRUST_COMPILER(MSVC, <, 19, 30)
#      undef THRUST_COMPILER_MSVC2019
#      define THRUST_COMPILER_MSVC2019() THRUST_COMPILER_MSVC()
#    endif // THRUST_COMPILER(MSVC, >=, 19, 20) && THRUST_COMPILER(MSVC, <, 19, 30)
#    if THRUST_COMPILER(MSVC, >=, 19, 30) && THRUST_COMPILER(MSVC, <, 19, 40)
#      undef THRUST_COMPILER_MSVC2022
#      define THRUST_COMPILER_MSVC2022() THRUST_COMPILER_MSVC()
#    endif // THRUST_COMPILER(MSVC, >=, 19, 30) && THRUST_COMPILER(MSVC, <, 19, 40)
#  elif defined(__CUDACC_RTC__)
#    undef THRUST_COMPILER_NVRTC
#    define THRUST_COMPILER_NVRTC() (__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#  endif

// The CUDA compiler version shares the implementation with the C++ compiler
#  define THRUST_CUDA_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) THRUST_COMPILER_MAKE_VERSION(_MAJOR, _MINOR)
#  define THRUST_CUDA_COMPILER(...)                         THRUST_VERSION_COMPARE(THRUST_CUDA_COMPILER_, THRUST_CUDA_COMPILER_##__VA_ARGS__)

#  define THRUST_CUDA_COMPILER_NVCC()  THRUST_VERSION_INVALID()
#  define THRUST_CUDA_COMPILER_NVHPC() THRUST_VERSION_INVALID()
#  define THRUST_CUDA_COMPILER_CLANG() THRUST_VERSION_INVALID()
#  define THRUST_CUDA_COMPILER_NVRTC() THRUST_VERSION_INVALID()

// Determine the cuda compiler
#  if defined(__NVCC__)
#    undef THRUST_CUDA_COMPILER_NVCC
#    define THRUST_CUDA_COMPILER_NVCC() (__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#  elif defined(_NVHPC_CUDA)
#    undef THRUST_CUDA_COMPILER_NVHPC
#    define THRUST_CUDA_COMPILER_NVHPC() THRUST_COMPILER_NVHPC()
#  elif defined(__CUDA__) && THRUST_COMPILER(CLANG)
#    undef THRUST_CUDA_COMPILER_CLANG
#    define THRUST_CUDA_COMPILER_CLANG() THRUST_COMPILER_CLANG()
#  elif THRUST_COMPILER(NVRTC)
#    undef THRUST_CUDA_COMPILER_NVRTC
#    define THRUST_CUDA_COMPILER_NVRTC() THRUST_COMPILER_NVRTC()
#  endif // ^^^ THRUST_COMPILER(NVRTC) ^^^

#  if THRUST_CUDA_COMPILER(NVCC) || THRUST_CUDA_COMPILER(CLANG) || THRUST_CUDA_COMPILER(NVHPC) \
    || THRUST_CUDA_COMPILER(NVRTC)
#    define THRUST_HAS_CUDA_COMPILER() 1
#  else // ^^^ has cuda compiler ^^^ / vvv no cuda compiler vvv
#    define THRUST_HAS_CUDA_COMPILER() 0
#  endif // ^^^ no cuda compiler ^^^

#  if defined(__CUDACC__) || THRUST_CUDA_COMPILER(NVHPC)
#    define THRUST_CUDA_COMPILATION() 1
#  else // ^^^ compiling .cu file ^^^ / vvv not compiling .cu file vvv
#    define THRUST_CUDA_COMPILATION() 0
#  endif // ^^^ not compiling .cu file ^^^

#  if (THRUST_CUDA_COMPILATION() && defined(__CUDA_ARCH__)) || THRUST_CUDA_COMPILER(NVHPC) \
    || (THRUST_COMPILER(HIP) && defined(__HIP_DEVICE_COMPILE__))
#    define THRUST_DEVICE_COMPILATION() 1
#  else // ^^^ compiling device code ^^^ / vvv not compiling device code vvv
#    define THRUST_DEVICE_COMPILATION() 0
#  endif // ^^^ not compiling device code ^^^

#  define THRUST_CUDACC_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)

// clang-cuda does not define __CUDACC_VER_MAJOR__ and friends. They are instead retrieved from the CUDA_VERSION macro
// defined in "cuda.h". clang-cuda automatically pre-includes "__clang_cuda_runtime_wrapper.h" which includes "cuda.h"
#  if THRUST_CUDA_COMPILER(NVCC) || THRUST_CUDA_COMPILER(NVHPC) || THRUST_CUDA_COMPILER(NVRTC)
#    define THRUST_CUDACC() (__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#  elif THRUST_CUDA_COMPILER(CLANG)
#    define THRUST_CUDACC() (CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10)
#  endif // ^^^ has cuda compiler ^^^

#  if !defined(THRUST_CUDACC) || !THRUST_CUDA_COMPILATION()
#    undef THRUST_CUDACC
#    define THRUST_CUDACC() THRUST_VERSION_INVALID()
#  endif // !THRUST_CUDACC || !THRUST_CUDA_COMPILATION()

#  define THRUST_CUDACC_EQUAL(...) THRUST_VERSION_COMPARE(THRUST_CUDACC_, THRUST_CUDACC, ==, __VA_ARGS__)
#  define THRUST_CUDACC_BELOW(...) THRUST_VERSION_COMPARE(THRUST_CUDACC_, THRUST_CUDACC, <, __VA_ARGS__)

#  if THRUST_CUDA_COMPILATION() && THRUST_CUDACC_BELOW(12) && !defined(THRUST_IGNORE_DEPRECATED_CUDA_BELOW_12)
#    error "CUDA versions below 12 are not supported." \
"Define THRUST_IGNORE_DEPRECATED_CUDA_BELOW_12 to suppress this message."
#  endif

// Define the pragma for the host compiler
#  if THRUST_COMPILER(MSVC)
#    define THRUST_PRAGMA(ARG) __pragma(ARG)
#  else
#    define THRUST_PRAGMA(ARG) _Pragma(THRUST_TO_STRING(ARG))
#  endif // THRUST_COMPILER(MSVC)

#  if THRUST_DEVICE_COMPILATION()
#    define THRUST_PRAGMA_UNROLL_FULL() THRUST_PRAGMA(unroll)
#  elif THRUST_COMPILER(NVHPC) || THRUST_COMPILER(NVRTC) || THRUST_COMPILER(CLANG) || THRUST_COMPILER(HIP)
#    define THRUST_PRAGMA_UNROLL_FULL() THRUST_PRAGMA(unroll)
#  elif THRUST_COMPILER(GCC, >=, 8)
#    define THRUST_PRAGMA_UNROLL_FULL() THRUST_PRAGMA(diag_suppress 1675) THRUST_PRAGMA(GCC unroll 65534)
#  else // ^^^ has pragma unroll support ^^^ / vvv no pragma unroll support vvv
#    define THRUST_PRAGMA_UNROLL_FULL()
#  endif // ^^^ no pragma unroll support ^^^

#endif
