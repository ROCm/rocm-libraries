//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__CCCL_COMPILER_H
#define LIBCXX_WRAPPER_STD__CCCL_COMPILER_H

// TODO(libhipcxx): check if libhipcxx supports THRUST_COMPILER_HIP, remove this file,
// and replace all THRUST* with _CCCL* in rocThrust once libhipcxx gets ready.

#include <thrust/detail/config/libcxx.h>

// 'libhipcxx' does not provide _CCCL_HAS_HIP_COMPILER, we must add it ourselves.
// When fully swapping over to libhipcxx, do not forget to port this properly!
#if defined(__HIP__)
#  define THRUST_HAS_HIP_COMPILER() 1
#else
#  define THRUST_HAS_HIP_COMPILER() 0
#endif // __HIP__

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_STD_INCLUDE(__cccl/compiler.h)

#  define THRUST_COMPILER(...) _CCCL_COMPILER(__VA_ARGS__)

#  define THRUST_CUDA_COMPILER(...)  _CCCL_CUDA_COMPILER(__VA_ARGS__)
#  define THRUST_HAS_CUDA_COMPILER() _CCCL_HAS_CUDA_COMPILER()
#  define THRUST_CUDA_COMPILATION()  _CCCL_CUDA_COMPILATION()

#  define THRUST_PRAGMA(ARG)          _CCCL_PRAGMA(ARG)
#  define THRUST_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA_UNROLL_FULL()

#else

#  include <thrust/detail/libcxx_wrapper/std/__cccl/deprecated.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/preprocessor.h>

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
#  define THRUST_COMPILER_CLANG()    THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_GCC()      THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_MSVC()     THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_MSVC2019() THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_MSVC2022() THRUST_VERSION_INVALID()
#  define THRUST_COMPILER_NVRTC()    THRUST_VERSION_INVALID()

#  if defined(__INTEL_COMPILER)
#    ifndef THRUST_IGNORE_DEPRECATED_COMPILER
#      warning \
        "The Intel C++ Compiler Classic (icc/icpc) is not supported by CCCL. Define THRUST_IGNORE_DEPRECATED_COMPILER to suppress this message."
#    endif // !THRUST_IGNORE_DEPRECATED_COMPILER
#  elif defined(__NVCOMPILER)
#    undef THRUST_COMPILER_NVHPC
#    define THRUST_COMPILER_NVHPC() (__NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__)
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

#  define THRUST_CUDA_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) THRUST_COMPILER_MAKE_VERSION(_MAJOR, _MINOR)
#  define THRUST_CUDA_COMPILER(...)                         THRUST_VERSION_COMPARE(THRUST_CUDA_COMPILER_, THRUST_CUDA_COMPILER_##__VA_ARGS__)

#  define THRUST_CUDA_COMPILER_NVCC()  THRUST_VERSION_INVALID()
#  define THRUST_CUDA_COMPILER_NVHPC() THRUST_VERSION_INVALID()
#  define THRUST_CUDA_COMPILER_CLANG() THRUST_VERSION_INVALID()
#  define THRUST_CUDA_COMPILER_NVRTC() THRUST_VERSION_INVALID()

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

#  if (THRUST_CUDA_COMPILATION() && defined(__CUDA_ARCH__)) || THRUST_CUDA_COMPILER(NVHPC)
#    define THRUST_DEVICE_COMPILATION() 1
#  else // ^^^ compiling device code ^^^ / vvv not compiling device code vvv
#    define THRUST_DEVICE_COMPILATION() 0
#  endif // ^^^ not compiling device code ^^^

#  define THRUST_CUDACC_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)

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

#  if THRUST_COMPILER(MSVC)
#    define THRUST_PRAGMA(ARG) __pragma(ARG)
#  else
#    define THRUST_PRAGMA(ARG) _Pragma(THRUST_TO_STRING(ARG))
#  endif // THRUST_COMPILER(MSVC)

#  if THRUST_DEVICE_COMPILATION()
#    define THRUST_PRAGMA_UNROLL(_N)    THRUST_PRAGMA(unroll _N)
#    define THRUST_PRAGMA_UNROLL_FULL() THRUST_PRAGMA(unroll)
#  elif THRUST_COMPILER(NVHPC) || THRUST_COMPILER(NVRTC) || THRUST_COMPILER(CLANG)
#    define THRUST_PRAGMA_UNROLL(_N)    THRUST_PRAGMA(unroll _N)
#    define THRUST_PRAGMA_UNROLL_FULL() THRUST_PRAGMA(unroll)
#  elif THRUST_COMPILER(GCC, >=, 8)
#    define THRUST_PRAGMA_UNROLL(_N) \
      THRUST_BEGIN_NV_DIAG_SUPPRESS(1675) THRUST_PRAGMA(GCC unroll _N) THRUST_END_NV_DIAG_SUPPRESS()
#    define THRUST_PRAGMA_UNROLL_FULL() THRUST_PRAGMA_UNROLL(65534)
#  else // ^^^ has pragma unroll support ^^^ / vvv no pragma unroll support vvv
#    define THRUST_PRAGMA_UNROLL(_N)
#    define THRUST_PRAGMA_UNROLL_FULL()
#  endif // ^^^ no pragma unroll support ^^^

#endif

#endif // LIBCXX_WRAPPER_STD__CCCL_COMPILER_H
