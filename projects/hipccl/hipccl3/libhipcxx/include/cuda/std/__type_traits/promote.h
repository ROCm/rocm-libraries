//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H
#define _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#ifdef _LIBCUDACXX_HAS_NVFP16
#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_fp16.h>
#else
#  include <cuda_fp16.h>
#endif
#endif // _LIBCUDACXX_HAS_NVFP16

#ifdef _LIBCUDACXX_HAS_NVBF16
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_bf16.h>
#else
#  include <cuda_bf16.h>
#endif
_CCCL_DIAG_POP
#endif // _LIBCUDACXX_HAS_NVBF16

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __numeric_type
{
  _LIBCUDACXX_HIDE_FROM_ABI static void __test(...);
#ifdef _LIBCUDACXX_HAS_NVFP16
  _LIBCUDACXX_HIDE_FROM_ABI static __half __test(__half);
#endif // _LIBCUDACXX_HAS_NVBF16
#ifdef _LIBCUDACXX_HAS_NVBF16
  _LIBCUDACXX_HIDE_FROM_ABI static __hip_bfloat16 __test(__hip_bfloat16);
#endif // _LIBCUDACXX_HAS_NVFP16
  _LIBCUDACXX_HIDE_FROM_ABI static float __test(float);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(char);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(int);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(unsigned);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(long);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(unsigned long);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(long long);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(unsigned long long);
  _LIBCUDACXX_HIDE_FROM_ABI static double __test(double);
  _LIBCUDACXX_HIDE_FROM_ABI static long double __test(long double);

  typedef decltype(__test(declval<_Tp>())) type;
  static const bool value = _IsNotSame<type, void>::value;
};

template <>
struct __numeric_type<void>
{
  static const bool value = true;
};

template <class _A1,
          class _A2 = void,
          class _A3 = void,
          bool      = __numeric_type<_A1>::value && __numeric_type<_A2>::value && __numeric_type<_A3>::value>
class __promote_imp
{
public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;

public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;

public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3>
{};

template <class _A1, class _A2 = void, class _A3 = void>
using __promote_t _CCCL_NODEBUG_ALIAS = typename __promote<_A1, _A2, _A3>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_PROMOTE_H
