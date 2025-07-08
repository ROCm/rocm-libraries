// -*- C++ -*-
//===--------------------------- string.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef _LIBCUDACXX_STRING_H
#define _LIBCUDACXX_STRING_H

/*
    string.h synopsis

Macros:

    NULL

Types:

    size_t

void* memcpy(void* restrict s1, const void* restrict s2, size_t n);
void* memmove(void* s1, const void* s2, size_t n);
char* strcpy (char* restrict s1, const char* restrict s2);
char* strncpy(char* restrict s1, const char* restrict s2, size_t n);
char* strcat (char* restrict s1, const char* restrict s2);
char* strncat(char* restrict s1, const char* restrict s2, size_t n);
int memcmp(const void* s1, const void* s2, size_t n);
int strcmp (const char* s1, const char* s2);
int strncmp(const char* s1, const char* s2, size_t n);
int strcoll(const char* s1, const char* s2);
size_t strxfrm(char* restrict s1, const char* restrict s2, size_t n);
const void* memchr(const void* s, int c, size_t n);
      void* memchr(      void* s, int c, size_t n);
const char* strchr(const char* s, int c);
      char* strchr(      char* s, int c);
size_t strcspn(const char* s1, const char* s2);
const char* strpbrk(const char* s1, const char* s2);
      char* strpbrk(      char* s1, const char* s2);
const char* strrchr(const char* s, int c);
      char* strrchr(      char* s, int c);
size_t strspn(const char* s1, const char* s2);
const char* strstr(const char* s1, const char* s2);
      char* strstr(      char* s1, const char* s2);
char* strtok(char* restrict s1, const char* restrict s2);
void* memset(void* s, int c, size_t n);
char* strerror(int errnum);
size_t strlen(const char* s);

*/

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <string.h>
#endif // !_CCCL_COMPILER_NVRTC

// MSVCRT, GNU libc and its derivates may already have the correct prototype in
// <string.h>. This macro can be defined by users if their C library provides
// the right signature.
// NOTE(HIP): Clang already overloads these APIs, so we dont need to do it here.
// /opt/rocm-6.5.0/lib/llvm/lib/clang/19/include/llvm_libc_wrappers/string.h
#if defined(__CORRECT_ISO_CPP_STRING_H_PROTO) || defined(_LIBCUDACXX_MSVCRT) || defined(__sun__) \
  || defined(_STRING_H_CPLUSPLUS_98_CONFORMANCE_) || defined(_CCCL_COMPILER_HIPRTC)
#  define _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS
#endif

#if defined(__cplusplus) && !defined(_LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS) \
  && defined(_LIBCUDACXX_PREFERRED_OVERLOAD)
extern "C++" {
_LIBCUDACXX_HIDE_FROM_ABI char* __libcpp_strchr(const char* __s, int __c)
{
  return (char*) strchr(__s, __c);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD const char* strchr(const char* __s, int __c)
{
  return __libcpp_strchr(__s, __c);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD char* strchr(char* __s, int __c)
{
  return __libcpp_strchr(__s, __c);
}

_LIBCUDACXX_HIDE_FROM_ABI char* __libcpp_strpbrk(const char* __s1, const char* __s2)
{
  return (char*) strpbrk(__s1, __s2);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD const char* strpbrk(const char* __s1, const char* __s2)
{
  return __libcpp_strpbrk(__s1, __s2);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD char* strpbrk(char* __s1, const char* __s2)
{
  return __libcpp_strpbrk(__s1, __s2);
}

_LIBCUDACXX_HIDE_FROM_ABI char* __libcpp_strrchr(const char* __s, int __c)
{
  return (char*) strrchr(__s, __c);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD const char* strrchr(const char* __s, int __c)
{
  return __libcpp_strrchr(__s, __c);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD char* strrchr(char* __s, int __c)
{
  return __libcpp_strrchr(__s, __c);
}

_LIBCUDACXX_HIDE_FROM_ABI void* __libcpp_memchr(const void* __s, int __c, size_t __n)
{
  return (void*) memchr(__s, __c, __n);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD const void* memchr(const void* __s, int __c, size_t __n)
{
  return __libcpp_memchr(__s, __c, __n);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD void* memchr(void* __s, int __c, size_t __n)
{
  return __libcpp_memchr(__s, __c, __n);
}

_LIBCUDACXX_HIDE_FROM_ABI char* __libcpp_strstr(const char* __s1, const char* __s2)
{
  return (char*) strstr(__s1, __s2);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD const char* strstr(const char* __s1, const char* __s2)
{
  return __libcpp_strstr(__s1, __s2);
}
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_PREFERRED_OVERLOAD char* strstr(char* __s1, const char* __s2)
{
  return __libcpp_strstr(__s1, __s2);
}
}
#endif

#endif // _LIBCUDACXX_STRING_H
