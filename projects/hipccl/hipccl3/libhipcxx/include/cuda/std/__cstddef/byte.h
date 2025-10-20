// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _LIBCUDACXX___CSTDDEF_BYTE_H
#define _LIBCUDACXX___CSTDDEF_BYTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION

enum class byte : unsigned char
{
};

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte operator|(byte __lhs, byte __rhs) noexcept
{
  return static_cast<byte>(
    static_cast<unsigned char>(static_cast<unsigned int>(__lhs) | static_cast<unsigned int>(__rhs)));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte& operator|=(byte& __lhs, byte __rhs) noexcept
{
  return __lhs = __lhs | __rhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte operator&(byte __lhs, byte __rhs) noexcept
{
  return static_cast<byte>(
    static_cast<unsigned char>(static_cast<unsigned int>(__lhs) & static_cast<unsigned int>(__rhs)));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte& operator&=(byte& __lhs, byte __rhs) noexcept
{
  return __lhs = __lhs & __rhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte operator^(byte __lhs, byte __rhs) noexcept
{
  return static_cast<byte>(
    static_cast<unsigned char>(static_cast<unsigned int>(__lhs) ^ static_cast<unsigned int>(__rhs)));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte& operator^=(byte& __lhs, byte __rhs) noexcept
{
  return __lhs = __lhs ^ __rhs;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr byte operator~(byte __b) noexcept
{
  return static_cast<byte>(static_cast<unsigned char>(~static_cast<unsigned int>(__b)));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_LIBCUDACXX_HIDE_FROM_ABI constexpr byte& operator<<=(byte& __lhs, _Integer __shift) noexcept
{
  return __lhs = __lhs << __shift;
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_LIBCUDACXX_HIDE_FROM_ABI constexpr byte operator<<(byte __lhs, _Integer __shift) noexcept
{
  return static_cast<byte>(static_cast<unsigned char>(static_cast<unsigned int>(__lhs) << __shift));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_LIBCUDACXX_HIDE_FROM_ABI constexpr byte& operator>>=(byte& __lhs, _Integer __shift) noexcept
{
  return __lhs = __lhs >> __shift;
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_LIBCUDACXX_HIDE_FROM_ABI constexpr byte operator>>(byte __lhs, _Integer __shift) noexcept
{
  return static_cast<byte>(static_cast<unsigned char>(static_cast<unsigned int>(__lhs) >> __shift));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_LIBCUDACXX_HIDE_FROM_ABI constexpr _Integer to_integer(byte __b) noexcept
{
  return static_cast<_Integer>(__b);
}

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___CSTDDEF_BYTE_H
