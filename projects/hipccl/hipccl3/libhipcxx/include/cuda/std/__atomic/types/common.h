//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _LIBCUDACXX___ATOMIC_TYPES_COMMON_H
#define _LIBCUDACXX___ATOMIC_TYPES_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstring>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __atomic_tag
{
  __atomic_base_tag,
  __atomic_locked_tag,
  __atomic_small_tag,
};

// Helpers to SFINAE on the tag inside the storage object
template <typename _Sto>
using __atomic_storage_is_base = enable_if_t<__atomic_tag::__atomic_base_tag == remove_cvref_t<_Sto>::__tag, int>;
template <typename _Sto>
using __atomic_storage_is_locked = enable_if_t<__atomic_tag::__atomic_locked_tag == remove_cvref_t<_Sto>::__tag, int>;
template <typename _Sto>
using __atomic_storage_is_small = enable_if_t<__atomic_tag::__atomic_small_tag == remove_cvref_t<_Sto>::__tag, int>;

template <typename _Tp>
using __atomic_underlying_t = typename _Tp::__underlying_t;
template <typename _Tp>
using __atomic_underlying_remove_cv_t = remove_cv_t<typename _Tp::__underlying_t>;

// [atomics.types.generic]p1 guarantees _Tp is trivially copyable. Because
// the default operator= in an object is not volatile, a byte-by-byte copy
// is required.
template <typename _Tp, typename _Tv>
_CCCL_HOST_DEVICE enable_if_t<_CCCL_TRAIT(is_assignable, _Tp&, _Tv)>
__atomic_assign_volatile(_Tp* __a_value, _Tv const& __val)
{
  *__a_value = __val;
}

template <typename _Tp, typename _Tv>
_CCCL_HOST_DEVICE enable_if_t<_CCCL_TRAIT(is_assignable, _Tp&, _Tv)>
__atomic_assign_volatile(_Tp volatile* __a_value, _Tv volatile const& __val)
{
  volatile char* __to         = reinterpret_cast<volatile char*>(__a_value);
  volatile char* __end        = __to + sizeof(_Tp);
  volatile const char* __from = reinterpret_cast<volatile const char*>(&__val);
  while (__to != __end)
  {
    *__to++ = *__from++;
  }
}

_CCCL_HOST_DEVICE inline int __atomic_memcmp(void const* __lhs, void const* __rhs, size_t __count)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE_LIBHIPCXX,
    // TODO(HIP/AMD): this is a temporal revert of an asm change made by Nvidia (possibly introducing UB that was fixed by the asm command)
    (auto __lhs_c = reinterpret_cast<unsigned char const*>(__lhs);
     auto __rhs_c = reinterpret_cast<unsigned char const*>(__rhs);

     while (__count--) {
       auto const __lhs_v = *__lhs_c++;
       auto const __rhs_v = *__rhs_c++;
       if (__lhs_v < __rhs_v)
       {
         return -1;
       }
       if (__lhs_v > __rhs_v)
       {
         return 1;
       }
     } return 0;),
    NV_IS_HOST_LIBHIPCXX,
    (return _CUDA_VSTD::memcmp(__lhs, __rhs, __count);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_TYPES_COMMON_H
