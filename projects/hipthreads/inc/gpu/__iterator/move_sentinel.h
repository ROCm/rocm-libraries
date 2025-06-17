// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef __GPU___ITERATOR_MOVE_SENTINEL_H
#define __GPU___ITERATOR_MOVE_SENTINEL_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

template <semiregular _Sent>
class _LIBGPU_TEMPLATE_VIS move_sentinel
{
public:
  __device__ _LIBGPU_HIDE_FROM_ABI
  move_sentinel() = default;

  __device__ _LIBGPU_HIDE_FROM_ABI constexpr
  explicit move_sentinel(_Sent __s) : __last_(std::move(__s)) {}

  template <class _S2>
    requires std::convertible_to<const _S2&, _Sent>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr
  move_sentinel(const move_sentinel<_S2>& __s) : __last_(__s.base()) {}

  template <class _S2>
    requires assignable_from<_Sent&, const _S2&>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr
  move_sentinel& operator=(const move_sentinel<_S2>& __s)
    { __last_ = __s.base(); return *this; }

  constexpr _Sent base() const { return __last_; }

private:
    _Sent __last_ = _Sent();
};

_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(move_sentinel);

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_MOVE_SENTINEL_H
