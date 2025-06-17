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

#ifndef __GPU___ALGORITHM_IN_OUT_OUT_RESULT_H
#define __GPU___ALGORITHM_IN_OUT_OUT_RESULT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

namespace ranges {
template <class _InIter1, class _OutIter1, class _OutIter2>
struct in_out_out_result {
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter1 in;
  _LIBGPU_NO_UNIQUE_ADDRESS _OutIter1 out1;
  _LIBGPU_NO_UNIQUE_ADDRESS _OutIter2 out2;

  template <class _InIter2, class _OutIter3, class _OutIter4>
    requires std::convertible_to<const _InIter1&, _InIter2>
          && std::convertible_to<const _OutIter1&, _OutIter3> && std::convertible_to<const _OutIter2&, _OutIter4>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr
  operator in_out_out_result<_InIter2, _OutIter3, _OutIter4>() const& {
    return {in, out1, out2};
  }

  template <class _InIter2, class _OutIter3, class _OutIter4>
    requires std::convertible_to<_InIter1, _InIter2>
          && std::convertible_to<_OutIter1, _OutIter3> && std::convertible_to<_OutIter2, _OutIter4>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr
  operator in_out_out_result<_InIter2, _OutIter3, _OutIter4>() && {
    return {std::move(in), std::move(out1), std::move(out2)};
  }
};
} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_IN_OUT_OUT_RESULT_H
