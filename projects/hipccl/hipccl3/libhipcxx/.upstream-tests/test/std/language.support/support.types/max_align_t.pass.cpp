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

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type
#include "test_macros.h"

#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
#  include <stdio.h>
#endif // TEST_COMPILER(NVRTC)

int main(int, char**)
{
#if TEST_STD_VER > 2017
  //  P0767
  static_assert(cuda::std::is_trivial<cuda::std::max_align_t>::value,
                "cuda::std::is_trivial<cuda::std::max_align_t>::value");
  static_assert(cuda::std::is_standard_layout<cuda::std::max_align_t>::value,
                "cuda::std::is_standard_layout<cuda::std::max_align_t>::value");
#else
  static_assert(cuda::std::is_pod<cuda::std::max_align_t>::value, "cuda::std::is_pod<cuda::std::max_align_t>::value");
#endif
  static_assert((cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long long>::value),
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long long>::value");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long double>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long double>::value");
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<void*>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<void*>::value");

  return 0;
}
