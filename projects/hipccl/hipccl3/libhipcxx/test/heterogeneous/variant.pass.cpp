//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc, hiprtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include <cuda/std/cassert>
#include <cuda/std/variant>

#include "helpers.h"

struct pod
{
  int val;

  __host__ __device__ friend bool operator==(pod lhs, pod rhs)
  {
    return lhs.val == rhs.val;
  }
};

using variant_t = cuda::std::variant<int, pod, double>;

template <typename T, int Val>
struct tester
{
  template <typename Variant>
  __host__ __device__ static void initialize(Variant&& v)
  {
    v = T{Val};
  }

  template <typename Variant>
  __host__ __device__ static void validate(Variant&& v)
  {
    assert(cuda::std::holds_alternative<T>(v));
    assert(cuda::std::get<T>(v) == T{Val});
  }
};

using testers =
  tester_list<tester<int, 10>, tester<int, 20>, tester<pod, 30>, tester<pod, 40>, tester<double, 50>, tester<double, 60>>;

void kernel_invoker()
{
  variant_t v;
  validate_pinned<variant_t, testers>(v);
}

int main(int arg, char** argv)
{
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  kernel_invoker();
#endif

  return 0;
}
