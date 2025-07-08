//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: hipcc, hiprtc
// NOTE(HIP): This test includes the header cuda/std/vector which does not exist.
// HIPCC gives the correct error: cuda/std/string file not found.
// However, other errors are expected, consider the expected-error. 
// We found _LIBCUDACXX_HAS_VECTOR guard in other tests that include the header, 
// indicating that the header is not implemented yet. 
// <cuda/std/array>

// bool operator==(array<T, N> const&, array<T, N> const&);
// bool operator!=(array<T, N> const&, array<T, N> const&);
// bool operator<(array<T, N> const&, array<T, N> const&);
// bool operator<=(array<T, N> const&, array<T, N> const&);
// bool operator>(array<T, N> const&, array<T, N> const&);
// bool operator>=(array<T, N> const&, array<T, N> const&);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/vector>

#include "test_macros.h"

template <class Array>
void test_compare(const Array& LHS, const Array& RHS)
{
  typedef cuda::std::vector<typename Array::value_type> Vector;
  const Vector LHSV(LHS.begin(), LHS.end());
  const Vector RHSV(RHS.begin(), RHS.end());
  assert((LHS == RHS) == (LHSV == RHSV));
  assert((LHS != RHS) == (LHSV != RHSV));
  assert((LHS < RHS) == (LHSV < RHSV));
  assert((LHS <= RHS) == (LHSV <= RHSV));
  assert((LHS > RHS) == (LHSV > RHSV));
  assert((LHS >= RHS) == (LHSV >= RHSV));
}

template <int Dummy>
struct NoCompare
{};

int main(int, char**)
{
  {
    typedef NoCompare<0> T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@*:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD(c1 == c1);
    TEST_IGNORE_NODISCARD(c1 < c1);
  }
  {
    typedef NoCompare<1> T;
    typedef cuda::std::array<T, 3> C;
    C c1 = {{}};
    // expected-error@*:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD(c1 != c1);
    TEST_IGNORE_NODISCARD(c1 > c1);
  }
  {
    typedef NoCompare<2> T;
    typedef cuda::std::array<T, 0> C;
    C c1 = {{}};
    // expected-error@*:* 2 {{invalid operands to binary expression}}
    TEST_IGNORE_NODISCARD(c1 == c1);
    TEST_IGNORE_NODISCARD(c1 < c1);
  }

  return 0;
}
