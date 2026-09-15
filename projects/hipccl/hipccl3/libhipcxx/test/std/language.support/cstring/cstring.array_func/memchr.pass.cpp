//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>

constexpr int not_found = -1;

__host__ __device__ void test(const char* ptr, int c, size_t n, int expected_pos)
{
  const void* ret = cuda::std::memchr(ptr, c, n);

  if (expected_pos == not_found)
  {
    assert(ret == nullptr);
  }
  else
  {
    assert(ret != nullptr);
    assert(static_cast<const char*>(ret) == ptr + expected_pos);
  }
}

int main(int, char**)
{
  test("abcde", '\0', 6, 5);
  test("abcde", '\0', 5, not_found);
  test("aaabb", 'b', 5, 3);
  test("aaabb", 'b', 4, 3);
  test("aaabb", 'b', 3, not_found);
  test("aaaa", 'b', 4, not_found);
  test("aaaa", 'a', 0, not_found);

  return 0;
}
