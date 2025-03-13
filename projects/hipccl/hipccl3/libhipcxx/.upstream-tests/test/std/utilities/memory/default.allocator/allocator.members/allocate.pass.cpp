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

// <memory>

// allocator:
// constexpr T* allocate(size_t n);

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef> // for cuda::std::max_align_t

#include "count_new.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

#if _LIBCUDACXX_HAS_ALIGNED_ALLOCATION()
static const bool UsingAlignedNew = true;
#else
static const bool UsingAlignedNew = false;
#endif

#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
TEST_GLOBAL_VARIABLE const cuda::std::size_t MaxAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else
TEST_GLOBAL_VARIABLE const cuda::std::size_t MaxAligned = cuda::std::alignment_of<cuda::std::max_align_t>::value;
#endif

TEST_GLOBAL_VARIABLE const cuda::std::size_t OverAligned = MaxAligned * 2;

TEST_GLOBAL_VARIABLE int AlignedType_constructed = 0;

template <cuda::std::size_t Align>
struct alignas(Align) AlignedType
{
  char data;
  __host__ __device__ AlignedType()
  {
    ++AlignedType_constructed;
  }
  __host__ __device__ AlignedType(AlignedType const&)
  {
    ++AlignedType_constructed;
  }
  __host__ __device__ ~AlignedType()
  {
    --AlignedType_constructed;
  }
};

template <cuda::std::size_t Align>
__host__ __device__ void test_aligned()
{
  typedef AlignedType<Align> T;
  AlignedType_constructed = 0;
  globalMemCounter.reset();
  cuda::std::allocator<T> a;
  const bool IsOverAlignedType = Align > MaxAligned;
  const bool ExpectAligned     = IsOverAlignedType && UsingAlignedNew;
  {
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(AlignedType_constructed == 0);
    globalMemCounter.last_new_size  = 0;
    globalMemCounter.last_new_align = 0;
    T* ap                           = a.allocate(3);
#if !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPCC) && !defined(TEST_COMPILER_HIPRTC)
    DoNotOptimize(ap);
#else
    const auto meow = reinterpret_cast<uintptr_t>(ap) + 2;
    (void) meow;
#endif // !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPCC) && !defined(TEST_COMPILER_HIPRTC)
    // assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(globalMemCounter.checkNewCalledEq(1));
    assert(globalMemCounter.checkAlignedNewCalledEq(ExpectAligned));
    assert(globalMemCounter.checkLastNewSizeEq(3 * sizeof(T)));
    assert(globalMemCounter.checkLastNewAlignEq(ExpectAligned ? Align : 0));
    assert(AlignedType_constructed == 0);
    globalMemCounter.last_delete_align = 0;
    a.deallocate(ap, 3);
    // assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(globalMemCounter.checkDeleteCalledEq(1));
    assert(globalMemCounter.checkAlignedDeleteCalledEq(ExpectAligned));
    assert(globalMemCounter.checkLastDeleteAlignEq(ExpectAligned ? Align : 0));
    assert(AlignedType_constructed == 0);
  }
}

#if TEST_STD_VER >= 2020
template <cuda::std::size_t Align>
__host__ __device__ constexpr bool test_aligned_constexpr()
{
  typedef AlignedType<Align> T;
  cuda::std::allocator<T> a;
  T* ap = a.allocate(3);
  a.deallocate(ap, 3);

  return true;
}
#endif // TEST_STD_VER >= 2020

int main(int, char**)
{
  test_aligned<1>();
  test_aligned<2>();
  test_aligned<4>();
  test_aligned<8>();
  test_aligned<16>();
  test_aligned<MaxAligned>();
  test_aligned<OverAligned>();
  test_aligned<OverAligned * 2>();

#if defined(_CCCL_HAS_CONSTEXPR_ALLOCATION)
  static_assert(test_aligned_constexpr<1>());
  static_assert(test_aligned_constexpr<2>());
  static_assert(test_aligned_constexpr<4>());
  static_assert(test_aligned_constexpr<8>());
  static_assert(test_aligned_constexpr<16>());
  static_assert(test_aligned_constexpr<MaxAligned>());
  static_assert(test_aligned_constexpr<OverAligned>());
  static_assert(test_aligned_constexpr<OverAligned * 2>());
#endif // _CCCL_HAS_CONSTEXPR_ALLOCATION

  return 0;
}
