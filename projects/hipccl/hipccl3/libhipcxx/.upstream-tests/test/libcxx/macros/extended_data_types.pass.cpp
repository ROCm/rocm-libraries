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
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cuda/std/__cccl/extended_data_types.h>

#include "test_macros.h"
#ifdef __HIP_PLATFORM_AMD__

#if _CCCL_HAS_NVFP4()
#  include <hip/hip_fp4.h>
#endif
#if _CCCL_HAS_NVFP6_E2M3()
#  include <hip/hip_fp6.h>
#endif
#if _CCCL_HAS_NVFP8()
#  include <hip/hip_fp8.h>
#endif
#if _CCCL_HAS_NVFP16()
#  include <hip/hip_fp16.h>
#endif
#if _CCCL_HAS_NVBF16()
#  include <hip/hip_bf16.h>
#endif

#else

#if _CCCL_HAS_NVFP4()
#  include <cuda_fp4.h>
#endif
#if _CCCL_HAS_NVFP6_E2M3()
#  include <cuda_fp6.h>
#endif
#if _CCCL_HAS_NVFP8()
#  include <cuda_fp8.h>
#endif
#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif
#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif

#endif

template <class T>
__host__ __device__ void test_nv_fp()
{
  auto v = T{1.0f};
  unused(v);
}

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  auto a = __int128(123456789123) + __int128(123456789123);
  auto b = __uint128_t(123456789123) + __uint128_t(123456789123);
  unused(a, b);
#endif

#if _CCCL_HAS_NVFP4_E2M1()
  test_nv_fp<__nv_fp4_e2m1>();
#endif
#if _CCCL_HAS_NVFP6_E3M2()
  test_nv_fp<__nv_fp6_e3m2>();
#endif
#if _CCCL_HAS_NVFP6_E2M3()
  test_nv_fp<__nv_fp6_e2m3>();
#endif
#if _CCCL_HAS_NVFP8_E4M3()
  test_nv_fp<__nv_fp8_e4m3>();
#endif
#if _CCCL_HAS_NVFP8_E5M2()
  test_nv_fp<__nv_fp8_e5m2>();
#endif
#if _CCCL_HAS_NVFP8_E8M0()
  test_nv_fp<__nv_fp8_e8m0>();
#endif
#if _CCCL_HAS_NVFP16()
  test_nv_fp<__half>();
#endif
#if _CCCL_HAS_NVBF16()
  test_nv_fp<__nv_bfloat16>();
#endif

#if _CCCL_HAS_FLOAT128()
  __float128 x5 = __float128(3.14) + __float128(3.14);
  unused(x5);
#endif

  return 0;
}
