/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef KERNEL_HEADER
#define KERNEL_HEADER

#if Tensile_RUNTIME_LANGUAGE_OCL
#include <string>
#else

#include "TensileTypes.h"
#include <hip/hip_runtime.h>

#include "ReductionTemplate.h"
#include "memory_gfx.h"


__device__ inline int GenDot4(int a, int b, int c)
{
#if(__hcc_workweek__ >= 19092) || __HIP_CLANG_ONLY__
    union
    {
        int32_t i;
        char4   z;
    } va, vb;
#else
    typedef struct
    {
        int c0 : 8, c1 : 8, c2 : 8, c3 : 8;
    } C4I8;
    union
    {
        int32_t i;
        C4I8    z;
    } va, vb;
#endif
    va.i = a;
    vb.i = b;

#if(__hcc_workweek__ >= 19092) || __HIP_CLANG_ONLY__
    return amd_mixed_dot(va.z, vb.z, c, true);
}
#else
    return c + (vb.z.c3 * va.z.c3 + vb.z.c2 * va.z.c2 + vb.z.c1 * va.z.c1 + vb.z.c0 * va.z.c0);
}
#endif

#endif // HIP

typedef _Float16 tensile_half2 __attribute__((ext_vector_type(2)));
typedef _Float16 tensile_half;

extern "C" __device__ tensile_half2 llvm_fma_v2f16(tensile_half2,
                                                   tensile_half2,
                                                   tensile_half2) __asm("llvm.fma.v2f16");

__device__ inline tensile_half2
    tensile_fmadd_half2(tensile_half2 multiplier, tensile_half2 multiplicand, tensile_half2 addend)
{
    tensile_half2 result;
    result = llvm_fma_v2f16(multiplier, multiplicand, addend);
    return result;
};

#endif
