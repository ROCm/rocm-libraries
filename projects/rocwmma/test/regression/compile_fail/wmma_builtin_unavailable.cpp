/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// Using an operation whose builtin the compiler does not provide fails with a
// message that names the builtin. Simulated here by reporting every WMMA builtin
// unavailable.
#define ROCWMMA_TEST_NO_AMDGCN_BUILTINS

#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

__global__ void wmmaF16(float16_t const* a, float16_t const* b, float32_t* c)
{
    fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
    fragment<matrix_b, 16, 16, 16, float16_t, col_major> fb;
    fragment<accumulator, 16, 16, 16, float32_t>         fc;
    load_matrix_sync(fa, a, 16);
    load_matrix_sync(fb, b, 16);
    fill_fragment(fc, 0.0f);
    mma_sync(fc, fa, fb, fc);
    store_matrix_sync(c, fc, 16, mem_row_major);
}
