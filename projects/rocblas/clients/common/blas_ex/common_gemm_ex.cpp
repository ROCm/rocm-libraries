/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "../common_helpers.hpp"
#include "testing_gemm_batched_ex.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched_ex.hpp"

#define INSTANTIATE_MIX(Ti_, To_, Tc_)                \
    INSTANTIATE_TESTS(gemm_ex, Ti_, To_, Tc_)         \
    INSTANTIATE_TESTS(gemm_batched_ex, Ti_, To_, Tc_) \
    INSTANTIATE_TESTS(gemm_strided_batched_ex, Ti_, To_, Tc_)

INSTANTIATE_MIX(signed char, int, int)
INSTANTIATE_MIX(rocblas_bfloat16, rocblas_bfloat16, float)
INSTANTIATE_MIX(rocblas_half, rocblas_half, float)
INSTANTIATE_MIX(rocblas_bfloat16, float, float)
INSTANTIATE_MIX(rocblas_half, float, float)

#define INSTANTIATE(T_)                            \
    INSTANTIATE_TESTS(gemm_ex, T_, T_, T_)         \
    INSTANTIATE_TESTS(gemm_batched_ex, T_, T_, T_) \
    INSTANTIATE_TESTS(gemm_strided_batched_ex, T_, T_, T_)

INSTANTIATE(rocblas_half)
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(rocblas_float_complex)
INSTANTIATE(rocblas_double_complex)
