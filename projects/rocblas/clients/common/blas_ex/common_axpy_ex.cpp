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
#include "testing_axpy_batched_ex.hpp"
#include "testing_axpy_ex.hpp"
#include "testing_axpy_strided_batched_ex.hpp"

#define INSTANTIATE_HPA(Ta_, Tx_, Ty_, Tex_)                \
    INSTANTIATE_TESTS(axpy_ex, Ta_, Tx_, Ty_, Tex_)         \
    INSTANTIATE_TESTS(axpy_batched_ex, Ta_, Tx_, Ty_, Tex_) \
    INSTANTIATE_TESTS(axpy_strided_batched_ex, Ta_, Tx_, Ty_, Tex_)

INSTANTIATE_HPA(rocblas_half, rocblas_half, float, float)
INSTANTIATE_HPA(rocblas_half, rocblas_half, rocblas_half, float)
INSTANTIATE_HPA(float, rocblas_half, rocblas_half, float)
INSTANTIATE_HPA(rocblas_bfloat16, rocblas_bfloat16, rocblas_bfloat16, float)
INSTANTIATE_HPA(float, rocblas_bfloat16, rocblas_bfloat16, float)

#define INSTANTIATE(T_)                    \
    INSTANTIATE_TESTS(axpy_ex, T_)         \
    INSTANTIATE_TESTS(axpy_batched_ex, T_) \
    INSTANTIATE_TESTS(axpy_strided_batched_ex, T_)

INSTANTIATE(rocblas_half)
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(rocblas_float_complex)
INSTANTIATE(rocblas_double_complex)
