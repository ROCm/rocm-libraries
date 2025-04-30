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

#define ROCBLAS_BETA_FEATURES_API

#include "../common_helpers.hpp"
#include "testing_gemm_batched_ex3.hpp"
#include "testing_gemm_ex3.hpp"
#include "testing_gemm_strided_batched_ex3.hpp"

#define INSTANTIATE_MIX(TiA_, TiB_, To_, Tc_)                 \
    INSTANTIATE_TESTS(gemm_ex3, TiA_, TiB_, To_, Tc_)         \
    INSTANTIATE_TESTS(gemm_batched_ex3, TiA_, TiB_, To_, Tc_) \
    INSTANTIATE_TESTS(gemm_strided_batched_ex3, TiA_, TiB_, To_, Tc_)

INSTANTIATE_MIX(rocblas_f8, rocblas_f8, float, float)
INSTANTIATE_MIX(rocblas_f8, float, rocblas_f8, float)
INSTANTIATE_MIX(rocblas_f8, rocblas_f8, rocblas_f8, float)
INSTANTIATE_MIX(rocblas_f8, rocblas_bf8, float, float)
INSTANTIATE_MIX(rocblas_f8, rocblas_bf8, rocblas_bf8, float)
INSTANTIATE_MIX(rocblas_f8, rocblas_f8, rocblas_bf8, float)
INSTANTIATE_MIX(rocblas_f8, rocblas_f8, rocblas_half, float)
INSTANTIATE_MIX(rocblas_f8, rocblas_bf8, rocblas_half, float)

INSTANTIATE_MIX(rocblas_bf8, rocblas_bf8, float, float)
INSTANTIATE_MIX(rocblas_bf8, rocblas_bf8, rocblas_bf8, float)
INSTANTIATE_MIX(rocblas_bf8, rocblas_bf8, rocblas_f8, float)
INSTANTIATE_MIX(rocblas_bf8, rocblas_f8, rocblas_bf8, float)
INSTANTIATE_MIX(rocblas_bf8, rocblas_f8, float, float)
INSTANTIATE_MIX(rocblas_bf8, float, rocblas_bf8, float)
INSTANTIATE_MIX(rocblas_bf8, float, float, float)
INSTANTIATE_MIX(rocblas_bf8, rocblas_bf8, rocblas_half, float)
INSTANTIATE_MIX(rocblas_bf8, rocblas_f8, rocblas_half, float)

INSTANTIATE_MIX(rocblas_half, rocblas_half, rocblas_half, rocblas_half)
INSTANTIATE_MIX(rocblas_half, rocblas_half, rocblas_half, float)
INSTANTIATE_MIX(rocblas_half, float, float, float)
INSTANTIATE_MIX(rocblas_half, rocblas_half, float, float)

INSTANTIATE_MIX(float, float, float, float)
INSTANTIATE_MIX(float, rocblas_f8, rocblas_f8, float)
INSTANTIATE_MIX(float, rocblas_bf8, rocblas_bf8, float)
INSTANTIATE_MIX(float, rocblas_bf8, float, float)
