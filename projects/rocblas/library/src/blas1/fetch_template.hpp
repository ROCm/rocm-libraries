/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

#pragma once

#include "rocblas.h"

template <typename T>
__device__ __host__ inline auto fetch_asum(T A)
{
    return A < 0 ? -A : A;
}

__device__ __host__ inline auto fetch_asum(const rocblas_float_complex& A)
{
    return asum(A);
}

__device__ __host__ inline auto fetch_asum(const rocblas_double_complex& A)
{
    return asum(A);
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return std::norm(A);
}

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline auto fetch_abs2(T A)
{
    return A * A;
}
