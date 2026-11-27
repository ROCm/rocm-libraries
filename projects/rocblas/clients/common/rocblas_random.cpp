/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_random.hpp"

const rocblas_rng_t        g_rocblas_seed(69069); // A fixed seed to start at
thread_local rocblas_rng_t t_rocblas_rng = g_rocblas_seed;

/* ============================================================================================ */

// Maps 32 random bits to [1, 10] without a division; the bias, at most 10 / 2^32, is negligible
static inline float uniform_int_1_10(uint64_t bits)
{
    return float(((bits & 0xffffffffull) * 10 >> 32) + 1);
}

float rocblas_uniform_int_1_10()
{
    return uniform_int_1_10(t_rocblas_rng());
}

void rocblas_uniform_int_1_10_run_float(float* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
        ptr[i] = rocblas_uniform_int_1_10();
}

void rocblas_uniform_int_1_10_run_double(double* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
        ptr[i] = rocblas_uniform_int_1_10();
}

void rocblas_uniform_int_1_10_run_float_complex(rocblas_float_complex* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
        ptr[i] = random_generator<rocblas_float_complex>();
}

void rocblas_uniform_int_1_10_run_double_complex(rocblas_double_complex* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
        ptr[i] = random_generator<rocblas_double_complex>();
}
