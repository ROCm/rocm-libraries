/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "client_utility.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define DIM1 1023
#define DIM2 1024
#define DIM3 1025

template <typename T>
void mat_mat_mult(T        alpha,
                  T        beta,
                  int      M,
                  int      N,
                  int      K,
                  const T* A,
                  int      As1,
                  int      As2,
                  const T* B,
                  int      Bs1,
                  int      Bs2,
                  T*       C,
                  int      Cs1,
                  int      Cs2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
            }
            C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

int main()
{
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    float             alpha = 1.1, beta = 0.9;

    rocblas_int m = DIM1, n = DIM2, k = DIM3;
    rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
    int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    rocblas_cout << "sgemm example" << std::endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a     = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        rocblas_cout << "N";
    }
    else
    {
        lda        = k;
        size_a     = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        rocblas_cout << "T";
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b     = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        rocblas_cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b     = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        rocblas_cout << "T: ";
    }
    ldc    = m;
    size_c = n * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hc_gold(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
    }
    hc_gold = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_ERROR(
        rocblas_sgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(float) * size_c, hipMemcpyDeviceToHost));

    rocblas_cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda
                 << ", " << ldb << ", " << ldc << std::endl;

    float max_relative_error = std::numeric_limits<float>::min();

    // calculate golden or correct result
    mat_mat_mult<float>(alpha,
                        beta,
                        m,
                        n,
                        k,
                        ha.data(),
                        a_stride_1,
                        a_stride_2,
                        hb.data(),
                        b_stride_1,
                        b_stride_2,
                        hc_gold.data(),
                        1,
                        ldc);

    for(int i = 0; i < size_c; i++)
    {
        float relative_error = (hc_gold[i] - hc[i]) / hc_gold[i];
        relative_error       = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error
            = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps       = std::numeric_limits<float>::epsilon();
    float tolerance = 10;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        rocblas_cout << "FAIL: max_relative_error = " << max_relative_error << std::endl;
    }
    else
    {
        rocblas_cout << "PASS: max_relative_error = " << max_relative_error << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
