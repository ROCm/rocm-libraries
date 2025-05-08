/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

template <typename T, typename MathOpT = T>
void mat_mat_mult(T   alpha,
                  T   beta,
                  int M,
                  int N,
                  int K,
                  T*  A,
                  int As1,
                  int As2,
                  T*  B,
                  int Bs1,
                  int Bs2,
                  T*  C,
                  int Cs1,
                  int Cs2,
                  T*  D,
                  int Ds1,
                  int Ds2)
{
    for(int i1 = 0; i1 < M; i1++)
    {
        for(int i2 = 0; i2 < N; i2++)
        {
            T t = 0.0;
            for(int i3 = 0; i3 < K; i3++)
            {
                t += T(MathOpT(A[i1 * As1 + i3 * As2])) * T(MathOpT(B[i3 * Bs1 + i2 * Bs2]));
            }
            D[i1 * Ds1 + i2 * Ds2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
        }
    }
}

int main()
{
    rocblas_datatype a_type       = rocblas_datatype_f32_r;
    rocblas_datatype b_type       = rocblas_datatype_f32_r;
    rocblas_datatype c_type       = rocblas_datatype_f32_r;
    rocblas_datatype d_type       = rocblas_datatype_f32_r;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    using a_t = float;
    using b_t = float;
    using c_t = float;
    using d_t = float;

    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;

    //rocblas_int m = 960;
    //rocblas_int n = 1024;
    //rocblas_int k = 1024;

    rocblas_int m = 127;
    rocblas_int n = 126;
    rocblas_int k = 126;

    float alpha       = 1.0;
    float beta        = 1.0;
    int   batch_count = 2;

    rocblas_stride stride_a, stride_b, stride_c, stride_d;
    size_t         lda, ldb, ldc = m, ldd = m;
    size_t         size_a1, size_b1, size_c1, size_d1;
    size_c1 = size_d1 = m * n;

    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a1    = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        std::cout << "N";
    }
    else
    {
        lda        = k;
        size_a1    = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        std::cout << "T";
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b1    = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        std::cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b1    = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        std::cout << "T: ";
    }
    stride_a = size_a1;
    stride_b = size_b1;
    stride_c = size_c1;
    stride_d = size_d1;

    std::cout << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", " << ldc << ", "
              << ldd << ", " << stride_a << ", " << stride_b << ", " << stride_c << ", " << stride_d
              << ", " << batch_count << ", " << alpha << ", " << beta << ", ";

    size_t size_a = batch_count == 0 ? size_a1 : size_a1 + stride_a * (batch_count - 1);
    size_t size_b = batch_count == 0 ? size_b1 : size_b1 + stride_b * (batch_count - 1);
    size_t size_c = batch_count == 0 ? size_c1 : size_c1 + stride_c * (batch_count - 1);
    size_t size_d = batch_count == 0 ? size_d1 : size_d1 + stride_d * (batch_count - 1);

    std::cout << "gemm_ex example" << std::endl;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    auto ha      = std::make_unique<a_t[]>(size_a);
    auto hb      = std::make_unique<b_t[]>(size_b);
    auto hc      = std::make_unique<c_t[]>(size_c);
    auto hd      = std::make_unique<d_t[]>(size_d);
    auto hd_gold = std::make_unique<d_t[]>(size_d);

    // initial data on host
    srand(1);
    for(size_t i = 0; i < size_a; ++i)
        ha[i] = rand() % 17;
    for(size_t i = 0; i < size_b; ++i)
        hb[i] = rand() % 17;
    for(size_t i = 0; i < size_c; ++i)
        hc[i] = rand() % 17;
    for(size_t i = 0; i < size_d; ++i)
        hd[i] = std::numeric_limits<d_t>::signaling_NaN();

    // allocate memory on device
    a_t* da;
    b_t* db;
    c_t* dc;
    d_t* dd;

    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(a_t)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(b_t)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(c_t)));
    CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(d_t)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, &ha[0], sizeof(a_t) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, &hb[0], sizeof(b_t) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, &hc[0], sizeof(c_t) * size_c, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dd, &hd[0], sizeof(d_t) * size_d, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    auto     algo           = rocblas_gemm_algo_standard;
    int32_t  solution_index = 0;
    uint32_t flags          = 0;
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, rocblas_xf32_xdl_math_op));
    rocblas_math_mode math_mode;
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(handle,
                                                        transa,
                                                        transb,
                                                        m,
                                                        n,
                                                        k,
                                                        &alpha,
                                                        da,
                                                        a_type,
                                                        lda,
                                                        stride_a,
                                                        db,
                                                        b_type,
                                                        ldb,
                                                        stride_b,
                                                        &beta,
                                                        dc,
                                                        c_type,
                                                        ldc,
                                                        stride_c,
                                                        dd,
                                                        d_type,
                                                        ldd,
                                                        stride_d,
                                                        batch_count,
                                                        compute_type,
                                                        algo,
                                                        solution_index,
                                                        flags));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(&hd[0], dd, sizeof(d_t) * size_d, hipMemcpyDeviceToHost));

    // calculate golden or correct result
    for(int i = 0; i < batch_count; i++)
    {
        a_t* a_ptr = &ha[i * stride_a];
        b_t* b_ptr = &hb[i * stride_b];
        c_t* c_ptr = &hc[i * stride_d];
        d_t* d_ptr = &hd_gold[i * stride_d];
        if(math_mode == rocblas_xf32_xdl_math_op)
            mat_mat_mult<a_t, rocblas_xfloat32>(alpha,
                                                beta,
                                                m,
                                                n,
                                                k,
                                                a_ptr,
                                                a_stride_1,
                                                a_stride_2,
                                                b_ptr,
                                                b_stride_1,
                                                b_stride_2,
                                                c_ptr,
                                                1,
                                                ldc,
                                                d_ptr,
                                                1,
                                                ldd);
        else
            mat_mat_mult<a_t>(alpha,
                              beta,
                              m,
                              n,
                              k,
                              a_ptr,
                              a_stride_1,
                              a_stride_2,
                              b_ptr,
                              b_stride_1,
                              b_stride_2,
                              c_ptr,
                              1,
                              ldc,
                              d_ptr,
                              1,
                              ldd);
    }

    double max_relative_error = 0;
    for(size_t i = 0; i < size_d; i++)
    {
        auto relative_error = fabs(double(hd_gold[i] - hd[i]) / hd_gold[i]);
        if(relative_error > max_relative_error)
            max_relative_error = relative_error;
    }

    auto eps       = std::numeric_limits<d_t>::epsilon();
    auto tolerance = 10.0;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
        std::cout << "FAIL: ";
    else
        std::cout << "PASS: ";
    std::cout << "max_relative_error = " << max_relative_error << std::endl;

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
