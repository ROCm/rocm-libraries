/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "blas3/rocblas_gemm.hpp"
#include "frequency_monitor.hpp"
#include "testing_common.hpp"

template <typename T>
void testing_gemm_batched_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_gemm_batched<T, true>
                                                                  : rocblas_gemm_batched<T, false>;
        auto rocblas_gemm_batched_fn_64 = arg.api & c_API_FORTRAN
                                              ? rocblas_gemm_batched_64<T, true>
                                              : rocblas_gemm_batched_64<T, false>;

        const int64_t M = 100;
        const int64_t N = 100;
        const int64_t K = 100;

        const int64_t lda = 100;
        const int64_t ldb = 100;
        const int64_t ldc = 100;

        const size_t safe_size = N * ldc;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, one_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        int64_t batch_count = 2;

        // allocate memory on device
        DEVICE_MEMCHECK(device_batch_vector<T>, dA, (safe_size, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dB, (safe_size, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dC, (safe_size, 1, batch_count));

        // clang-format off

// check for valid enum
DAPI_EXPECT( rocblas_status_invalid_value,rocblas_gemm_batched_fn,(handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_value,rocblas_gemm_batched_fn,(handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

// check for invalid size
DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, transA, transB, -1, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, transA, transB, M, -1, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, -1, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, -1) );

// check for invalid leading dimension
DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA.ptr_on_device(), M-1, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), K-1, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, alpha,
dA.ptr_on_device(), K-1, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), N-1, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_batched_fn,(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), M-1, batch_count) );

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
DAPI_EXPECT( rocblas_status_invalid_handle,rocblas_gemm_batched_fn,(nullptr, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, nullptr,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, alpha,
nullptr, lda, dB.ptr_on_device(), ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), lda, nullptr, ldb, beta, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, nullptr, dC.ptr_on_device(), ldc, batch_count) );

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), lda, dB.ptr_on_device(), ldb, beta, nullptr, ldc, batch_count) );

// If batch_count==0, then all pointers can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, nullptr, nullptr, lda,
nullptr, ldb, nullptr, nullptr, ldc, 0));

// If M==0, then all pointers can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_fn,(handle, transA, transB, 0, N, K, nullptr, nullptr, lda,
nullptr, ldb, nullptr, nullptr, ldc, batch_count));

// If N==0, then A and B can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_fn,(handle, transA, transB, M, 0, K, nullptr, nullptr, lda,
nullptr, ldb, nullptr, nullptr, ldc, batch_count));

// the following tests still output to C

// If K==0, then alpha, A and B can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_fn,(handle, transA, transB, M, N, 0,
nullptr, nullptr, lda, nullptr, ldb, beta, dC.ptr_on_device(), ldc, batch_count));

// If alpha==0, then A and B can be nullptr without issue
DAPI_CHECK(rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, zero,
nullptr, lda, nullptr, ldb, beta, dC.ptr_on_device(), ldc, batch_count));

// If alpha==0 and beta==1, then A, B and C can be nullptr without issue
DAPI_CHECK(rocblas_gemm_batched_fn,(handle, transA, transB, M, N, K, zero,
nullptr, lda, nullptr, ldb, one, nullptr, ldc, batch_count));

        // clang-format on
    }
}

template <typename T>
void testing_gemm_batched(const Arguments& arg)
{
    auto rocblas_gemm_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_gemm_batched<T, true> : rocblas_gemm_batched<T, false>;
    auto rocblas_gemm_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_gemm_batched_64<T, true>
                                                              : rocblas_gemm_batched_64<T, false>;

    rocblas_local_handle handle{arg};

    int64_t           M           = arg.M;
    int64_t           N           = arg.N;
    int64_t           K           = arg.K;
    T                 h_alpha     = arg.get_alpha<T>();
    T                 h_beta      = arg.get_beta<T>();
    int64_t           lda         = arg.lda;
    int64_t           ldb         = arg.ldb;
    int64_t           ldc         = arg.ldc;
    int64_t           batch_count = arg.batch_count;
    rocblas_operation transA      = char2rocblas_operation(arg.transA);
    rocblas_operation transB      = char2rocblas_operation(arg.transB);

    int64_t A_row = transA == rocblas_operation_none ? M : std::max(K, int64_t(1));
    int64_t A_col = transA == rocblas_operation_none ? std::max(K, int64_t(1)) : M;
    int64_t B_row = transB == rocblas_operation_none ? std::max(K, int64_t(1)) : N;
    int64_t B_col = transB == rocblas_operation_none ? N : std::max(K, int64_t(1));

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, math_mode));
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    // for internal interface testing
    // using arg.stride_x,y,d for offset testing
    rocblas_stride offsetA = arg.api == INTERNAL ? arg.stride_x : 0;
    rocblas_stride offsetB = arg.api == INTERNAL ? arg.stride_y : 0;
    rocblas_stride offsetC = arg.api == INTERNAL ? arg.stride_d : 0;

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta.
    bool invalid_size
        = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_gemm_batched_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     nullptr,
                     ldc,
                     batch_count));
        return;
    }

    double cpu_time_used = 0.0;
    double rocblas_error = 0.0, error_hst_ptr = 0.0, error_dev_ptr = 0.0;

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        DEVICE_MEMCHECK(device_batch_vector<T>, dA, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dB, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dC, (1, 1, batch_count));
        DAPI_CHECK(rocblas_gemm_batched_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    K,
                    &h_alpha,
                    dA.ptr_on_device(),
                    lda,
                    dB.ptr_on_device(),
                    ldb,
                    &h_beta,
                    dC.ptr_on_device(),
                    ldc,
                    batch_count));
        return;
    }
#endif

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<T>, hA, (A_row, A_col, lda, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hB, (B_row, B_col, ldb, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hC, (M, N, ldc, batch_count));
    HOST_MEMCHECK(host_vector<T>, halpha, (1));
    HOST_MEMCHECK(host_vector<T>, hbeta, (1));
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dA, (A_row, A_col, lda, batch_count, false, offsetA));
    DEVICE_MEMCHECK(device_batch_matrix<T>, dB, (B_row, B_col, ldb, batch_count, false, offsetB));
    DEVICE_MEMCHECK(device_batch_matrix<T>, dC, (M, N, ldc, batch_count, false, offsetC));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        HOST_MEMCHECK(host_batch_matrix<T>, hC_gold, (M, N, ldc, batch_count));
        hC_gold.copy_from(hC);

        // ROCBLAS rocblas_pointer_mode_host
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            if(arg.api != INTERNAL)
            {
                DAPI_CHECK(rocblas_gemm_batched_fn,
                           (handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            &h_alpha,
                            dA.ptr_on_device(),
                            lda,
                            dB.ptr_on_device(),
                            ldb,
                            &h_beta,
                            dC.ptr_on_device(),
                            ldc,
                            batch_count));
            }
            else
            {
                constexpr rocblas_stride strideA = 0, strideB = 0, strideC = 0;

                CHECK_ROCBLAS_ERROR(rocblas_internal_gemm_batched_template<T>(handle,
                                                                              transA,
                                                                              transB,
                                                                              M,
                                                                              N,
                                                                              K,
                                                                              &h_alpha,
                                                                              dA.ptr_on_device(),
                                                                              -offsetA,
                                                                              lda,
                                                                              strideA,
                                                                              dB.ptr_on_device(),
                                                                              -offsetB,
                                                                              ldb,
                                                                              strideB,
                                                                              &h_beta,
                                                                              dC.ptr_on_device(),
                                                                              -offsetC,
                                                                              ldc,
                                                                              strideC,
                                                                              batch_count));
            }
            handle.post_test(arg);
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            // ROCBLAS rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

            if(arg.api != INTERNAL)
            {
                DAPI_CHECK(rocblas_gemm_batched_fn,
                           (handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            d_alpha,
                            dA.ptr_on_device(),
                            lda,
                            dB.ptr_on_device(),
                            ldb,
                            d_beta,
                            dC.ptr_on_device(),
                            ldc,
                            batch_count));

                if(arg.repeatability_check)
                {
                    HOST_MEMCHECK(host_batch_matrix<T>, hC_copy, (M, N, ldc, batch_count));
                    CHECK_HIP_ERROR(hC.transfer_from(dC));

                    // multi-GPU support
                    int device_id, device_count;
                    CHECK_HIP_ERROR(limit_device_count(device_count, (int)arg.devices));

                    for(int dev_id = 0; dev_id < device_count; dev_id++)
                    {
                        CHECK_HIP_ERROR(hipGetDevice(&device_id));
                        if(device_id != dev_id)
                            CHECK_HIP_ERROR(hipSetDevice(dev_id));

                        //New rocblas handle for new device
                        rocblas_local_handle handle_copy{arg};

                        //Allocate device memory in new device
                        DEVICE_MEMCHECK(device_batch_matrix<T>,
                                        dA_copy,
                                        (A_row, A_col, lda, batch_count, false, offsetA));
                        DEVICE_MEMCHECK(device_batch_matrix<T>,
                                        dB_copy,
                                        (B_row, B_col, ldb, batch_count, false, offsetB));
                        DEVICE_MEMCHECK(device_batch_matrix<T>,
                                        dC_copy,
                                        (M, N, ldc, batch_count, false, offsetC));
                        DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                        DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                        // copy data from CPU to device
                        CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                        CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                        CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));
                        CHECK_HIP_ERROR(d_beta_copy.transfer_from(hbeta));

                        CHECK_ROCBLAS_ERROR(
                            rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                        for(int runs = 0; runs < arg.iters; runs++)
                        {
                            CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));

                            DAPI_CHECK(rocblas_gemm_batched_fn,
                                       (handle_copy,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        d_alpha_copy,
                                        dA_copy.ptr_on_device(),
                                        lda,
                                        dB_copy.ptr_on_device(),
                                        ldb,
                                        d_beta_copy,
                                        dC_copy.ptr_on_device(),
                                        ldc,
                                        batch_count));

                            CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                            unit_check_general<T>(M, N, ldc, hC, hC_copy, batch_count);
                        }
                    }
                    return;
                }
            }
            else
            {
                // only testing internal for simpler code since using different device_batch_matrix initialization
                constexpr rocblas_stride strideA = 0, strideB = 0, strideC = 0;

                CHECK_ROCBLAS_ERROR(rocblas_internal_gemm_batched_template<T>(handle,
                                                                              transA,
                                                                              transB,
                                                                              M,
                                                                              N,
                                                                              K,
                                                                              d_alpha,
                                                                              dA.ptr_on_device(),
                                                                              -offsetA,
                                                                              lda,
                                                                              strideA,
                                                                              dB.ptr_on_device(),
                                                                              -offsetB,
                                                                              ldb,
                                                                              strideB,
                                                                              d_beta,
                                                                              dC.ptr_on_device(),
                                                                              -offsetC,
                                                                              ldc,
                                                                              strideC,
                                                                              batch_count));
            }
        }

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if(std::is_same<T, float>{} && math_mode == rocblas_xf32_xdl_math_op)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                type_to_xdl_math_op_type<rocblas_xfloat32, float>(hA[b], hA.nmemb());
                type_to_xdl_math_op_type<rocblas_xfloat32, float>(hB[b], hB.nmemb());
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_gemm<T>(
                transA, transB, M, N, K, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_gold[b], ldc);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_to_gold = [&] {
            if(arg.unit_check)
            {
                if(std::is_same_v<T,
                                  rocblas_half> && (rocblas_handle(handle)->getArchMajor() == 11))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<T, T, T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC, batch_count, tol);
                }
                else if(reduction_requires_near<T>(arg, K))
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC, batch_count);
                }
            }

            double error = 0;
            if(arg.norm_check)
            {
                error = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC, batch_count));
            }
            return error;
        };

        // check error and norm
        if(arg.pointer_mode_host)
        {
            error_hst_ptr = compare_to_gold();
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hC.transfer_from(dC));
            error_dev_ptr = compare_to_gold();
        }
        rocblas_error = error_dev_ptr > error_hst_ptr ? error_dev_ptr : error_hst_ptr;
    }

    if(arg.timing && arg.api != INTERNAL)
    {
        double gpu_time_used     = 0.0;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        FrequencyMonitor& freq_monitor = getFrequencyMonitor();
        freq_monitor.start();

        for(int i = 0; i < total_calls; i++)
        {
            if(i == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            DAPI_DISPATCH(rocblas_gemm_batched_fn,
                          (handle,
                           transA,
                           transB,
                           M,
                           N,
                           K,
                           &h_alpha,
                           dA.ptr_on_device(),
                           lda,
                           dB.ptr_on_device(),
                           ldb,
                           &h_beta,
                           dC.ptr_on_device(),
                           ldc,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        freq_monitor.stop();

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_beta,
                      e_ldb,
                      e_ldc,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gemm_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value);
    }
}
