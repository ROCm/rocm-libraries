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

/* ============================================================================================ */
template <typename T>
void testing_gemm_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gemm_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_strided_batched<T, true>
                                                  : rocblas_gemm_strided_batched<T, false>;
    auto rocblas_gemm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_strided_batched_64<T, true>
                                                  : rocblas_gemm_strided_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const int64_t M = 100;
        const int64_t N = 101;
        const int64_t K = 102;

        const int64_t lda = 103;
        const int64_t ldb = 103;
        const int64_t ldc = 103;

        const int64_t batch_count = 1;

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, one_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        int64_t A_row = transA == rocblas_operation_none ? M : std::max(K, int64_t(1));
        int64_t A_col = transA == rocblas_operation_none ? std::max(K, int64_t(1)) : M;
        int64_t B_row = transB == rocblas_operation_none ? std::max(K, int64_t(1)) : N;
        int64_t B_col = transB == rocblas_operation_none ? N : std::max(K, int64_t(1));

        const int64_t stride_a = lda * A_col;
        const int64_t stride_b = ldb * B_col;
        const int64_t stride_c = ldc * N;

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

        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t           solution_index = 0;
        int64_t           flags          = 0;

        const size_t safe_size = stride_c;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Allocate device memory
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dA, (A_row, A_col, lda, stride_a, batch_count));
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dB, (B_row, B_col, ldb, stride_b, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (M, N, ldc, stride_c, batch_count));

        // clang-format off

// check for valid enum
DAPI_EXPECT( rocblas_status_invalid_value,rocblas_gemm_strided_batched_fn, (handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_value,rocblas_gemm_strided_batched_fn, (handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

// check for invalid size
DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, transA, transB, -1, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT(rocblas_status_invalid_size ,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, -1, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, -1, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, -1));

// check for invalid leading dimension
DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA, M-1, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, rocblas_operation_none, rocblas_operation_none, M, N, K, alpha,
dA, lda, stride_a, dB, K-1, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, alpha,
dA, K-1, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT(rocblas_status_invalid_size ,rocblas_gemm_strided_batched_fn, (handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, alpha,
dA, lda, stride_a, dB, N-1, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_size,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, M-1, stride_c, batch_count));

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
DAPI_EXPECT(rocblas_status_invalid_handle ,rocblas_gemm_strided_batched_fn, (nullptr, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, nullptr,
dA, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, alpha,
nullptr, lda, stride_a, dB, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT(rocblas_status_invalid_pointer ,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, nullptr, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, nullptr, dC, ldc, stride_c, batch_count));

DAPI_EXPECT( rocblas_status_invalid_pointer,rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, alpha,
dA, lda, stride_a, dB, ldb, stride_b, beta, nullptr, ldc, stride_c, batch_count));

// If batch_count==0, then all pointers can be nullptr without issue
DAPI_CHECK( rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, nullptr, nullptr, ldc, stride_c, 0));

// If M==0, then all pointers can be nullptr without issue
DAPI_CHECK( rocblas_gemm_strided_batched_fn, (handle, transA, transB, 0, N, K, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, nullptr, nullptr, ldc, stride_c, batch_count));

// If N==0, then all pointers can be nullptr without issue
DAPI_CHECK( rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, 0, K, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, nullptr, nullptr, ldc, stride_c, batch_count));

// If alpha==0 and beta==1, then A, B and C can be nullptr without issue
DAPI_CHECK( rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, zero,
nullptr, lda, stride_a, nullptr, ldb, stride_b, one, nullptr, ldc, stride_c, batch_count));

// the following tests still output to C

// If K==0, then alpha, A and B can both be nullptr without issue.
DAPI_CHECK( rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, 0, nullptr,
nullptr, lda, stride_a, nullptr, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

// If alpha==0, then A and B can both be nullptr without issue.
DAPI_CHECK( rocblas_gemm_strided_batched_fn, (handle, transA, transB, M, N, K, zero,
nullptr, lda, stride_a, nullptr, ldb, stride_b, beta, dC, ldc, stride_c, batch_count));

        // clang-format on
    }
}

template <typename T>
void testing_gemm_strided_batched(const Arguments& arg)
{
    auto rocblas_gemm_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_strided_batched<T, true>
                                                  : rocblas_gemm_strided_batched<T, false>;
    auto rocblas_gemm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_strided_batched_64<T, true>
                                                  : rocblas_gemm_strided_batched_64<T, false>;

    rocblas_local_handle handle{arg};

    int64_t           M           = arg.M;
    int64_t           N           = arg.N;
    int64_t           K           = arg.K;
    T                 h_alpha     = arg.get_alpha<T>();
    T                 h_beta      = arg.get_beta<T>();
    int64_t           lda         = arg.lda;
    int64_t           ldb         = arg.ldb;
    int64_t           ldc         = arg.ldc;
    int64_t           stride_a    = arg.stride_a;
    int64_t           stride_b    = arg.stride_b;
    int64_t           stride_c    = arg.stride_c;
    int64_t           batch_count = arg.batch_count;
    rocblas_operation transA      = char2rocblas_operation(arg.transA);
    rocblas_operation transB      = char2rocblas_operation(arg.transB);

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, math_mode));
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    int64_t A_row = transA == rocblas_operation_none ? M : std::max(K, int64_t(1));
    int64_t A_col = transA == rocblas_operation_none ? std::max(K, int64_t(1)) : M;
    int64_t B_row = transB == rocblas_operation_none ? std::max(K, int64_t(1)) : N;
    int64_t B_col = transB == rocblas_operation_none ? N : std::max(K, int64_t(1));

    // check here to prevent undefined memory allocation error
    // Note: K==0 is not an early exit, since C must still be multiplied by beta
    bool invalid_size
        = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_gemm_strided_batched_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     stride_a,
                     nullptr,
                     ldb,
                     stride_b,
                     nullptr,
                     nullptr,
                     ldc,
                     stride_c,
                     batch_count));
        return;
    }

    double cpu_time_used = 0.0;
    double rocblas_error = 0.0, error_hst_ptr = 0.0, error_dev_ptr = 0.0;

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        DEVICE_MEMCHECK(device_vector<T>, dA, (1));
        DEVICE_MEMCHECK(device_vector<T>, dB, (1));
        DEVICE_MEMCHECK(device_vector<T>, dC, (1));
        DAPI_CHECK(rocblas_gemm_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    K,
                    &h_alpha,
                    dA,
                    lda,
                    stride_a,
                    dB,
                    ldb,
                    stride_b,
                    &h_beta,
                    dC,
                    ldc,
                    stride_c,
                    batch_count));
        return;
    }
#endif

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (A_row, A_col, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hB, (B_row, B_col, ldb, stride_b, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC, (M, N, ldc, stride_c, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (A_row, A_col, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dB, (B_row, B_col, ldb, stride_b, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (M, N, ldc, stride_c, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        HOST_MEMCHECK(host_strided_batch_matrix<T>, hC_gold, (M, N, ldc, stride_c, batch_count));

        // copy data from CPU to device
        hC_gold.copy_from(hC);

        // ROCBLAS rocblas_pointer_mode_host
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            if(arg.api != INTERNAL)
            {
                DAPI_CHECK(rocblas_gemm_strided_batched_fn,
                           (handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            &h_alpha,
                            dA,
                            lda,
                            stride_a,
                            dB,
                            ldb,
                            stride_b,
                            &h_beta,
                            dC,
                            ldc,
                            stride_c,
                            batch_count));
            }
            else
            {
                // using arg.stride_x,y,d for offset testing
                rocblas_stride offsetA = arg.stride_x;
                rocblas_stride offsetB = arg.stride_y;
                rocblas_stride offsetC = arg.stride_d;

                CHECK_ROCBLAS_ERROR(rocblas_internal_gemm_template<T>(handle,
                                                                      transA,
                                                                      transB,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      &h_alpha,
                                                                      (const T*)dA + offsetA,
                                                                      -offsetA,
                                                                      lda,
                                                                      stride_a,
                                                                      (const T*)dB + offsetB,
                                                                      -offsetB,
                                                                      ldb,
                                                                      stride_b,
                                                                      &h_beta,
                                                                      (T*)dC + offsetC,
                                                                      -offsetC,
                                                                      ldc,
                                                                      stride_c,
                                                                      batch_count));
            }
            handle.post_test(arg);

            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            // ROCBLAS rocblas_pointer_mode_device
            // don't need to check internal again
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

            DAPI_CHECK(rocblas_gemm_strided_batched_fn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        K,
                        d_alpha,
                        dA,
                        lda,
                        stride_a,
                        dB,
                        ldb,
                        stride_b,
                        d_beta,
                        dC,
                        ldc,
                        stride_c,
                        batch_count));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(
                    host_strided_batch_matrix<T>, hC_copy, (M, N, ldc, stride_c, batch_count));
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
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dA_copy,
                                    (A_row, A_col, lda, stride_a, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dB_copy,
                                    (B_row, B_col, ldb, stride_b, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dC_copy,
                                    (M, N, ldc, stride_c, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(
                        hipMemcpy(d_alpha_copy, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(d_beta_copy, &h_beta, sizeof(T), hipMemcpyHostToDevice));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));

                        DAPI_CHECK(rocblas_gemm_strided_batched_fn,
                                   (handle_copy,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    stride_a,
                                    dB_copy,
                                    ldb,
                                    stride_b,
                                    d_beta,
                                    dC_copy,
                                    ldc,
                                    stride_c,
                                    batch_count));

                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(M, N, ldc, stride_c, hC, hC_copy, batch_count);
                    }
                }
                return;
            }
        }

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if(std::is_same<T, float>{} && math_mode == rocblas_xf32_xdl_math_op)
        {
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hA.data(), hA.nmemb());
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hB.data(), hB.nmemb());
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
                    near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count, tol);
                }
                else if(reduction_requires_near<T>(arg, K))
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
                }
            }
            double error = 0;
            if(arg.norm_check)
            {
                error = std::abs(
                    norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count));
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
            // fetch GPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
            error_dev_ptr = compare_to_gold();
        }
        rocblas_error = error_hst_ptr > error_dev_ptr ? error_hst_ptr : error_dev_ptr;
    }

    if(arg.timing)
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

            DAPI_DISPATCH(rocblas_gemm_strided_batched_fn,
                          (handle,
                           transA,
                           transB,
                           M,
                           N,
                           K,
                           &h_alpha,
                           dA,
                           lda,
                           stride_a,
                           dB,
                           ldb,
                           stride_b,
                           &h_beta,
                           dC,
                           ldc,
                           stride_c,
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
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
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
