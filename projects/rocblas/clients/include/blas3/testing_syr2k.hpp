/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_common.hpp"

template <typename T, bool TWOK = true>
void testing_syr2k_bad_arg(const Arguments& arg)
{
    auto rocblas_syrXX_fn
        = TWOK ? (arg.api & c_API_FORTRAN ? rocblas_syr2k<T, true> : rocblas_syr2k<T, false>)
               : (arg.api & c_API_FORTRAN ? rocblas_syrkx<T, true> : rocblas_syrkx<T, false>);

    auto rocblas_syrXX_fn_64
        = TWOK ? (arg.api & c_API_FORTRAN ? rocblas_syr2k_64<T, true> : rocblas_syr2k_64<T, false>)
               : (arg.api & c_API_FORTRAN ? rocblas_syrkx_64<T, true> : rocblas_syrkx_64<T, false>);

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const int64_t           N      = 100;
        const int64_t           K      = 100;
        const int64_t           lda    = 100;
        const int64_t           ldb    = 100;
        const int64_t           ldc    = 100;

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

        size_t rows = (transA != rocblas_operation_none ? std::max(K, int64_t(1)) : N);
        size_t cols = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (rows, cols, lda));
        DEVICE_MEMCHECK(device_matrix<T>, dB, (rows, cols, ldb));
        DEVICE_MEMCHECK(device_matrix<T>, dC, (N, N, ldc));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_syrXX_fn,
                    (nullptr, uplo, transA, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc));

        // invalid values
        DAPI_EXPECT(
            rocblas_status_invalid_value,
            rocblas_syrXX_fn,
            (handle, rocblas_fill_full, transA, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_syrXX_fn,
                    (handle,
                     uplo,
                     (rocblas_operation)rocblas_fill_full,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc));

        if(rocblas_is_complex<T>)
        {
            DAPI_EXPECT(rocblas_status_invalid_value,
                        rocblas_syrXX_fn,
                        (handle,
                         uplo,
                         rocblas_operation_conjugate_transpose,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         dB,
                         ldb,
                         beta,
                         dC,
                         ldc));
        }

        // alpha/beta pointer checks
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_syrXX_fn,
                    (handle, uplo, transA, N, K, nullptr, dA, lda, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_syrXX_fn,
                    (handle, uplo, transA, N, K, alpha, dA, lda, dB, ldb, nullptr, dC, ldc));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            // alpha and beta can only be inspected in host_mode so A and B validated
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syrXX_fn,
                        (handle, uplo, transA, N, K, alpha, nullptr, lda, dB, ldb, beta, dC, ldc));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syrXX_fn,
                        (handle, uplo, transA, N, K, alpha, dA, lda, nullptr, ldb, beta, dC, ldc));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syrXX_fn,
                        (handle, uplo, transA, N, K, alpha, dA, lda, dB, ldb, beta, nullptr, ldc));
        }

        // size
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_syrXX_fn,
                    (handle, uplo, transA, N, K, alpha, dA, lda - 1, dB, ldb, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_syrXX_fn,
                    (handle, uplo, transA, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc - 1));

        // N==0 quick return for no ops with null pointers
        DAPI_CHECK(
            rocblas_syrXX_fn,
            (handle, uplo, transA, 0, K, nullptr, nullptr, lda, dB, ldb, nullptr, nullptr, ldc));

        // k==0 and beta==1 all A, B, C pointers may be null
        DAPI_CHECK(
            rocblas_syrXX_fn,
            (handle, uplo, transA, N, 0, alpha, nullptr, lda, nullptr, ldb, one, nullptr, ldc));

        // alpha==0 and beta==1 all pointers may be null
        DAPI_CHECK(
            rocblas_syrXX_fn,
            (handle, uplo, transA, N, K, zero, nullptr, lda, nullptr, ldb, one, nullptr, ldc));
    }
}

template <typename T, bool TWOK = true>
void testing_syr2k(const Arguments& arg)
{
    auto rocblas_syrXX_fn
        = TWOK ? (arg.api & c_API_FORTRAN ? rocblas_syr2k<T, true> : rocblas_syr2k<T, false>)
               : (arg.api & c_API_FORTRAN ? rocblas_syrkx<T, true> : rocblas_syrkx<T, false>);

    auto rocblas_syrXX_fn_64
        = TWOK ? (arg.api & c_API_FORTRAN ? rocblas_syr2k_64<T, true> : rocblas_syr2k_64<T, false>)
               : (arg.api & c_API_FORTRAN ? rocblas_syrkx_64<T, true> : rocblas_syrkx_64<T, false>);

    auto syrXX_gflop_count_fn = TWOK ? syr2k_gflop_count<T> : syrkx_gflop_count<T>;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    int64_t              N      = arg.N;
    int64_t              K      = arg.K;
    int64_t              lda    = arg.lda;
    int64_t              ldb    = arg.ldb;
    int64_t              ldc    = arg.ldc;
    T                    alpha  = arg.get_alpha<T>();
    T                    beta   = arg.get_beta<T>();

    double cpu_time_used;
    double error_host   = 0.0;
    double error_device = 0.0;

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = N < 0 || K < 0 || ldc < N
                        || (transA == rocblas_operation_none && (lda < N || ldb < N))
                        || (transA != rocblas_operation_none && (lda < K || ldb < K));
    if(N == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_syrXX_fn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     nullptr,
                     ldc));

        return;
    }

    size_t rows = (transA != rocblas_operation_none ? std::max(K, int64_t(1)) : N);
    size_t cols = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (rows, cols, lda));
    HOST_MEMCHECK(host_matrix<T>, hB, (rows, cols, ldb));
    HOST_MEMCHECK(host_matrix<T>, hC, (N, N, ldc));
    HOST_MEMCHECK(host_matrix<T>, hC_gold, (N, N, ldc));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<T>, h_beta, (1));

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (rows, cols, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dB, (rows, cols, ldb));
    DEVICE_MEMCHECK(device_matrix<T>, dC, (N, N, ldc));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true, false);
    rocblas_init_matrix(hC, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix);

    if(TWOK)
    {
        rocblas_init_matrix(
            hB, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, false, true);
    }
    else
    { // using syrk as syrkx reference so testing with B = A
        rocblas_copy_matrix((T*)hA, (T*)hB, rows, cols, lda, ldb);
    }

    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // host alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dC.transfer_from(hC));
            handle.pre_test(arg);
            DAPI_CHECK(
                rocblas_syrXX_fn,
                (handle, uplo, transA, N, K, &h_alpha[0], dA, lda, dB, ldb, &h_beta[0], dC, ldc));

            handle.post_test(arg);
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            // device alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

            DAPI_CHECK(rocblas_syrXX_fn,
                       (handle, uplo, transA, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_matrix<T>, hC_copy, (N, N, ldc));
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
                    DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (rows, cols, lda));
                    DEVICE_MEMCHECK(device_matrix<T>, dB_copy, (rows, cols, ldb));
                    DEVICE_MEMCHECK(device_matrix<T>, dC_copy, (N, N, ldc));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(h_beta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));
                        DAPI_CHECK(rocblas_syrXX_fn,
                                   (handle_copy,
                                    uplo,
                                    transA,
                                    N,
                                    K,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    dB_copy,
                                    ldb,
                                    d_beta_copy,
                                    dC_copy,
                                    ldc));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(N, N, ldc, hC, hC_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        if(TWOK)
        {
            ref_syr2k<T>(uplo, transA, N, K, h_alpha[0], hA, lda, hB, ldb, h_beta[0], hC_gold, ldc);
        }
        else
        { // syrkx
            ref_syrk<T>(uplo,
                        transA,
                        N,
                        K,
                        h_alpha[0],
                        hA,
                        lda,
                        h_beta[0],
                        hC_gold,
                        ldc); // B must == A to use syrk as reference
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                bool use_near = reduction_requires_near<T>(arg, K);
                if(use_near)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, tol);
                }
                else if(std::is_same_v<
                            T,
                            rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, hC_gold, hC);
                }
            }

            if(arg.norm_check)
            {
                error_host = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC));
            }
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));

            if(arg.unit_check)
            {
                bool use_near = reduction_requires_near<T>(arg, K);
                if(use_near)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, tol);
                }
                else if(std::is_same_v<
                            T,
                            rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, hC_gold, hC);
                }
            }

            if(arg.norm_check)
            {
                error_device = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC));
            }
        }
    }
    else
    {
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_syrXX_fn,
                          (handle, uplo, transA, N, K, h_alpha, dA, lda, dB, ldb, h_beta, dC, ldc));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        double gflops = syrXX_gflop_count_fn(N, K);
        ArgumentModel<e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_ldb, e_beta, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gflops,
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
