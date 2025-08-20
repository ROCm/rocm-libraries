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

template <typename T>
void testing_herk_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_herk_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_herk_strided_batched<T, real_t<T>, true>
                                               : rocblas_herk_strided_batched<T, real_t<T>, false>;
    auto rocblas_herk_strided_batched_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_herk_strided_batched_64<T, real_t<T>, true>
                                  : rocblas_herk_strided_batched_64<T, real_t<T>, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill      uplo        = rocblas_fill_upper;
        const rocblas_operation transA      = rocblas_operation_none;
        const int64_t           N           = 100;
        const int64_t           K           = 99;
        const int64_t           lda         = 100;
        const int64_t           ldc         = 100;
        rocblas_stride          strideA     = 1;
        rocblas_stride          strideC     = 1;
        int64_t                 batch_count = 2;

        using U = real_t<T>;
        DEVICE_MEMCHECK(device_vector<U>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<U>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<U>, one_d, (1));
        DEVICE_MEMCHECK(device_vector<U>, zero_d, (1));

        const U alpha_h(1), zero_h(0);
        const U beta_h(2), one_h(1);

        const U* alpha = &alpha_h;
        const U* zero  = &zero_h;
        const U* beta  = &beta_h;
        const U* one   = &one_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
        }

        size_t cols = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);
        size_t rows = (transA != rocblas_operation_none ? std::max(K, int64_t(1)) : N);

        // Allocate device memory
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dA, (rows, cols, lda, strideA, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (N, N, ldc, strideC, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_herk_strided_batched_fn,
                    (nullptr,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        // values
        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     rocblas_fill_full,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     (rocblas_operation)rocblas_fill_full,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     rocblas_operation_transpose,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        // size
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda - 1,
                     strideA,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     beta,
                     dC,
                     ldc - 1,
                     strideC,
                     batch_count));

        // alpha/beta
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     dA,
                     lda,
                     strideA,
                     beta,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     nullptr,
                     dC,
                     ldc,
                     strideC,
                     batch_count));

        // invalid pointers
        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_herk_strided_batched_fn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         nullptr,
                         lda,
                         strideA,
                         beta,
                         dC,
                         ldc,
                         strideC,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_herk_strided_batched_fn,
                        (handle,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         strideA,
                         beta,
                         nullptr,
                         ldc,
                         strideC,
                         batch_count));
        }

        // batch_count==0 quick return for no ops with null pointers
        DAPI_CHECK(rocblas_herk_strided_batched_fn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    nullptr,
                    ldc,
                    strideC,
                    0));

        // N==0 quick return for no ops with null pointers
        DAPI_CHECK(rocblas_herk_strided_batched_fn,
                   (handle,
                    uplo,
                    transA,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    nullptr,
                    nullptr,
                    ldc,
                    strideC,
                    batch_count));

        // k==0 and beta==1 quick return with null pointers
        DAPI_CHECK(rocblas_herk_strided_batched_fn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    strideA,
                    one,
                    nullptr,
                    ldc,
                    strideC,
                    batch_count));

        // alpha==0 and beta==1 quick return with null pointers
        DAPI_CHECK(rocblas_herk_strided_batched_fn,
                   (handle,
                    uplo,
                    transA,
                    N,
                    K,
                    zero,
                    nullptr,
                    lda,
                    strideA,
                    one,
                    nullptr,
                    ldc,
                    strideC,
                    batch_count));
    }
}

template <typename T>
void testing_herk_strided_batched(const Arguments& arg)
{
    auto rocblas_herk_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_herk_strided_batched<T, real_t<T>, true>
                                               : rocblas_herk_strided_batched<T, real_t<T>, false>;
    auto rocblas_herk_strided_batched_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_herk_strided_batched_64<T, real_t<T>, true>
                                  : rocblas_herk_strided_batched_64<T, real_t<T>, false>;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    int64_t              N      = arg.N;
    int64_t              K      = arg.K;
    int64_t              lda    = arg.lda;
    int64_t              ldc    = arg.ldc;
    using U                     = real_t<T>;
    U              alpha        = arg.get_alpha<U>();
    U              beta         = arg.get_beta<U>();
    rocblas_stride strideA      = arg.stride_a;
    rocblas_stride strideC      = arg.stride_c;
    int64_t        batch_count  = arg.batch_count;

    double cpu_time_used;
    double error_host   = 0.0;
    double error_device = 0.0;

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == rocblas_operation_none && lda < N)
                        || (transA != rocblas_operation_none && lda < K) || batch_count < 0;
    if(N == 0 || batch_count == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_herk_strided_batched_fn,
                    (handle,
                     uplo,
                     transA,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     nullptr,
                     ldc,
                     strideC,
                     batch_count));

        return;
    }

    size_t cols = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);
    size_t rows = (transA != rocblas_operation_none ? std::max(K, int64_t(1)) : N);

    strideA = std::max(strideA, rocblas_stride(size_t(lda) * cols));
    strideC = std::max(strideC, rocblas_stride(size_t(ldc) * N));

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (rows, cols, lda, strideA, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC, (N, N, ldc, strideC, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC_gold, (N, N, ldc, strideC, batch_count));
    HOST_MEMCHECK(host_vector<U>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<U>, h_beta, (1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (rows, cols, lda, strideA, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (N, N, ldc, strideC, batch_count));
    DEVICE_MEMCHECK(device_vector<U>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<U>, d_beta, (1));

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true, true);
    rocblas_init_matrix(
        hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix, false, true);

    hC_gold.copy_from(hC);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // host alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dC.transfer_from(hC));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_herk_strided_batched_fn,
                       (handle,
                        uplo,
                        transA,
                        N,
                        K,
                        &h_alpha[0],
                        dA,
                        lda,
                        strideA,
                        &h_beta[0],
                        dC,
                        ldc,
                        strideC,
                        batch_count));
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

            DAPI_CHECK(rocblas_herk_strided_batched_fn,
                       (handle,
                        uplo,
                        transA,
                        N,
                        K,
                        d_alpha,
                        dA,
                        lda,
                        strideA,
                        d_beta,
                        dC,
                        ldc,
                        strideC,
                        batch_count));
            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(
                    host_strided_batch_matrix<T>, hC_copy, (N, N, ldc, strideC, batch_count));

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
                                    (rows, cols, lda, strideA, batch_count));
                    DEVICE_MEMCHECK(
                        device_strided_batch_matrix<T>, dC_copy, (N, N, ldc, strideC, batch_count));
                    DEVICE_MEMCHECK(device_vector<U>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<U>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(h_beta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));
                        DAPI_CHECK(rocblas_herk_strided_batched_fn,
                                   (handle_copy,
                                    uplo,
                                    transA,
                                    N,
                                    K,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    strideA,
                                    d_beta_copy,
                                    dC_copy,
                                    ldc,
                                    strideC,
                                    batch_count));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(N, N, ldc, strideC, hC, hC_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        // cpu reference
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_herk<T>(uplo, transA, N, K, h_alpha[0], hA[b], lda, h_beta[0], hC_gold[b], ldc);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_hC_to_gold = [&] {
            if(arg.unit_check)
            {
                if(std::is_same_v<
                       T,
                       rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, strideC, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, strideC, hC_gold, hC, batch_count);
                }
            }

            double error = 0;
            if(arg.norm_check)
            {
                error = std::abs(
                    norm_check_general<T>('F', N, N, ldc, strideC, hC_gold, hC, batch_count));
            }
            return error;
        };

        if(arg.pointer_mode_host)
        {
            error_host = compare_hC_to_gold();
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
            error_device = compare_hC_to_gold();
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

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_herk_strided_batched_fn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           h_alpha,
                           dA,
                           lda,
                           strideA,
                           h_beta,
                           dC,
                           ldc,
                           strideC,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        Arguments targ(arg);
        targ.stride_a = strideA;
        targ.stride_c = strideC;
        ArgumentModel<e_uplo,
                      e_transA,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         targ,
                         gpu_time_used,
                         herk_gflop_count<T>(N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
