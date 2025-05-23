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
 *
 * ************************************************************************ */

#pragma once

#include "testing_common.hpp"

template <typename T>
void testing_hbmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_hbmv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_hbmv_batched<T, true> : rocblas_hbmv_batched<T, false>;

    auto rocblas_hbmv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_hbmv_batched_64<T, true>
                                                              : rocblas_hbmv_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill uplo        = rocblas_fill_upper;
        const int64_t      N           = 100;
        const int64_t      K           = 5;
        const int64_t      lda         = 100;
        const int64_t      incx        = 1;
        const int64_t      incy        = 1;
        const int64_t      batch_count = 2;

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

        rocblas_int banded_matrix_row = K + 1;

        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_matrix<T>, dAb, (banded_matrix_row, N, lda, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dy, (N, incy, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_hbmv_batched_fn,
                    (nullptr,
                     uplo,
                     N,
                     K,
                     alpha,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     beta,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_hbmv_batched_fn,
                    (handle,
                     rocblas_fill_full,
                     N,
                     K,
                     alpha,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     beta,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_hbmv_batched_fn,
                    (handle,
                     uplo,
                     N,
                     K,
                     nullptr,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     beta,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_hbmv_batched_fn,
                    (handle,
                     uplo,
                     N,
                     K,
                     alpha,
                     dAb.ptr_on_device(),
                     lda,
                     dx.ptr_on_device(),
                     incx,
                     nullptr,
                     dy.ptr_on_device(),
                     incy,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_hbmv_batched_fn,
                        (handle,
                         uplo,
                         N,
                         K,
                         alpha,
                         nullptr,
                         lda,
                         dx.ptr_on_device(),
                         incx,
                         beta,
                         dy.ptr_on_device(),
                         incy,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_hbmv_batched_fn,
                        (handle,
                         uplo,
                         N,
                         K,
                         alpha,
                         dAb.ptr_on_device(),
                         lda,
                         nullptr,
                         incx,
                         beta,
                         dy.ptr_on_device(),
                         incy,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_hbmv_batched_fn,
                        (handle,
                         uplo,
                         N,
                         K,
                         alpha,
                         dAb.ptr_on_device(),
                         lda,
                         dx.ptr_on_device(),
                         incx,
                         beta,
                         nullptr,
                         incy,
                         batch_count));
        }

        // If N==0, then all pointers can be nullptr without error
        DAPI_CHECK(rocblas_hbmv_batched_fn,
                   (handle,
                    uplo,
                    0,
                    K,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    nullptr,
                    nullptr,
                    incy,
                    batch_count));

        // If alpha==0 then A and x may be nullptr without error
        DAPI_CHECK(rocblas_hbmv_batched_fn,
                   (handle,
                    uplo,
                    N,
                    K,
                    zero,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    beta,
                    dy.ptr_on_device(),
                    incy,
                    batch_count));

        // If alpha==0 && beta==1, then A, x and y may be nullptr without error
        DAPI_CHECK(rocblas_hbmv_batched_fn,
                   (handle,
                    uplo,
                    N,
                    K,
                    zero,
                    nullptr,
                    lda,
                    nullptr,
                    incx,
                    one,
                    nullptr,
                    incy,
                    batch_count));

        // If batch_count==0, then all pointers can be nullptr without error
        DAPI_CHECK(
            rocblas_hbmv_batched_fn,
            (handle, uplo, N, K, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy, 0));
    }
}

template <typename T>
void testing_hbmv_batched(const Arguments& arg)
{
    auto rocblas_hbmv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_hbmv_batched<T, true> : rocblas_hbmv_batched<T, false>;

    auto rocblas_hbmv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_hbmv_batched_64<T, true>
                                                              : rocblas_hbmv_batched_64<T, false>;

    int64_t              N                 = arg.N;
    int64_t              K                 = arg.K;
    int64_t              lda               = arg.lda;
    int64_t              incx              = arg.incx;
    int64_t              incy              = arg.incy;
    int64_t              batch_count       = arg.batch_count;
    int64_t              banded_matrix_row = K + 1;
    T                    h_alpha           = arg.get_alpha<T>();
    T                    h_beta            = arg.get_beta<T>();
    rocblas_fill         uplo              = char2rocblas_fill(arg.uplo);
    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size
        = N < 0 || K < 0 || lda < banded_matrix_row || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_hbmv_batched_fn,
                    (handle,
                     uplo,
                     N,
                     K,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     incx,
                     nullptr,
                     nullptr,
                     incy,
                     batch_count));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<T>, hAb, (banded_matrix_row, N, lda, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hy, (N, incy, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hy_gold, (N, incy, batch_count));
    HOST_MEMCHECK(host_vector<T>, halpha, (1));
    HOST_MEMCHECK(host_vector<T>, hbeta, (1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dAb, (banded_matrix_row, N, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dy, (N, incy, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hAb, arg, rocblas_client_alpha_sets_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    hy_gold.copy_from(hy);
    CHECK_HIP_ERROR(dAb.transfer_from(hAb));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;
    double rocblas_error_host   = 0.0;
    double rocblas_error_device = 0.0;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_hbmv_batched_fn,
                       (handle,
                        uplo,
                        N,
                        K,
                        &h_alpha,
                        dAb.ptr_on_device(),
                        lda,
                        dx.ptr_on_device(),
                        incx,
                        &h_beta,
                        dy.ptr_on_device(),
                        incy,
                        batch_count));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

            CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_hbmv_batched_fn,
                       (handle,
                        uplo,
                        N,
                        K,
                        d_alpha,
                        dAb.ptr_on_device(),
                        lda,
                        dx.ptr_on_device(),
                        incx,
                        d_beta,
                        dy.ptr_on_device(),
                        incy,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_batch_vector<T>, hy_copy, (N, incy, batch_count));
                // copy output from device to CPU
                CHECK_HIP_ERROR(hy.transfer_from(dy));

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

                    // Allocate device memory
                    DEVICE_MEMCHECK(
                        device_batch_matrix<T>, dAb_copy, (banded_matrix_row, N, lda, batch_count));
                    DEVICE_MEMCHECK(device_batch_vector<T>, dx_copy, (N, incx, batch_count));
                    DEVICE_MEMCHECK(device_batch_vector<T>, dy_copy, (N, incy, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    CHECK_HIP_ERROR(dAb_copy.transfer_from(hAb));
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(hbeta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_hbmv_batched_fn,
                                   (handle_copy,
                                    uplo,
                                    N,
                                    K,
                                    d_alpha_copy,
                                    dAb_copy.ptr_on_device(),
                                    lda,
                                    dx_copy.ptr_on_device(),
                                    incx,
                                    d_beta_copy,
                                    dy_copy.ptr_on_device(),
                                    incy,
                                    batch_count));
                        // copy output from device to CPU
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));
                        unit_check_general<T>(1, N, incy, hy, hy_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(size_t b = 0; b < batch_count; b++)
            ref_hbmv<T>(uplo, N, K, h_alpha, hAb[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
                unit_check_general<T>(1, N, incy, hy_gold, hy, batch_count);
            if(arg.norm_check)
                rocblas_error_host
                    = norm_check_general<T>('F', 1, N, incy, hy_gold, hy, batch_count);
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
                unit_check_general<T>(1, N, incy, hy_gold, hy, batch_count);
            if(arg.norm_check)
                rocblas_error_device
                    = norm_check_general<T>('F', 1, N, incy, hy_gold, hy, batch_count);
        }
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

            DAPI_DISPATCH(rocblas_hbmv_batched_fn,
                          (handle,
                           uplo,
                           N,
                           K,
                           &h_alpha,
                           dAb.ptr_on_device(),
                           lda,
                           dx.ptr_on_device(),
                           incx,
                           &h_beta,
                           dy.ptr_on_device(),
                           incy,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo, e_N, e_K, e_alpha, e_lda, e_incx, e_beta, e_incy, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         hbmv_gflop_count<T>(N, K),
                         hbmv_gbyte_count<T>(N, K),
                         cpu_time_used,
                         rocblas_error_host,
                         rocblas_error_device);
    }
}
