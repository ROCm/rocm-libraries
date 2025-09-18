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
void testing_gbmv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gbmv_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_gbmv_strided_batched<T, true>
                                                  : rocblas_gbmv_strided_batched<T, false>;
    auto rocblas_gbmv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_gbmv_strided_batched_64<T, true>
                                                  : rocblas_gbmv_strided_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_operation transA      = rocblas_operation_none;
        const int64_t           M           = 100;
        const int64_t           N           = 100;
        const int64_t           KL          = 5;
        const int64_t           KU          = 5;
        const int64_t           lda         = 100;
        const int64_t           incx        = 1;
        const int64_t           incy        = 1;
        const rocblas_stride    stride_A    = 10000;
        const rocblas_stride    stride_x    = 100;
        const rocblas_stride    stride_y    = 100;
        const int64_t           batch_count = 2;

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

        const int64_t banded_matrix_row = KL + KU + 1;

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                        dAb,
                        (banded_matrix_row, N, lda, stride_A, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (M, incy, stride_y, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_gbmv_strided_batched_fn,
                    (nullptr,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     alpha,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_gbmv_strided_batched_fn,
                    (handle,
                     (rocblas_operation)rocblas_fill_full,
                     M,
                     N,
                     KL,
                     KU,
                     alpha,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gbmv_strided_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     nullptr,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     beta,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_gbmv_strided_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     alpha,
                     dAb,
                     lda,
                     stride_A,
                     dx,
                     incx,
                     stride_x,
                     nullptr,
                     dy,
                     incy,
                     stride_y,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gbmv_strided_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         KL,
                         KU,
                         alpha,
                         nullptr,
                         lda,
                         stride_A,
                         dx,
                         incx,
                         stride_x,
                         beta,
                         dy,
                         incy,
                         stride_y,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gbmv_strided_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         KL,
                         KU,
                         alpha,
                         dAb,
                         lda,
                         stride_A,
                         nullptr,
                         incx,
                         stride_x,
                         beta,
                         dy,
                         incy,
                         stride_y,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_gbmv_strided_batched_fn,
                        (handle,
                         transA,
                         M,
                         N,
                         KL,
                         KU,
                         alpha,
                         dAb,
                         lda,
                         stride_A,
                         dx,
                         incx,
                         stride_x,
                         beta,
                         nullptr,
                         incy,
                         stride_y,
                         batch_count));
        }

        // If M==0, then all pointers may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                   (handle,
                    transA,
                    0,
                    N,
                    KL,
                    KU,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));

        // If N==0, then all pointers may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    0,
                    KL,
                    KU,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));

        // If alpha==0, then A and X may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    KL,
                    KU,
                    zero,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    beta,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        // If alpha==0 && beta==1, then A, X and Y may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    KL,
                    KU,
                    zero,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    one,
                    nullptr,
                    incy,
                    stride_y,
                    batch_count));

        // If batch_count == 0, then all pointers may be nullptr without error
        DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                   (handle,
                    transA,
                    M,
                    N,
                    KL,
                    KU,
                    nullptr,
                    nullptr,
                    lda,
                    stride_A,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    nullptr,
                    incy,
                    stride_y,
                    0));
    }
}

template <typename T>
void testing_gbmv_strided_batched(const Arguments& arg)
{
    auto rocblas_gbmv_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_gbmv_strided_batched<T, true>
                                                  : rocblas_gbmv_strided_batched<T, false>;
    auto rocblas_gbmv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_gbmv_strided_batched_64<T, true>
                                                  : rocblas_gbmv_strided_batched_64<T, false>;

    int64_t           M                 = arg.M;
    int64_t           N                 = arg.N;
    int64_t           KL                = arg.KL;
    int64_t           KU                = arg.KU;
    int64_t           lda               = arg.lda;
    int64_t           incx              = arg.incx;
    int64_t           incy              = arg.incy;
    T                 h_alpha           = arg.get_alpha<T>();
    T                 h_beta            = arg.get_beta<T>();
    rocblas_operation transA            = char2rocblas_operation(arg.transA);
    rocblas_stride    stride_A          = arg.stride_a;
    rocblas_stride    stride_x          = arg.stride_x;
    rocblas_stride    stride_y          = arg.stride_y;
    int64_t           batch_count       = arg.batch_count;
    int64_t           banded_matrix_row = KL + KU + 1;

    rocblas_local_handle handle{arg};
    size_t               dim_x;
    size_t               dim_y;

    if(transA == rocblas_operation_none)
    {
        dim_x = N;
        dim_y = M;
    }
    else
    {
        dim_x = M;
        dim_y = N;
    }

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < banded_matrix_row || !incx || !incy || KL < 0
                        || KU < 0 || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_gbmv_strided_batched_fn,
                    (handle,
                     transA,
                     M,
                     N,
                     KL,
                     KU,
                     nullptr,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     nullptr,
                     incy,
                     stride_y,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    HOST_MEMCHECK(
        host_strided_batch_matrix<T>, hAb, (banded_matrix_row, N, lda, stride_A, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (dim_x, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy, (dim_y, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy_gold, (dim_y, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_vector<T>, halpha, (1));
    HOST_MEMCHECK(host_vector<T>, hbeta, (1));
    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    // Allocate device memory
    DEVICE_MEMCHECK(
        device_strided_batch_matrix<T>, dAb, (banded_matrix_row, N, lda, stride_A, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (dim_x, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (dim_y, incy, stride_y, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hAb, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_gold = hy: save a copy in hy_gold which will be output of
    // CPU BLAS
    hy_gold.copy_from(hy);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAb.transfer_from(hAb));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
    CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));

    double gpu_time_used, cpu_time_used;
    double error_host = 0.0, error_device = 0.0;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                       (handle,
                        transA,
                        M,
                        N,
                        KL,
                        KU,
                        &h_alpha,
                        dAb,
                        lda,
                        stride_A,
                        dx,
                        incx,
                        stride_x,
                        &h_beta,
                        dy,
                        incy,
                        stride_y,
                        batch_count));
            handle.post_test(arg);
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                       (handle,
                        transA,
                        M,
                        N,
                        KL,
                        KU,
                        d_alpha,
                        dAb,
                        lda,
                        stride_A,
                        dx,
                        incx,
                        stride_x,
                        d_beta,
                        dy,
                        incy,
                        stride_y,
                        batch_count));
            handle.post_test(arg);
            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(
                    host_strided_batch_vector<T>, hy_copy, (dim_y, incy, stride_y, batch_count));
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
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dAb_copy,
                                    (banded_matrix_row, N, lda, stride_A, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_vector<T>,
                                    dx_copy,
                                    (dim_x, incx, stride_x, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_vector<T>,
                                    dy_copy,
                                    (dim_y, incy, stride_y, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    CHECK_HIP_ERROR(dAb_copy.transfer_from(hAb));
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(dy_copy.transfer_from(hy));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(hbeta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {

                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_gbmv_strided_batched_fn,
                                   (handle_copy,
                                    transA,
                                    M,
                                    N,
                                    KL,
                                    KU,
                                    d_alpha_copy,
                                    dAb_copy,
                                    lda,
                                    stride_A,
                                    dx_copy,
                                    incx,
                                    stride_x,
                                    d_beta_copy,
                                    dy_copy,
                                    incy,
                                    stride_y,
                                    batch_count));
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));
                        unit_check_general<T>(1, dim_y, incy, stride_y, hy, hy_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int64_t b = 0; b < batch_count; ++b)
        {
            ref_gbmv<T>(
                transA, M, N, KL, KU, h_alpha, hAb[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, dim_y, incy, stride_y, hy_gold, hy, batch_count);
            }

            if(arg.norm_check)
            {
                error_host = norm_check_general<T>(
                    'F', 1, dim_y, incy, stride_y, hy_gold, hy, batch_count);
            }
        }
        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, dim_y, incy, stride_y, hy_gold, hy, batch_count);
            }

            if(arg.norm_check)
            {
                error_device = norm_check_general<T>(
                    'F', 1, dim_y, incy, stride_y, hy_gold, hy, batch_count);
            }
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
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            DAPI_DISPATCH(rocblas_gbmv_strided_batched_fn,
                          (handle,
                           transA,
                           M,
                           N,
                           KL,
                           KU,
                           &h_alpha,
                           dAb,
                           lda,
                           stride_A,
                           dx,
                           incx,
                           stride_x,
                           &h_beta,
                           dy,
                           incy,
                           stride_y,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_M,
                      e_N,
                      e_KL,
                      e_KU,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_beta,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gbmv_gflop_count<T>(transA, M, N, KL, KU),
                         gbmv_gbyte_count<T>(transA, M, N, KL, KU),
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
