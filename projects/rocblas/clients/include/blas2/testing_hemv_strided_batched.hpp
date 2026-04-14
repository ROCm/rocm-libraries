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
void testing_hemv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_hemv_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_hemv_strided_batched<T, true>
                                                  : rocblas_hemv_strided_batched<T, false>;
    auto rocblas_hemv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_hemv_strided_batched_64<T, true>
                                                  : rocblas_hemv_strided_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill   uplo        = rocblas_fill_upper;
        const int64_t        N           = 100;
        const int64_t        lda         = 100;
        const int64_t        incx        = 1;
        const int64_t        incy        = 1;
        const int64_t        batch_count = 2;
        const rocblas_stride stride_A    = 10000;
        const rocblas_stride stride_x    = 100;
        const rocblas_stride stride_y    = 100;

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

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (N, N, lda, stride_A, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stride_y, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_hemv_strided_batched_fn,
                    (nullptr,
                     uplo,
                     N,
                     alpha,
                     dA,
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
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     rocblas_fill_full,
                     N,
                     alpha,
                     dA,
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
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     dA,
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
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     alpha,
                     dA,
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
                        rocblas_hemv_strided_batched_fn,
                        (handle,
                         uplo,
                         N,
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
                        rocblas_hemv_strided_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dA,
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
                        rocblas_hemv_strided_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dA,
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

        // If N==0, then all pointers may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     0,
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

        // If alpha==0 then A and X may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
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
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
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

        // If batch_count==0, then all pointers may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     alpha,
                     nullptr,
                     lda,
                     stride_A,
                     nullptr,
                     incx,
                     stride_x,
                     beta,
                     nullptr,
                     incy,
                     stride_y,
                     0));

        if(arg.api & c_API_64)
        {
            int64_t n_large = 2147483649;
            DAPI_EXPECT(rocblas_status_invalid_size,
                        rocblas_hemv_strided_batched_fn,
                        (handle,
                         uplo,
                         n_large,
                         alpha,
                         dA,
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
        }
    }
}

template <typename T>
void testing_hemv_strided_batched(const Arguments& arg)
{
    auto rocblas_hemv_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_hemv_strided_batched<T, true>
                                                  : rocblas_hemv_strided_batched<T, false>;
    auto rocblas_hemv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_hemv_strided_batched_64<T, true>
                                                  : rocblas_hemv_strided_batched_64<T, false>;

    int64_t        N           = arg.N;
    int64_t        lda         = arg.lda;
    int64_t        incx        = arg.incx;
    int64_t        incy        = arg.incy;
    T              h_alpha     = arg.get_alpha<T>();
    T              h_beta      = arg.get_beta<T>();
    rocblas_fill   uplo        = char2rocblas_fill(arg.uplo);
    rocblas_stride stride_A    = arg.stride_a;
    rocblas_stride stride_x    = arg.stride_x;
    rocblas_stride stride_y    = arg.stride_y;
    int64_t        batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_hemv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
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

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (N, N, lda, stride_A, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (N, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy, (N, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy_gold, (N, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_vector<T>, halpha, (1));
    HOST_MEMCHECK(host_vector<T>, hbeta, (1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (N, N, lda, stride_A, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stride_y, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_hermitian_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    halpha[0] = h_alpha;
    hbeta[0]  = h_beta;

    hy_gold.copy_from(hy);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;
    double rocblas_error_1;
    double rocblas_error_2;

    /* =====================================================================
    ROCBLAS
    ===================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_hemv_strided_batched_fn,
                       (handle,
                        uplo,
                        N,
                        &h_alpha,
                        dA,
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
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(hbeta));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dy.transfer_from(hy_gold));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_hemv_strided_batched_fn,
                       (handle,
                        uplo,
                        N,
                        d_alpha,
                        dA,
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
                    host_strided_batch_vector<T>, hy_copy, (N, incy, stride_y, batch_count));
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
                                    dA_copy,
                                    (N, N, lda, stride_A, batch_count));
                    DEVICE_MEMCHECK(
                        device_strided_batch_vector<T>, dx_copy, (N, incx, stride_x, batch_count));
                    DEVICE_MEMCHECK(
                        device_strided_batch_vector<T>, dy_copy, (N, incy, stride_y, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(hbeta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_hemv_strided_batched_fn,
                                   (handle_copy,
                                    uplo,
                                    N,
                                    d_alpha_copy,
                                    dA_copy,
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
                        unit_check_general<T>(1, N, incy, stride_y, hy, hy_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(int64_t b = 0; b < batch_count; b++)
            ref_hemv<T>(uplo, N, h_alpha, hA[b], lda, hx[b], incx, h_beta, hy_gold[b], incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy output from device to CPU

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, stride_y, hy_gold, hy, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_1
                    = norm_check_general<T>('F', 1, N, incy, stride_y, hy_gold, hy, batch_count);
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incy, stride_y, hy_gold, hy, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_2
                    = norm_check_general<T>('F', 1, N, incy, stride_y, hy_gold, hy, batch_count);
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
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_hemv_strided_batched_fn,
                          (handle,
                           uplo,
                           N,
                           &h_alpha,
                           dA,
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

        ArgumentModel<e_uplo,
                      e_N,
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
                         hemv_gflop_count<T>(N),
                         hemv_gbyte_count<T>(N),
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
