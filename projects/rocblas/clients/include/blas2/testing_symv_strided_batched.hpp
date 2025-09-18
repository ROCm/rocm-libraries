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

#include "testing_common.hpp"

template <typename T>
void testing_symv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_symv_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_symv_strided_batched<T, true>
                                                  : rocblas_symv_strided_batched<T, false>;
    auto rocblas_symv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_symv_strided_batched_64<T, true>
                                                  : rocblas_symv_strided_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_fill uplo        = rocblas_fill_upper;
        int64_t      N           = 100;
        int64_t      incx        = 1;
        int64_t      incy        = 1;
        int64_t      lda         = 100;
        int64_t      batch_count = 2;

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

        rocblas_stride strideA = size_t(N) * lda;
        rocblas_stride stridex = size_t(N) * incx;
        rocblas_stride stridey = size_t(N) * incy;

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (N, N, lda, strideA, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stridex, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stridey, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_symv_strided_batched_fn,
                    (nullptr,
                     uplo,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     rocblas_fill_full,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     dA,
                     lda,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     alpha,
                     dA,
                     lda,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     nullptr,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_symv_strided_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         nullptr,
                         lda,
                         strideA,
                         dx,
                         incx,
                         stridex,
                         beta,
                         dy,
                         incy,
                         stridey,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_symv_strided_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dA,
                         lda,
                         strideA,
                         nullptr,
                         incx,
                         stridex,
                         beta,
                         dy,
                         incy,
                         stridey,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_symv_strided_batched_fn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dA,
                         lda,
                         strideA,
                         dx,
                         incx,
                         stridex,
                         beta,
                         nullptr,
                         incy,
                         stridey,
                         batch_count));
        }

        // If N==0, then all pointers may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     0,
                     nullptr,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     incx,
                     stridex,
                     nullptr,
                     nullptr,
                     incy,
                     stridey,
                     batch_count));

        // If alpha==0, then A and x may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     zero,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        // If alpha==0 && beta==1, then A, x and y may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     zero,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     incx,
                     stridex,
                     one,
                     nullptr,
                     incy,
                     stridey,
                     batch_count));

        // If batch_count==0, then all pointers may be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     incx,
                     stridex,
                     nullptr,
                     nullptr,
                     incy,
                     stridey,
                     0));

        if(arg.api & c_API_64)
        {
            int64_t n_large = 2147483649;
            DAPI_EXPECT(rocblas_status_invalid_size,
                        rocblas_symv_strided_batched_fn,
                        (handle,
                         uplo,
                         n_large,
                         alpha,
                         dA,
                         lda,
                         strideA,
                         dx,
                         incx,
                         stridex,
                         beta,
                         dy,
                         incy,
                         stridey,
                         batch_count));
        }
    }
}

template <typename T>
void testing_symv_strided_batched(const Arguments& arg)
{
    auto rocblas_symv_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_symv_strided_batched<T, true>
                                                  : rocblas_symv_strided_batched<T, false>;
    auto rocblas_symv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_symv_strided_batched_64<T, true>
                                                  : rocblas_symv_strided_batched_64<T, false>;

    int64_t N    = arg.N;
    int64_t lda  = arg.lda;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    HOST_MEMCHECK(host_vector<T>, alpha, (1));
    HOST_MEMCHECK(host_vector<T>, beta, (1));
    alpha[0] = arg.get_alpha<T>();
    beta[0]  = arg.get_beta<T>();

    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    int64_t      batch_count = arg.batch_count;

    rocblas_stride strideA  = size_t(lda) * N;
    size_t         abs_incx = incx >= 0 ? incx : -incx;
    size_t         abs_incy = incy >= 0 ? incy : -incy;
    rocblas_stride stridex  = size_t(N) * abs_incx;
    rocblas_stride stridey  = size_t(N) * abs_incy;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = N < 0 || lda < 1 || lda < N || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_symv_strided_batched_fn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     strideA,
                     nullptr,
                     incx,
                     stridex,
                     nullptr,
                     nullptr,
                     incy,
                     stridey,
                     batch_count));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (N, N, lda, strideA, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (N, incx, stridex, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy, (N, incy, stridey, batch_count));
    HOST_MEMCHECK(
        host_strided_batch_vector<T>, hy_gold, (N, incy, stridey, batch_count)); // gold standard

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (N, N, lda, strideA, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stridex, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stridey, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_symmetric_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);
    rocblas_init_vector(hy, arg, rocblas_client_beta_sets_nan);

    // make copy in hy_gold which will later be used with CPU BLAS
    hy_gold.copy_from(hy);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    double cpu_time_used;
    double h_error, d_error;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // rocblas_pointer_mode_host test
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_symv_strided_batched_fn,
                       (handle,
                        uplo,
                        N,
                        alpha,
                        dA,
                        lda,
                        strideA,
                        dx,
                        incx,
                        stridex,
                        beta,
                        dy,
                        incy,
                        stridey,
                        batch_count));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            // rocblas_pointer_mode_device test
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(d_alpha.transfer_from(alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(beta));

            dy.transfer_from(hy_gold);

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_symv_strided_batched_fn,
                       (handle,
                        uplo,
                        N,
                        d_alpha,
                        dA,
                        lda,
                        strideA,
                        dx,
                        incx,
                        stridex,
                        d_beta,
                        dy,
                        incy,
                        stridey,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                // copy output from device to CPU
                CHECK_HIP_ERROR(hy.transfer_from(dy));

                HOST_MEMCHECK(
                    host_strided_batch_vector<T>, hy_copy, (N, incy, stridey, batch_count));
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
                        device_strided_batch_matrix<T>, dA_copy, (N, N, lda, strideA, batch_count));
                    DEVICE_MEMCHECK(
                        device_strided_batch_vector<T>, dx_copy, (N, incx, stridex, batch_count));
                    DEVICE_MEMCHECK(
                        device_strided_batch_vector<T>, dy_copy, (N, incy, stridey, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(beta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        CHECK_ROCBLAS_ERROR(rocblas_symv_strided_batched_fn(handle_copy,
                                                                            uplo,
                                                                            N,
                                                                            d_alpha_copy,
                                                                            dA_copy,
                                                                            lda,
                                                                            strideA,
                                                                            dx_copy,
                                                                            incx,
                                                                            stridex,
                                                                            d_beta_copy,
                                                                            dy_copy,
                                                                            incy,
                                                                            stridey,
                                                                            batch_count));
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));

                        unit_check_general<T>(1, N, incy, stridey, hy, hy_copy, batch_count);
                    }
                }
                return;
            }
        }

        // cpu reference
        cpu_time_used = get_time_us_no_sync();

        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_symv<T>(uplo, N, alpha[0], hA[b], lda, hx[b], incx, beta[0], hy_gold[b], incy);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    unit_check_general<T>(1, N, incy, stridey, hy_gold, hy, batch_count);
                }
                else
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(1, N, incy, stridey, hy_gold, hy, batch_count, tol);
                }
            }

            if(arg.norm_check)
            {
                h_error = norm_check_general<T>('F', 1, N, incy, stridey, hy_gold, hy, batch_count);
            }
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                if(std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    unit_check_general<T>(1, N, incy, stridey, hy_gold, hy, batch_count);
                }
                else
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(1, N, incy, stridey, hy_gold, hy, batch_count, tol);
                }
            }

            if(arg.norm_check)
            {
                d_error = norm_check_general<T>('F', 1, N, incy, stridey, hy_gold, hy, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_symv_strided_batched_fn,
                          (handle,
                           uplo,
                           N,
                           alpha,
                           dA,
                           lda,
                           strideA,
                           dx,
                           incx,
                           stridex,
                           beta,
                           dy,
                           incy,
                           stridey,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        Arguments targ(arg);
        targ.stride_a = strideA;
        targ.stride_x = stridex;
        targ.stride_y = stridey;

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
                         targ,
                         gpu_time_used,
                         symv_gflop_count<T>(N),
                         symv_gbyte_count<T>(N),
                         cpu_time_used,
                         h_error,
                         d_error);
    }
}
