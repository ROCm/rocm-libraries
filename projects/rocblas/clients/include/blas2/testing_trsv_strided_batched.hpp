/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_trsv_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_trsv_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_trsv_strided_batched<T, true>
                                               : rocblas_trsv_strided_batched<T, false>;

    auto rocblas_trsv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_trsv_strided_batched_64<T, true>
                                                  : rocblas_trsv_strided_batched_64<T, false>;

    const int64_t           N           = 100;
    const int64_t           lda         = 100;
    const int64_t           incx        = 1;
    const int64_t           batch_count = 1;
    const rocblas_stride    stride_a    = N * lda;
    const rocblas_stride    stride_x    = N;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (N, N, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (N, incx, stride_x, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (N, N, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));

    // Checks.
    DAPI_EXPECT(
        rocblas_status_invalid_handle,
        rocblas_trsv_strided_batched_fn,
        (nullptr, uplo, transA, diag, N, dA, lda, stride_a, dx, incx, stride_x, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_trsv_strided_batched_fn,
                (handle,
                 rocblas_fill_full,
                 transA,
                 diag,
                 N,
                 dA,
                 lda,
                 stride_a,
                 dx,
                 incx,
                 stride_x,
                 batch_count));

    // arg_checks code shared so transA, diag tested only in non-batched
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_trsv_strided_batched_fn,
        (handle, uplo, transA, diag, N, nullptr, lda, stride_a, dx, incx, stride_x, batch_count));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_trsv_strided_batched_fn,
        (handle, uplo, transA, diag, N, dA, lda, stride_a, nullptr, incx, stride_x, batch_count));
}

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40

template <typename T>
void testing_trsv_strided_batched(const Arguments& arg)
{
    auto rocblas_trsv_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_trsv_strided_batched<T, true>
                                               : rocblas_trsv_strided_batched<T, false>;

    auto rocblas_trsv_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_trsv_strided_batched_64<T, true>
                                                  : rocblas_trsv_strided_batched_64<T, false>;

    int64_t        N           = arg.N;
    int64_t        lda         = arg.lda;
    int64_t        incx        = arg.incx;
    int64_t        batch_count = arg.batch_count;
    char           char_uplo   = arg.uplo;
    char           char_transA = arg.transA;
    char           char_diag   = arg.diag;
    rocblas_stride stride_a    = arg.stride_a;
    rocblas_stride stride_x    = arg.stride_x;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_trsv_strided_batched_fn,
                    (handle,
                     uplo,
                     transA,
                     diag,
                     N,
                     nullptr,
                     lda,
                     stride_a,
                     nullptr,
                     incx,
                     stride_x,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (N, N, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hAAT, (N, N, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hb, (N, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (N, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx_or_b, (N, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, cpu_x_or_b, (N, incx, stride_x, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (N, N, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx_or_b, (N, incx, stride_x, batch_count));

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        rocblas_client_never_set_nan,
                        rocblas_client_diagonally_dominant_triangular_matrix,
                        true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, hA);
    }

    hb.copy_from(hx);

    // Calculate hb = hA*hx;
    for(size_t b = 0; b < batch_count; b++)
    {
        ref_trmv<T>(uplo, transA, diag, N, hA[b], lda, hb + stride_x * b, incx);
    }

    cpu_x_or_b.copy_from(hb);
    hx_or_b.copy_from(hb);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));
    CHECK_HIP_ERROR(dA.transfer_from(hA));

    double error_host              = 0.0;
    double error_device            = 0.0;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // calculate dxorb <- A^(-1) b   rocblas_device_pointer_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_trsv_strided_batched_fn,
                       (handle,
                        uplo,
                        transA,
                        diag,
                        N,
                        dA,
                        lda,
                        stride_a,
                        dx_or_b,
                        incx,
                        stride_x,
                        batch_count));
            handle.post_test(arg);

            CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));
        }

        if(arg.pointer_mode_device)
        {
            // calculate dxorb <- A^(-1) b   rocblas_device_pointer_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dx_or_b.transfer_from(cpu_x_or_b));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_trsv_strided_batched_fn,
                       (handle,
                        uplo,
                        transA,
                        diag,
                        N,
                        dA,
                        lda,
                        stride_a,
                        dx_or_b,
                        incx,
                        stride_x,
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(
                    host_strided_batch_vector<T>, hx_or_b_copy, (N, incx, stride_x, batch_count));
                CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));
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
                                    (N, N, lda, stride_a, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_vector<T>,
                                    dx_or_b_copy,
                                    (N, incx, stride_x, batch_count));

                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dx_or_b_copy.transfer_from(cpu_x_or_b));
                        DAPI_CHECK(rocblas_trsv_strided_batched_fn,
                                   (handle_copy,
                                    uplo,
                                    transA,
                                    diag,
                                    N,
                                    dA_copy,
                                    lda,
                                    stride_a,
                                    dx_or_b_copy,
                                    incx,
                                    stride_x,
                                    batch_count));
                        CHECK_HIP_ERROR(hx_or_b_copy.transfer_from(dx_or_b_copy));
                        unit_check_general<T>(
                            1, N, incx, stride_x, hx_or_b, hx_or_b_copy, batch_count);
                    }
                }
                return;
            }
        }

        if(arg.pointer_mode_host)
        {
            //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
            // calculate norm 1 of vector E
            error_host = vector_norm_1(N, incx, hx, hx_or_b);

            if(arg.unit_check)
                trsm_err_res_check<T>(error_host, N, error_eps_multiplier, eps);

            for(size_t b = 0; b < batch_count; b++)
                ref_trmv<T>(uplo, transA, diag, N, hA[b], lda, hx_or_b[b], incx);

            auto error_host_res = vector_norm_1(N, incx, hx_or_b, hb);
            if(arg.unit_check)
                trsm_err_res_check<T>(error_host_res, N, error_eps_multiplier, eps);
            error_host = std::max(error_host, error_host_res);
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));
            error_device = vector_norm_1(N, incx, hx, hx_or_b);

            if(arg.unit_check)
                trsm_err_res_check<T>(error_host, N, error_eps_multiplier, eps);

            for(size_t b = 0; b < batch_count; b++)
                ref_trmv<T>(uplo, transA, diag, N, hA[b], lda, hx_or_b[b], incx);

            auto error_device_res = vector_norm_1(N, incx, hx_or_b, hb);
            if(arg.unit_check)
                trsm_err_res_check<T>(error_device_res, N, error_eps_multiplier, eps);
            error_device = std::max(error_device, error_device_res);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used, cpu_time_used = 0.0;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_trsv_strided_batched_fn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           N,
                           dA,
                           lda,
                           stride_a,
                           dx_or_b,
                           incx,
                           stride_x,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        if(arg.unit_check || arg.norm_check)
        {
            // CPU cblas
            cpu_time_used = get_time_us_no_sync();

            for(size_t b = 0; b < batch_count; b++)
                ref_trsv<T>(uplo, transA, diag, N, hA[b], lda, cpu_x_or_b[b], incx);

            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        ArgumentModel<e_uplo,
                      e_transA,
                      e_diag,
                      e_N,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trsv_gflop_count<T>(N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         error_host,
                         error_device);
    }
}
