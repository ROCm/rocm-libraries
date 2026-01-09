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
void testing_tbsv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_tbsv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_tbsv_batched<T, true> : rocblas_tbsv_batched<T, false>;

    auto rocblas_tbsv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_tbsv_batched_64<T, true>
                                                              : rocblas_tbsv_batched_64<T, false>;

    const int64_t           N                 = 100;
    const int64_t           K                 = 5;
    const int64_t           lda               = 100;
    const int64_t           incx              = 1;
    const int64_t           batch_count       = 5;
    const int64_t           banded_matrix_row = K + 1;
    const rocblas_operation transA            = rocblas_operation_none;
    const rocblas_fill      uplo              = rocblas_fill_lower;
    const rocblas_diagonal  diag              = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dA, (banded_matrix_row, N, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));

    //
    // Checks.
    //
    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_tbsv_batched_fn,
                (nullptr,
                 uplo,
                 transA,
                 diag,
                 N,
                 K,
                 dA.ptr_on_device(),
                 lda,
                 dx.ptr_on_device(),
                 incx,
                 batch_count));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tbsv_batched_fn,
                (handle,
                 rocblas_fill_full,
                 transA,
                 diag,
                 N,
                 K,
                 dA.ptr_on_device(),
                 lda,
                 dx.ptr_on_device(),
                 incx,
                 batch_count));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_tbsv_batched_fn,
        (handle, uplo, transA, diag, N, K, nullptr, lda, dx.ptr_on_device(), incx, batch_count));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_tbsv_batched_fn,
        (handle, uplo, transA, diag, N, K, dA.ptr_on_device(), lda, nullptr, incx, batch_count));
}

template <typename T>
void testing_tbsv_batched(const Arguments& arg)
{
    auto rocblas_tbsv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_tbsv_batched<T, true> : rocblas_tbsv_batched<T, false>;

    auto rocblas_tbsv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_tbsv_batched_64<T, true>
                                                              : rocblas_tbsv_batched_64<T, false>;

    int64_t           N                 = arg.N;
    int64_t           K                 = arg.K;
    int64_t           lda               = arg.lda;
    int64_t           incx              = arg.incx;
    int64_t           batch_count       = arg.batch_count;
    const int64_t     banded_matrix_row = K + 1;
    char              char_uplo         = arg.uplo;
    char              char_transA       = arg.transA;
    char              char_diag         = arg.diag;
    rocblas_fill      uplo              = char2rocblas_fill(char_uplo);
    rocblas_operation transA            = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag              = char2rocblas_diagonal(char_diag);

    rocblas_status       status;
    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = N < 0 || K < 0 || lda < banded_matrix_row || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_tbsv_batched_fn,
                    (handle, uplo, transA, diag, N, K, nullptr, lda, nullptr, incx, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAb), `d` is in GPU (device) memory (eg dAb).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<T>, hA, (N, N, N, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hAb, (banded_matrix_row, N, lda, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hb, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx_or_b, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, cpu_x_or_b, (N, incx, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dAb, (banded_matrix_row, N, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx_or_b, (N, incx, batch_count));

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        rocblas_client_never_set_nan,
                        rocblas_client_diagonally_dominant_triangular_matrix,
                        true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    // Make hA a banded matrix with k sub/super-diagonals
    banded_matrix_setup<T>(uplo == rocblas_fill_upper, hA, K);

    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, hA);
    }

    // Convert regular-storage hA to banded-storage hAb
    regular_to_banded(uplo == rocblas_fill_upper, hA, hAb, K);

    CHECK_HIP_ERROR(dAb.transfer_from(hAb));

    hb.copy_from(hx);

    // Calculate hb = hA*hx;
    for(size_t b = 0; b < batch_count; b++)
    {
        ref_tbmv<T>(uplo, transA, diag, N, K, hAb[b], lda, hb[b], incx);
    }

    cpu_x_or_b.copy_from(hb);
    hx_or_b.copy_from(hb);

    double max_err                 = 0.0;
    double max_err_res             = 0.0;
    double error_eps_multiplier    = 40.0;
    double residual_eps_multiplier = 40.0;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dxorb <- A^(-1) b
        CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

        handle.pre_test(arg);
        DAPI_CHECK(rocblas_tbsv_batched_fn,
                   (handle,
                    uplo,
                    transA,
                    diag,
                    N,
                    K,
                    dAb.ptr_on_device(),
                    lda,
                    dx_or_b.ptr_on_device(),
                    incx,
                    batch_count));
        handle.post_test(arg);

        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        if(arg.repeatability_check)
        {
            HOST_MEMCHECK(host_batch_vector<T>, hx_or_b_copy, (N, incx, batch_count));

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
                DEVICE_MEMCHECK(device_batch_vector<T>, dx_or_b_copy, (N, incx, batch_count));

                CHECK_HIP_ERROR(dAb_copy.transfer_from(hAb));

                CHECK_ROCBLAS_ERROR(
                    rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                for(int runs = 0; runs < arg.iters; runs++)
                {
                    CHECK_HIP_ERROR(dx_or_b_copy.transfer_from(hb));

                    DAPI_CHECK(rocblas_tbsv_batched_fn,
                               (handle_copy,
                                uplo,
                                transA,
                                diag,
                                N,
                                K,
                                dAb_copy.ptr_on_device(),
                                lda,
                                dx_or_b_copy.ptr_on_device(),
                                incx,
                                batch_count));
                    CHECK_HIP_ERROR(hx_or_b_copy.transfer_from(dx_or_b_copy));
                    unit_check_general<T>(1, N, incx, hx_or_b, hx_or_b_copy, batch_count);
                }
            }
            return;
        }

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate norm 1 of vector E
        max_err = rocblas_abs(vector_norm_1<T>(N, incx, hx, hx_or_b));
        if(arg.unit_check)
            trsm_err_res_check<T>(max_err, N, error_eps_multiplier, eps);

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_tbmv<T>(uplo, transA, diag, N, K, hAb[b], lda, hx_or_b[b], incx);
        }

        //calculate norm 1 of res
        max_err_res = rocblas_abs(vector_norm_1<T>(N, incx, hx_or_b, hb));
        if(arg.unit_check)
            trsm_err_res_check<T>(max_err_res, N, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        double gpu_time_used, cpu_time_used = 0.0;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_tbsv_batched_fn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           N,
                           K,
                           dAb.ptr_on_device(),
                           lda,
                           dx_or_b.ptr_on_device(),
                           incx,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        if(arg.unit_check || arg.norm_check)
        {
            // CPU cblas
            cpu_time_used = get_time_us_no_sync();

            if(arg.norm_check)
                for(size_t b = 0; b < batch_count; b++)
                    ref_tbsv<T>(uplo, transA, diag, N, K, hAb[b], lda, cpu_x_or_b[b], incx);

            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_K, e_lda, e_incx, e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         tbsv_gflop_count<T>(N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         max_err,
                         max_err_res);
    }
}
