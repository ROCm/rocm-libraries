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
void testing_trmv_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_trmv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_trmv_batched<T, true> : rocblas_trmv_batched<T, false>;

    auto rocblas_trmv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_trmv_batched_64<T, true>
                                                              : rocblas_trmv_batched_64<T, false>;

    const int64_t           N           = 100;
    const int64_t           lda         = 100;
    const int64_t           incx        = 1;
    const rocblas_int       batch_count = 1;
    const rocblas_operation transA      = rocblas_operation_none;
    const rocblas_fill      uplo        = rocblas_fill_lower;
    const rocblas_diagonal  diag        = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dA, (N, N, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));

    // Checks.
    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_trmv_batched_fn,
                (nullptr,
                 uplo,
                 transA,
                 diag,
                 N,
                 dA.ptr_on_device(),
                 lda,
                 dx.ptr_on_device(),
                 incx,
                 batch_count));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_trmv_batched_fn,
                (handle,
                 rocblas_fill_full,
                 transA,
                 diag,
                 N,
                 dA.ptr_on_device(),
                 lda,
                 dx.ptr_on_device(),
                 incx,
                 batch_count));

    // arg_checks code shared so transA, diag tested only in non-batched
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_trmv_batched_fn,
        (handle, uplo, transA, diag, N, nullptr, lda, dx.ptr_on_device(), incx, batch_count));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_trmv_batched_fn,
        (handle, uplo, transA, diag, N, dA.ptr_on_device(), lda, nullptr, incx, batch_count));
}

template <typename T>
void testing_trmv_batched(const Arguments& arg)
{
    auto rocblas_trmv_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_trmv_batched<T, true> : rocblas_trmv_batched<T, false>;

    auto rocblas_trmv_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_trmv_batched_64<T, true>
                                                              : rocblas_trmv_batched_64<T, false>;

    int64_t N = arg.N, lda = arg.lda, incx = arg.incx, batch_count = arg.batch_count;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle{arg};

    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_trmv_batched_fn,
                    (handle, uplo, transA, diag, N, nullptr, lda, nullptr, incx, batch_count));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<T>, hA, (N, N, lda, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hres, (N, incx, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dA, (N, N, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used, rocblas_error;
    auto   dA_on_device = dA.ptr_on_device();
    auto   dx_on_device = dx.ptr_on_device();

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        handle.pre_test(arg);
        // GPU BLAS
        DAPI_CHECK(
            rocblas_trmv_batched_fn,
            (handle, uplo, transA, diag, N, dA_on_device, lda, dx_on_device, incx, batch_count));
        handle.post_test(arg);

        // fetch GPU
        CHECK_HIP_ERROR(hres.transfer_from(dx));

        if(arg.repeatability_check)
        {
            HOST_MEMCHECK(host_batch_vector<T>, hres_copy, (N, incx, batch_count));

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
                DEVICE_MEMCHECK(device_batch_matrix<T>, dA_copy, (N, N, lda, batch_count));
                DEVICE_MEMCHECK(device_batch_vector<T>, dx_copy, (N, incx, batch_count));

                CHECK_HIP_ERROR(dA_copy.transfer_from(hA));

                CHECK_ROCBLAS_ERROR(
                    rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                for(int runs = 0; runs < arg.iters; runs++)
                {
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    DAPI_CHECK(rocblas_trmv_batched_fn,
                               (handle_copy,
                                uplo,
                                transA,
                                diag,
                                N,
                                dA_copy.ptr_on_device(),
                                lda,
                                dx_copy.ptr_on_device(),
                                incx,
                                batch_count));
                    CHECK_HIP_ERROR(hres_copy.transfer_from(dx_copy));
                    unit_check_general<T>(1, N, incx, hres, hres_copy, batch_count);
                }
            }
            return;
        }

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();
            for(size_t batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                ref_trmv<T>(uplo, transA, diag, N, hA[batch_index], lda, hx[batch_index], incx);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // Unit check.
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx, hres, batch_count);
        }

        // Norm check.
        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, incx, hx, hres, batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_trmv_batched_fn,
                          (handle,
                           uplo,
                           transA,
                           diag,
                           N,
                           dA_on_device,
                           lda,
                           dx_on_device,
                           incx,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        // Log performance
        ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_lda, e_incx, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            trmv_gflop_count<T>(N),
            trmv_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
