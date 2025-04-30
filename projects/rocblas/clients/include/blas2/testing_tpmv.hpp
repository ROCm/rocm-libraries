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
void testing_tpmv_bad_arg(const Arguments& arg)
{
    auto rocblas_tpmv_fn = arg.api & c_API_FORTRAN ? rocblas_tpmv<T, true> : rocblas_tpmv<T, false>;
    auto rocblas_tpmv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_tpmv_64<T, true> : rocblas_tpmv_64<T, false>;

    const int64_t           N      = 100;
    const int64_t           incx   = 1;
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_fill      uplo   = rocblas_fill_lower;
    const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dAp, (1, rocblas_packed_matrix_size(N), 1));
    DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));

    //
    // Checks.
    //
    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_tpmv_fn,
                (nullptr, uplo, transA, diag, N, dAp, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tpmv_fn,
                (handle, rocblas_fill_full, transA, diag, N, dAp, dx, incx));

    // arg_checks code shared so transA, diag tested only in non-batched
    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_tpmv_fn,
                (handle, uplo, (rocblas_operation)rocblas_fill_full, diag, N, dAp, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpmv_fn,
                (handle, uplo, transA, diag, N, nullptr, dx, incx));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_tpmv_fn,
                (handle, uplo, transA, diag, N, dAp, nullptr, incx));
    // If N is 64 bit
    if(arg.api & c_API_64)
    {
        int64_t n_over_int32 = 2147483649;
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_tpmv_fn,
                    (nullptr, uplo, transA, diag, n_over_int32, dAp, dx, incx));
    }
}

template <typename T>
void testing_tpmv(const Arguments& arg)
{
    auto rocblas_tpmv_fn = arg.api & c_API_FORTRAN ? rocblas_tpmv<T, true> : rocblas_tpmv<T, false>;

    auto rocblas_tpmv_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_tpmv_64<T, true> : rocblas_tpmv_64<T, false>;

    int64_t N = arg.N, incx = arg.incx;

    char char_uplo = arg.uplo, char_transA = arg.transA, char_diag = arg.diag;

    rocblas_fill         uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation    transA = char2rocblas_operation(char_transA);
    rocblas_diagonal     diag   = char2rocblas_diagonal(char_diag);
    rocblas_local_handle handle{arg};

    bool invalid_size = N < 0 || !incx;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_tpmv_fn,
                    (handle, uplo, transA, diag, N, nullptr, nullptr, incx));

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (N, N, N));
    HOST_MEMCHECK(host_matrix<T>, hAp, (1, rocblas_packed_matrix_size(N), 1));
    HOST_MEMCHECK(host_vector<T>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<T>, hres, (N, incx));

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dAp, (1, rocblas_packed_matrix_size(N), 1));
    DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used;
    double rocblas_error;

    /* =====================================================================
     ROCBLAS
     =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {

        handle.pre_test(arg);
        // ROCBLAS
        DAPI_CHECK(rocblas_tpmv_fn, (handle, uplo, transA, diag, N, dAp, dx, incx));
        handle.post_test(arg);

        // fetch GPU
        CHECK_HIP_ERROR(hres.transfer_from(dx));
        if(arg.repeatability_check)
        {
            HOST_MEMCHECK(host_vector<T>, hres_copy, (N, incx));

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
                DEVICE_MEMCHECK(device_matrix<T>, dAp_copy, (1, rocblas_packed_matrix_size(N), 1));
                DEVICE_MEMCHECK(device_vector<T>, dx_copy, (N, incx));

                CHECK_HIP_ERROR(dAp_copy.transfer_from(hAp));

                CHECK_ROCBLAS_ERROR(
                    rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                for(int runs = 0; runs < arg.iters; runs++)
                {
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    DAPI_CHECK(rocblas_tpmv_fn,
                               (handle_copy, uplo, transA, diag, N, dAp_copy, dx_copy, incx));
                    CHECK_HIP_ERROR(hres_copy.transfer_from(dx_copy));
                    unit_check_general<T>(1, N, incx, hres, hres_copy);
                }
            }
            return;
        }

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();
            ref_tpmv<T>(uplo, transA, diag, N, hAp, hx, incx);
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // Unit check.
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incx, hx, hres);
        }

        // Norm check.
        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, incx, hx, hres);
        }
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

            DAPI_DISPATCH(rocblas_tpmv_fn, (handle, uplo, transA, diag, N, dAp, dx, incx));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // Log performance.
        ArgumentModel<e_uplo, e_transA, e_diag, e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                                           arg,
                                                                           gpu_time_used,
                                                                           tpmv_gflop_count<T>(N),
                                                                           tpmv_gbyte_count<T>(N),
                                                                           cpu_time_used,
                                                                           rocblas_error);
    }
}
