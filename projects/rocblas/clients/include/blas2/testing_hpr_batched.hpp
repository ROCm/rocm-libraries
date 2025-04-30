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
void testing_hpr_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_hpr_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_hpr_batched<T, true> : rocblas_hpr_batched<T, false>;
    auto rocblas_hpr_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_hpr_batched_64<T, true>
                                                             : rocblas_hpr_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_fill uplo        = rocblas_fill_upper;
        int64_t      N           = 100;
        int64_t      incx        = 1;
        int64_t      batch_count = 2;

        using U = real_t<T>;

        DEVICE_MEMCHECK(device_vector<U>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<U>, zero_d, (1));

        const U alpha_h(1), zero_h(0);

        const U* alpha = &alpha_h;
        const U* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
        DEVICE_MEMCHECK(
            device_batch_matrix<T>, dAp, (1, rocblas_packed_matrix_size(N), 1, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_hpr_batched_fn,
                    (nullptr, uplo, N, alpha, dx, incx, dAp, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_hpr_batched_fn,
                    (handle, rocblas_fill_full, N, alpha, dx, incx, dAp, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_hpr_batched_fn,
                    (handle, uplo, N, nullptr, dx, incx, dAp, batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_hpr_batched_fn,
                        (handle, uplo, N, alpha, nullptr, incx, dAp, batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_hpr_batched_fn,
                        (handle, uplo, N, alpha, dx, incx, nullptr, batch_count));
        }

        // N==0 all pointers may be null
        DAPI_CHECK(rocblas_hpr_batched_fn,
                   (handle, uplo, 0, nullptr, nullptr, incx, nullptr, batch_count));

        // alpha==0 all pointers may be null
        DAPI_CHECK(rocblas_hpr_batched_fn,
                   (handle, uplo, 0, zero, nullptr, incx, nullptr, batch_count));

        // batch_count==0 all pointers may be null
        DAPI_CHECK(rocblas_hpr_batched_fn, (handle, uplo, N, nullptr, nullptr, incx, nullptr, 0));
    }
}

template <typename T>
void testing_hpr_batched(const Arguments& arg)
{
    auto rocblas_hpr_batched_fn
        = arg.api & c_API_FORTRAN ? rocblas_hpr_batched<T, true> : rocblas_hpr_batched<T, false>;
    auto rocblas_hpr_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_hpr_batched_64<T, true>
                                                             : rocblas_hpr_batched_64<T, false>;

    int64_t      N           = arg.N;
    int64_t      incx        = arg.incx;
    real_t<T>    h_alpha     = arg.get_alpha<real_t<T>>();
    rocblas_fill uplo        = char2rocblas_fill(arg.uplo);
    int64_t      batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument check before allocating invalid memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_hpr_batched_fn,
                    (handle, uplo, N, nullptr, nullptr, incx, nullptr, batch_count));
        return;
    }

    size_t size_A = rocblas_packed_matrix_size(N);

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<T>, hA, (N, N, N, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hAp, (1, size_A, 1, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hAp_gold, (1, size_A, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_vector<real_t<T>>, halpha, (1));

    halpha[0] = h_alpha;

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dAp, (1, size_A, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
    DEVICE_MEMCHECK(device_vector<real_t<T>>, d_alpha, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_hermitian_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false, true);

    //regular to packed matrix conversion
    regular_to_packed(uplo == rocblas_fill_upper, hA, hAp, N);

    hAp_gold.copy_from(hAp);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used;
    double rocblas_error_host;
    double rocblas_error_device;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_hpr_batched_fn,
                       (handle,
                        uplo,
                        N,
                        &h_alpha,
                        dx.ptr_on_device(),
                        incx,
                        dAp.ptr_on_device(),
                        batch_count));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hAp.transfer_from(dAp));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));
            CHECK_HIP_ERROR(dAp.transfer_from(hAp_gold));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_hpr_batched_fn,
                       (handle,
                        uplo,
                        N,
                        d_alpha,
                        dx.ptr_on_device(),
                        incx,
                        dAp.ptr_on_device(),
                        batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_batch_matrix<T>, hAp_copy, (1, size_A, 1, batch_count));
                CHECK_HIP_ERROR(hAp.transfer_from(dAp));
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
                    DEVICE_MEMCHECK(device_batch_matrix<T>, dAp_copy, (1, size_A, 1, batch_count));
                    DEVICE_MEMCHECK(device_batch_vector<T>, dx_copy, (N, incx, batch_count));
                    DEVICE_MEMCHECK(device_vector<real_t<T>>, d_alpha_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dAp_copy.transfer_from(hAp_gold));
                        DAPI_CHECK(rocblas_hpr_batched_fn,
                                   (handle_copy,
                                    uplo,
                                    N,
                                    d_alpha_copy,
                                    dx_copy.ptr_on_device(),
                                    incx,
                                    dAp_copy.ptr_on_device(),
                                    batch_count));
                        CHECK_HIP_ERROR(hAp_copy.transfer_from(dAp_copy));
                        unit_check_general<T>(1, size_A, 1, hAp, hAp_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(size_t i = 0; i < batch_count; i++)
        {
            ref_hpr<T>(uplo, N, h_alpha, hx[i], incx, hAp_gold[i]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(1, size_A, 1, hAp_gold, hAp, batch_count, tol);
            }

            if(arg.norm_check)
            {
                rocblas_error_host
                    = norm_check_general<T>('F', 1, size_A, 1, hAp_gold, hAp, batch_count);
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hAp.transfer_from(dAp));
            if(arg.unit_check)
            {
                const double tol = N * sum_error_tolerance<T>;
                near_check_general<T>(1, size_A, 1, hAp_gold, hAp, batch_count, tol);
            }

            if(arg.norm_check)
            {
                rocblas_error_device
                    = norm_check_general<T>('F', 1, size_A, 1, hAp_gold, hAp, batch_count);
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

            DAPI_DISPATCH(rocblas_hpr_batched_fn,
                          (handle,
                           uplo,
                           N,
                           &h_alpha,
                           dx.ptr_on_device(),
                           incx,
                           dAp.ptr_on_device(),
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_N, e_alpha, e_incx, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            hpr_gflop_count<T>(N),
            hpr_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_host,
            rocblas_error_device);
    }
}
