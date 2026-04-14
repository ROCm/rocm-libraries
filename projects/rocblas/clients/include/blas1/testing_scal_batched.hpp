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

template <typename T, typename U = T>
void testing_scal_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_scal_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_scal_batched<T, U, true>
                                                              : rocblas_scal_batched<T, U, false>;
    auto rocblas_scal_batched_fn_64 = arg.api & c_API_FORTRAN
                                          ? rocblas_scal_batched_64<T, U, true>
                                          : rocblas_scal_batched_64<T, U, false>;

    int64_t N           = 100;
    int64_t incx        = 1;
    U       h_alpha     = U(1.0);
    int64_t batch_count = 2;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_scal_batched_fn,
                (nullptr, N, &h_alpha, dx.ptr_on_device(), incx, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_scal_batched_fn,
                (handle, N, nullptr, dx.ptr_on_device(), incx, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_scal_batched_fn,
                (handle, N, &h_alpha, nullptr, incx, batch_count));
}

template <typename T, typename U = T>
void testing_scal_batched(const Arguments& arg)
{
    auto rocblas_scal_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_scal_batched<T, U, true>
                                                              : rocblas_scal_batched<T, U, false>;
    auto rocblas_scal_batched_fn_64 = arg.api & c_API_FORTRAN
                                          ? rocblas_scal_batched_64<T, U, true>
                                          : rocblas_scal_batched_64<T, U, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    U       h_alpha     = arg.get_alpha<U>();
    int64_t batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N < 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_scal_batched_fn, (handle, N, nullptr, nullptr, incx, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx_gold, (N, incx, batch_count));
    HOST_MEMCHECK(host_vector<U>, halpha, (1));
    halpha[0] = h_alpha;

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<U>, d_alpha, (1));

    // Initialize memory on host.
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    hx_gold.copy_from(hx);

    // copy data from CPU to device
    // 1. User intermediate arrays to access device memory from host

    double gpu_time_used, cpu_time_used;
    double rocblas_error_host   = 0.0;
    double rocblas_error_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));

        CHECK_HIP_ERROR(dx.transfer_from(hx));

        if(arg.pointer_mode_host)
        {
            // GPU BLAS, rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_scal_batched_fn,
                       (handle, N, &h_alpha, dx.ptr_on_device(), incx, batch_count));
            handle.post_test(arg);

            // Transfer output from device to CPU
            CHECK_HIP_ERROR(hx.transfer_from(dx));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(dx.transfer_from(hx_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

            // GPU BLAS, rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_scal_batched_fn,
                       (handle, N, d_alpha, dx.ptr_on_device(), incx, batch_count));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_batch_vector<T>, hx_copy, (N, incx, batch_count));
                CHECK_HIP_ERROR(hx.transfer_from(dx));

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

                    // Allocate device memory in new device
                    DEVICE_MEMCHECK(device_batch_vector<T>, dx_copy, (N, incx, batch_count));
                    DEVICE_MEMCHECK(device_vector<U>, d_alpha_copy, (1));

                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dx_copy.transfer_from(hx_gold));
                        DAPI_CHECK(rocblas_scal_batched_fn,
                                   (handle_copy,
                                    N,
                                    d_alpha_copy,
                                    dx_copy.ptr_on_device(),
                                    incx,
                                    batch_count));
                        CHECK_HIP_ERROR(hx_copy.transfer_from(dx_copy));
                        unit_check_general<T>(1, N, incx, hx, hx_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_scal(N, h_alpha, (T*)hx_gold[b], incx);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incx, hx_gold, hx, batch_count);
            }
            if(arg.norm_check)
            {
                rocblas_error_host
                    = norm_check_general<T>('F', 1, N, incx, hx_gold, hx, batch_count);
            }
        }

        if(arg.pointer_mode_device)
        {
            // Transfer output from device to CPU
            CHECK_HIP_ERROR(hx.transfer_from(dx));

            if(arg.unit_check)
            {
                unit_check_general<T>(1, N, incx, hx_gold, hx, batch_count);
            }
            if(arg.norm_check)
            {
                rocblas_error_device
                    = norm_check_general<T>('F', 1, N, incx, hx_gold, hx, batch_count);
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        size_t x_cached_size     = N * batch_count * sizeof(T);
        size_t flush_batch_count = calculate_flush_batch_count(
            arg.flush_batch_count, arg.flush_memory_size, x_cached_size);

        DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count * flush_batch_count));

        CHECK_HIP_ERROR(dx.broadcast_one_batch_vector_from(hx));

        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        auto lambda_to_benchmark = [&](int flush_index) {
            DAPI_DISPATCH(rocblas_scal_batched_fn,
                          (handle,
                           N,
                           &h_alpha,
                           (dx.ptr_on_device() + (flush_index * batch_count)),
                           incx,
                           batch_count));
        };

        Benchmark<decltype(lambda_to_benchmark)> benchmark_scal_batched(
            lambda_to_benchmark, stream, arg, flush_batch_count);

        benchmark_scal_batched.run_timer();

        ArgumentModel<e_N, e_alpha, e_incx, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            benchmark_scal_batched.get_hot_time(),
            scal_gflop_count<T, U>(N),
            scal_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_host,
            rocblas_error_device);
    }
}
