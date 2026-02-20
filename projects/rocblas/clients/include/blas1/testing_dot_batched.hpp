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

template <typename T, bool CONJ = false>
void testing_dot_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_dot_batched_fn
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_batched<T, true> : rocblas_dot_batched<T, true>)
              : (CONJ ? rocblas_dotc_batched<T, false> : rocblas_dot_batched<T, false>);
    auto rocblas_dot_batched_fn_64
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_batched_64<T, true> : rocblas_dot_batched_64<T, true>)
              : (CONJ ? rocblas_dotc_batched_64<T, false> : rocblas_dot_batched_64<T, false>);

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        int64_t N           = 100;
        int64_t incx        = 1;
        int64_t incy        = 1;
        int64_t stride_y    = incy * N;
        int64_t batch_count = 5;

        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<T>, dy, (N, incy, batch_count));
        DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result, (batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_dot_batched_fn,
                    (nullptr,
                     N,
                     dx.ptr_on_device(),
                     incx,
                     dy.ptr_on_device(),
                     incy,
                     batch_count,
                     d_rocblas_result));

        DAPI_EXPECT(
            rocblas_status_invalid_pointer,
            rocblas_dot_batched_fn,
            (handle, N, nullptr, incx, dy.ptr_on_device(), incy, batch_count, d_rocblas_result));
        DAPI_EXPECT(
            rocblas_status_invalid_pointer,
            rocblas_dot_batched_fn,
            (handle, N, dx.ptr_on_device(), incx, nullptr, incy, batch_count, d_rocblas_result));
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_dot_batched_fn,
                    (handle, N, dx, incx, dy, incy, batch_count, nullptr));
    }
}

template <typename T, bool CONJ = false>
void testing_dot_batched(const Arguments& arg)
{
    auto rocblas_dot_batched_fn
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_batched<T, true> : rocblas_dot_batched<T, true>)
              : (CONJ ? rocblas_dotc_batched<T, false> : rocblas_dot_batched<T, false>);
    auto rocblas_dot_batched_fn_64
        = arg.api & c_API_FORTRAN
              ? (CONJ ? rocblas_dotc_batched_64<T, true> : rocblas_dot_batched_64<T, true>)
              : (CONJ ? rocblas_dotc_batched_64<T, false> : rocblas_dot_batched_64<T, false>);

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t batch_count = arg.batch_count;

    double               rocblas_error_host   = 0;
    double               rocblas_error_device = 0;
    rocblas_local_handle handle{arg};

    // check to prevent undefined memmory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result, (std::max(batch_count, int64_t(1))));

        HOST_MEMCHECK(host_vector<T>, h_rocblas_result, (std::max(batch_count, int64_t(1))));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_dot_batched_fn,
                   (handle, N, nullptr, incx, nullptr, incy, batch_count, d_rocblas_result));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_dot_batched_fn,
                   (handle, N, nullptr, incx, nullptr, incy, batch_count, h_rocblas_result));

        if(batch_count > 0)
        {
            HOST_MEMCHECK(host_vector<T>, cpu_0, (batch_count));
            HOST_MEMCHECK(host_vector<T>, gpu_0, (batch_count));
            CHECK_HIP_ERROR(gpu_0.transfer_from(d_rocblas_result));
            unit_check_general<T>(1, 1, 1, 1, cpu_0, gpu_0, batch_count);
            unit_check_general<T>(1, 1, 1, 1, cpu_0, h_rocblas_result, batch_count);
        }

        return;
    }
    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hy, (N, incy, batch_count));
    HOST_MEMCHECK(host_vector<T>, cpu_result, (batch_count));
    HOST_MEMCHECK(host_vector<T>, rocblas_result_host, (batch_count));
    HOST_MEMCHECK(host_vector<T>, rocblas_result_device, (batch_count));

    //Device-arrays of pointers to device memory
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dy, (N, incy, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result_device, (batch_count));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false, true);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;

    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
    auto  dy_ptr = (arg.algo) ? dx.ptr_on_device() : dy.ptr_on_device();
    auto& hy_ptr = (arg.algo) ? hx : hy;
    if(arg.algo)
        incy = incx;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // GPU BLAS, rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            DAPI_CHECK(rocblas_dot_batched_fn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy_ptr,
                        incy,
                        batch_count,
                        rocblas_result_host));
        }

        if(arg.pointer_mode_device)
        {
            // GPU BLAS, rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_dot_batched_fn,
                       (handle,
                        N,
                        dx.ptr_on_device(),
                        incx,
                        dy_ptr,
                        incy,
                        batch_count,
                        d_rocblas_result_device));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                CHECK_HIP_ERROR(rocblas_result_device.transfer_from(d_rocblas_result_device));
                HOST_MEMCHECK(host_vector<T>, rocblas_result_device_copy, (batch_count));

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

                    //Allocate device memory in new device
                    DEVICE_MEMCHECK(device_batch_vector<T>, dx_copy, (N, incx, batch_count));
                    DEVICE_MEMCHECK(device_batch_vector<T>, dy_copy, (N, incy, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result_device_copy, (batch_count));

                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(dy_copy.transfer_from(hy));

                    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
                    auto dy_ptr_copy
                        = (arg.algo) ? dx_copy.ptr_on_device() : dy_copy.ptr_on_device();

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        DAPI_CHECK(rocblas_dot_batched_fn,
                                   (handle_copy,
                                    N,
                                    dx_copy.ptr_on_device(),
                                    incx,
                                    dy_ptr_copy,
                                    incy,
                                    batch_count,
                                    d_rocblas_result_device_copy));

                        CHECK_HIP_ERROR(
                            rocblas_result_device_copy.transfer_from(d_rocblas_result_device_copy));

                        unit_check_general<T>(1,
                                              1,
                                              1,
                                              1,
                                              rocblas_result_device,
                                              rocblas_result_device_copy,
                                              batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(size_t b = 0; b < batch_count; ++b)
        {
            (CONJ ? ref_dotc<T, T>
                  : ref_dot<T, T>)(N, hx[b], incx, hy_ptr[b], incy, &cpu_result[b]);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // For large N, rocblas_half tends to diverge proportional to N
        // Tolerance is slightly greater than 1 / 1024.0
        bool near_check = arg.initialization == rocblas_initialization::hpl
                          || (std::is_same_v<T, rocblas_half> && N > 10000);

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(near_check)
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(
                        1, 1, 1, 1, cpu_result, rocblas_result_host, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(1, 1, 1, 1, cpu_result, rocblas_result_host, batch_count);
                }
            }

            if(arg.norm_check)
            {
                for(size_t b = 0; b < batch_count; ++b)
                {
                    rocblas_error_host
                        += rocblas_abs((cpu_result[b] - rocblas_result_host[b]) / cpu_result[b]);
                }
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(rocblas_result_device.transfer_from(d_rocblas_result_device));

            if(arg.unit_check)
            {
                if(near_check)
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(
                        1, 1, 1, 1, cpu_result, rocblas_result_device, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(
                        1, 1, 1, 1, cpu_result, rocblas_result_device, batch_count);
                }
            }

            if(arg.norm_check)
            {
                for(size_t b = 0; b < batch_count; ++b)
                {
                    rocblas_error_device
                        += rocblas_abs((cpu_result[b] - rocblas_result_device[b]) / cpu_result[b]);
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_dot_batched_fn,
                          (handle,
                           N,
                           dx.ptr_on_device(),
                           incx,
                           dy_ptr,
                           incy,
                           batch_count,
                           d_rocblas_result_device));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count, e_algo>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            dot_gflop_count<CONJ, T>(N),
            dot_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error_host,
            rocblas_error_device);
    }
}
