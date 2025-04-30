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

#include "rocblas_iamax_iamin_ref.hpp"
#include "testing_common.hpp"

template <typename T, typename R, typename FUNC>
void testing_iamax_iamin_batched_bad_arg(const Arguments& arg, FUNC func)
{

    rocblas_local_handle handle{arg};

    int64_t N = 100, incx = 1, batch_count = 2;

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        R h_rocblas_result; // host address as all early returns

        EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, batch_count, &h_rocblas_result),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, batch_count, &h_rocblas_result),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, batch_count, nullptr),
                              rocblas_status_invalid_pointer);
    }
}

template <typename T,
          void (*REFBLAS_FUNC)(int64_t, const T*, int64_t, int64_t*),
          typename R,
          typename FUNC>
void testing_iamax_iamin_batched(const Arguments& arg, FUNC func)
{
    int64_t        N = arg.N, incx = arg.incx, batch_count = arg.batch_count;
    rocblas_stride stride_x = arg.stride_x;

    rocblas_local_handle handle{arg};

    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        DEVICE_MEMCHECK(device_vector<R>, d_rocblas_result, (std::max(batch_count, int64_t(1))));

        HOST_MEMCHECK(host_vector<R>, h_rocblas_result, (std::max(batch_count, int64_t(1))));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, batch_count, d_rocblas_result));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, batch_count, h_rocblas_result));

        if(batch_count > 0)
        {
            HOST_MEMCHECK(host_vector<R>, cpu_0, (batch_count));
            HOST_MEMCHECK(host_vector<R>, gpu_0, (batch_count));
            CHECK_HIP_ERROR(gpu_0.transfer_from(d_rocblas_result));
            unit_check_general<R>(1, 1, 1, 1, cpu_0, gpu_0, batch_count);
            unit_check_general<R>(1, 1, 1, 1, cpu_0, h_rocblas_result, batch_count);
        }

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_vector<T>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_vector<R>, hr1, (batch_count));
    HOST_MEMCHECK(host_vector<R>, hr2, (batch_count));
    HOST_MEMCHECK(host_vector<R>, cpu_result, (batch_count));
    int64_t i64_result;

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (N, incx, batch_count));
    DEVICE_MEMCHECK(device_vector<R>, dr, (batch_count));

    // Initialize memory on host.
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // Copy data from host to device.
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used      = 0.0;
    double rocblas_error_host = 0.0, rocblas_error_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, hr1));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, dr));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<R>, hr_copy, (batch_count));
                CHECK_HIP_ERROR(hr2.transfer_from(dr));

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
                    DEVICE_MEMCHECK(device_vector<R>, dr_copy, (batch_count));

                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_ROCBLAS_ERROR(func(
                            handle_copy, N, dx_copy.ptr_on_device(), incx, batch_count, dr_copy));
                        CHECK_HIP_ERROR(hr_copy.transfer_from(dr_copy));
                        unit_check_general<R>(batch_count, 1, 1, hr2, hr_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();

            for(size_t batch_index = 0; batch_index < batch_count; ++batch_index)
            {
                REFBLAS_FUNC(N, hx[batch_index], incx, &i64_result);
                cpu_result[batch_index] = R(i64_result);
            }

            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<R>(batch_count, 1, 1, cpu_result, hr1);
            }

            if(arg.norm_check)
            {
                for(size_t batch_index = 0; batch_index < batch_count; ++batch_index)
                {
                    double a1          = double(hr1[batch_index]);
                    double c           = double(cpu_result[batch_index]);
                    rocblas_error_host = std::max(rocblas_error_host, std::abs((c - a1) / c));
                }
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hr2.transfer_from(dr));

            if(arg.unit_check)
            {
                unit_check_general<R>(batch_count, 1, 1, cpu_result, hr2);
            }

            if(arg.norm_check)
            {

                for(size_t batch_index = 0; batch_index < batch_count; ++batch_index)
                {
                    double a2            = double(hr2[batch_index]);
                    double c             = double(cpu_result[batch_index]);
                    rocblas_error_device = std::max(rocblas_error_device, std::abs((c - a2) / c));
                }
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used = 0.0;

        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_ROCBLAS_ERROR(func(handle, N, dx.ptr_on_device(), incx, batch_count, dr));
        }

        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx, e_batch_count>{}.log_args<T>(rocblas_cout,
                                                                arg,
                                                                gpu_time_used,
                                                                ArgumentLogging::NA_value,
                                                                iamax_iamin_gbyte_count<T>(N),
                                                                cpu_time_used,
                                                                rocblas_error_host,
                                                                rocblas_error_device);
    }
}

// dispatch code

template <typename T>
void testing_iamax_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_iamax_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_iamax_batched<T, true>
                                                               : rocblas_iamax_batched<T, false>;
    auto rocblas_iamax_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_iamax_batched_64<T, true>
                                                               : rocblas_iamax_batched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched_bad_arg<T, int64_t>(arg, rocblas_iamax_batched_fn_64);
    else
        testing_iamax_iamin_batched_bad_arg<T, rocblas_int>(arg, rocblas_iamax_batched_fn);
}

template <typename T>
void testing_iamin_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_iamin_batched_fn = arg.api & c_API_FORTRAN ? rocblas_iamin_batched<T, true>
                                                            : rocblas_iamin_batched<T, false>;

    auto rocblas_iamin_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_iamin_batched_64<T, true>
                                                               : rocblas_iamin_batched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched_bad_arg<T, int64_t>(arg, rocblas_iamin_batched_fn_64);
    else
        testing_iamax_iamin_batched_bad_arg<T, rocblas_int>(arg, rocblas_iamin_batched_fn);
}

template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
    auto rocblas_iamax_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_iamax_batched<T, true>
                                                               : rocblas_iamax_batched<T, false>;
    auto rocblas_iamax_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_iamax_batched_64<T, true>
                                                               : rocblas_iamax_batched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched<T, rocblas_iamax_iamin_ref::iamax<T>, int64_t>(
            arg, rocblas_iamax_batched_fn_64);
    else
        testing_iamax_iamin_batched<T, rocblas_iamax_iamin_ref::iamax<T>, rocblas_int>(
            arg, rocblas_iamax_batched_fn);
}

template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
    auto rocblas_iamin_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_iamin_batched<T, true>
                                                               : rocblas_iamin_batched<T, false>;
    auto rocblas_iamin_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_iamin_batched_64<T, true>
                                                               : rocblas_iamin_batched_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_batched<T, rocblas_iamax_iamin_ref::iamin<T>, int64_t>(
            arg, rocblas_iamin_batched_fn_64);
    else
        testing_iamax_iamin_batched<T, rocblas_iamax_iamin_ref::iamin<T>, rocblas_int>(
            arg, rocblas_iamin_batched_fn);
}
