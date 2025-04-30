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
void testing_iamax_iamin_bad_arg(const Arguments& arg, FUNC func)
{
    int64_t N    = 100;
    int64_t incx = 1;

    rocblas_local_handle handle{arg};

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));

    R h_rocblas_result;

    EXPECT_ROCBLAS_STATUS(func(nullptr, N, dx, incx, &h_rocblas_result),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(func(handle, N, nullptr, incx, &h_rocblas_result),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(func(handle, N, dx, incx, nullptr), rocblas_status_invalid_pointer);
}

template <typename T,
          void (*REFBLAS_FUNC)(int64_t, const T*, int64_t, int64_t*),
          typename R,
          typename FUNC>
void testing_iamax_iamin(const Arguments& arg, FUNC func)
{
    int64_t N    = arg.N;
    int64_t incx = arg.incx;

    R h_rocblas_result_host;
    R h_rocblas_result_device;

    R rocblas_error_host;
    R rocblas_error_device;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        DEVICE_MEMCHECK(device_vector<R>, d_result, (1));

        HOST_MEMCHECK(host_vector<R>, h_result, (1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, d_result));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(func(handle, N, nullptr, incx, h_result));

        R dev_ptr_result, host_ptr_result;
        CHECK_HIP_ERROR(hipMemcpy(&dev_ptr_result, d_result, sizeof(R), hipMemcpyDeviceToHost));

        host_ptr_result = h_result[0];

        R cpu_0 = R(0); // 0 is invalid 1 based index
        unit_check_general<R>(1, 1, 1, &cpu_0, &dev_ptr_result);
        unit_check_general<R>(1, 1, 1, &cpu_0, &host_ptr_result);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<T>, hx, (N, incx));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<R>, d_rocblas_result, (1));

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // GPU BLAS rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_ROCBLAS_ERROR(func(handle, N, dx, incx, &h_rocblas_result_host));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            CHECK_ROCBLAS_ERROR(func(handle, N, dx, incx, d_rocblas_result));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                R h_rocblas_result_copy;
                CHECK_HIP_ERROR(hipMemcpy(
                    &h_rocblas_result_device, d_rocblas_result, sizeof(R), hipMemcpyDeviceToHost));

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
                    DEVICE_MEMCHECK(device_vector<T>, dx_copy, (N, incx));
                    DEVICE_MEMCHECK(device_vector<R>, d_rocblas_result_copy, (1));

                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_ROCBLAS_ERROR(
                            func(handle_copy, N, dx_copy, incx, d_rocblas_result_copy));
                        CHECK_HIP_ERROR(hipMemcpy(&h_rocblas_result_copy,
                                                  d_rocblas_result_copy,
                                                  sizeof(R),
                                                  hipMemcpyDeviceToHost));
                        unit_check_general<R>(
                            1, 1, 1, &h_rocblas_result_device, &h_rocblas_result_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        int64_t result_i64;
        REFBLAS_FUNC(N, hx, incx, &result_i64);
        R cpu_result  = R(result_i64);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<R>(1, 1, 1, &cpu_result, &h_rocblas_result_host);
            }

            if(arg.norm_check)
            {
                rocblas_error_host = h_rocblas_result_host - cpu_result;
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(
                &h_rocblas_result_device, d_rocblas_result, sizeof(R), hipMemcpyDeviceToHost));

            if(arg.unit_check)
            {
                unit_check_general<R>(1, 1, 1, &cpu_result, &h_rocblas_result_device);
            }

            if(arg.norm_check)
            {
                rocblas_error_device = h_rocblas_result_device - cpu_result;
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;

        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_ROCBLAS_ERROR(func(handle, N, dx, incx, d_rocblas_result));
        }

        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                 arg,
                                                 gpu_time_used,
                                                 ArgumentLogging::NA_value,
                                                 iamax_iamin_gbyte_count<T>(N),
                                                 cpu_time_used,
                                                 rocblas_error_host,
                                                 rocblas_error_device);
    }
}

// iamin and iamax testing don't use the DAPI pattern as result type pointers can't be used
// for both API the same as int64->int32 for all other arguments.
// The ILP64 API testing templates are instantiated separately for rocblas_int and int64_t API

template <typename T>
void testing_iamax_bad_arg(const Arguments& arg)
{
    auto rocblas_iamax_fn
        = arg.api & c_API_FORTRAN ? rocblas_iamax<T, true> : rocblas_iamax<T, false>;
    auto rocblas_iamax_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_iamax_64<T, true> : rocblas_iamax_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_bad_arg<T, int64_t>(arg, rocblas_iamax_fn_64);
    else
        testing_iamax_iamin_bad_arg<T, rocblas_int>(arg, rocblas_iamax_fn);
}

template <typename T>
void testing_iamin_bad_arg(const Arguments& arg)
{
    auto rocblas_iamin_fn
        = arg.api & c_API_FORTRAN ? rocblas_iamin<T, true> : rocblas_iamin<T, false>;
    auto rocblas_iamin_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_iamin_64<T, true> : rocblas_iamin_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin_bad_arg<T, int64_t>(arg, rocblas_iamin_fn_64);
    else
        testing_iamax_iamin_bad_arg<T, rocblas_int>(arg, rocblas_iamin_fn);
}

template <typename T>
void testing_iamax(const Arguments& arg)
{
    auto rocblas_iamax_fn
        = arg.api & c_API_FORTRAN ? rocblas_iamax<T, true> : rocblas_iamax<T, false>;
    auto rocblas_iamax_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_iamax_64<T, true> : rocblas_iamax_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin<T, rocblas_iamax_iamin_ref::iamax<T>, int64_t>(arg,
                                                                           rocblas_iamax_fn_64);
    else
        testing_iamax_iamin<T, rocblas_iamax_iamin_ref::iamax<T>, rocblas_int>(arg,
                                                                               rocblas_iamax_fn);
}

template <typename T>
void testing_iamin(const Arguments& arg)
{
    auto rocblas_iamin_fn
        = arg.api & c_API_FORTRAN ? rocblas_iamin<T, true> : rocblas_iamin<T, false>;
    auto rocblas_iamin_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_iamin_64<T, true> : rocblas_iamin_64<T, false>;

    if(arg.api & c_API_64)
        testing_iamax_iamin<T, rocblas_iamax_iamin_ref::iamin<T>, int64_t>(arg,
                                                                           rocblas_iamin_fn_64);
    else
        testing_iamax_iamin<T, rocblas_iamax_iamin_ref::iamin<T>, rocblas_int>(arg,
                                                                               rocblas_iamin_fn);
}
