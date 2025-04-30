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
void testing_asum_bad_arg(const Arguments& arg)
{
    auto rocblas_asum_fn = arg.api & c_API_FORTRAN ? rocblas_asum<T, true> : rocblas_asum<T, false>;
    auto rocblas_asum_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_asum_64<T, true> : rocblas_asum_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;

        using RT = real_t<T>;

        RT  rocblas_result   = 10;
        RT* h_rocblas_result = &rocblas_result;

        // Allocate device memory
        DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_asum_fn,
                    (nullptr, N, dx, incx, h_rocblas_result));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_asum_fn,
                    (handle, N, nullptr, incx, h_rocblas_result));
        DAPI_EXPECT(
            rocblas_status_invalid_pointer, rocblas_asum_fn, (handle, N, dx, incx, nullptr));
    }
}

template <typename T>
void testing_asum(const Arguments& arg)
{
    auto rocblas_asum_fn = arg.api & c_API_FORTRAN ? rocblas_asum<T, true> : rocblas_asum<T, false>;
    auto rocblas_asum_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_asum_64<T, true> : rocblas_asum_64<T, false>;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;

    using RT = real_t<T>;

    RT rocblas_result;
    RT hr_gold;

    double error_host_ptr;
    double error_device_ptr;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        static const size_t safe_size = 100; // arbitrarily set to 100
        DEVICE_MEMCHECK(device_vector<T>, dx, (safe_size));
        DEVICE_MEMCHECK(device_vector<RT>, dr, (1));

        HOST_MEMCHECK(host_vector<RT>, hr_1, (1));
        HOST_MEMCHECK(host_vector<RT>, hr_2, (1));
        HOST_MEMCHECK(host_vector<RT>, result_0, (1));

        result_0[0] = RT(0);

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_asum_fn, (handle, N, dx, incx, hr_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_asum_fn, (handle, N, dx, incx, dr));
        CHECK_HIP_ERROR(hr_2.transfer_from(dr));

        // check that result is set to 0
        unit_check_general<RT, RT>(1, 1, 1, result_0, hr_1);
        unit_check_general<RT, RT>(1, 1, 1, result_0, hr_2);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<T>, hx, (N, incx));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<RT>, dr, (1));

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double cpu_time_used;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            DAPI_CHECK(rocblas_asum_fn, (handle, N, dx, incx, &rocblas_result));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_asum_fn, (handle, N, dx, incx, dr));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                RT rocblas_result_copy;
                CHECK_HIP_ERROR(hipMemcpy(&rocblas_result, dr, sizeof(RT), hipMemcpyDeviceToHost));

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
                    DEVICE_MEMCHECK(device_vector<T>, dx_copy, (N, incx));
                    DEVICE_MEMCHECK(device_vector<RT>, dr_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        DAPI_CHECK(rocblas_asum_fn, (handle_copy, N, dx_copy, incx, dr_copy));
                        CHECK_HIP_ERROR(hipMemcpy(
                            &rocblas_result_copy, dr_copy, sizeof(RT), hipMemcpyDeviceToHost));
                        unit_check_general<RT, RT>(1, 1, 1, &rocblas_result, &rocblas_result_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        ref_asum<T>(N, hx, incx, &hr_gold);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_to_gold = [&] {
            if(arg.unit_check)
            {
                // Near check for asum ILP64 bit
                bool near_check = arg.initialization == rocblas_initialization::hpl;
                if(near_check)
                {
                    double abs_error = sum_near_tolerance<T>(N, hr_gold);
                    near_check_general<RT, RT>(1, 1, 1, &hr_gold, &rocblas_result, abs_error);
                }
                else
                    unit_check_general<RT, RT>(1, 1, 1, &hr_gold, &rocblas_result);
            }

            double error = 0.0;
            if(arg.norm_check)
            {
                error = std::abs((hr_gold - rocblas_result) / hr_gold);
            }
            return error;
        };

        if(arg.pointer_mode_host)
        {
            error_host_ptr = compare_to_gold();
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(&rocblas_result, dr, sizeof(RT), hipMemcpyDeviceToHost));
            error_device_ptr = compare_to_gold();
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_asum_fn, (handle, N, dx, incx, dr));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(rocblas_cout,
                                                 arg,
                                                 gpu_time_used,
                                                 asum_gflop_count<T>(N),
                                                 asum_gflop_count<T>(N),
                                                 cpu_time_used,
                                                 error_host_ptr,
                                                 error_device_ptr);
    }
}
