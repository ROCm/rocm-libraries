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
void testing_rotmg_bad_arg(const Arguments& arg)
{
    auto rocblas_rotmg_fn
        = arg.api & c_API_FORTRAN ? rocblas_rotmg<T, true> : rocblas_rotmg<T, false>;
    auto rocblas_rotmg_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_rotmg_64<T, true> : rocblas_rotmg_64<T, false>;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, d1, (1, 1));
    DEVICE_MEMCHECK(device_vector<T>, d2, (1, 1));
    DEVICE_MEMCHECK(device_vector<T>, x1, (1, 1));
    DEVICE_MEMCHECK(device_vector<T>, y1, (1, 1));
    DEVICE_MEMCHECK(device_vector<T>, param, (5, 1));

    DAPI_EXPECT(rocblas_status_invalid_handle, rocblas_rotmg_fn, (nullptr, d1, d2, x1, y1, param));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer, rocblas_rotmg_fn, (handle, nullptr, d2, x1, y1, param));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer, rocblas_rotmg_fn, (handle, d1, nullptr, x1, y1, param));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer, rocblas_rotmg_fn, (handle, d1, d2, nullptr, y1, param));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer, rocblas_rotmg_fn, (handle, d1, d2, x1, nullptr, param));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer, rocblas_rotmg_fn, (handle, d1, d2, x1, y1, nullptr));
}

template <typename T>
void testing_rotmg(const Arguments& arg)
{
    auto rocblas_rotmg_fn
        = arg.api & c_API_FORTRAN ? rocblas_rotmg<T, true> : rocblas_rotmg<T, false>;
    auto rocblas_rotmg_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_rotmg_64<T, true> : rocblas_rotmg_64<T, false>;

    rocblas_local_handle handle{arg};

    double gpu_time_used, cpu_time_used;
    double error_host, error_device;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    HOST_MEMCHECK(host_vector<T>, params, (9, 1));

    if(arg.unit_check || arg.norm_check)
    {
        const int TEST_COUNT = 100;
        int test_count = !(arg.api & c_API_64) ? TEST_COUNT : 1; // only test 1 64bit API sizes
        for(int i = 0; i < test_count; i++)
        {
            // Initialize data on host memory
            rocblas_init_vector(params, arg, rocblas_client_alpha_sets_nan, true);

            // CPU BLAS
            host_vector<T> hparams_gold = params;

            cpu_time_used = get_time_us_no_sync();
            ref_rotmg<T>(&hparams_gold[0],
                         &hparams_gold[1],
                         &hparams_gold[2],
                         &hparams_gold[3],
                         &hparams_gold[4]);
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;

            if(arg.pointer_mode_host)
            {
                // Naming: `h` is in CPU (host) memory(eg hparams), `d` is in GPU (device) memory (eg dparams).
                // Allocate host memory
                host_vector<T> hparams = params;
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

                handle.pre_test(arg);
                DAPI_CHECK(
                    rocblas_rotmg_fn,
                    (handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]));
                handle.post_test(arg);

                if(arg.unit_check)
                    near_check_general<T>(1, 9, 1, hparams_gold, hparams, rel_error);

                if(arg.norm_check)
                    error_host = norm_check_general<T>('F', 1, 9, 1, hparams_gold, hparams);
            }

            if(arg.pointer_mode_device)
            {
                // Allocate device memory
                DEVICE_MEMCHECK(device_vector<T>, dparams, (9, 1));

                CHECK_HIP_ERROR(dparams.transfer_from(params));

                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

                handle.pre_test(arg);
                DAPI_CHECK(rocblas_rotmg_fn,
                           (handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));
                handle.post_test(arg);

                HOST_MEMCHECK(host_vector<T>, hparams, (9, 1));

                CHECK_HIP_ERROR(hparams.transfer_from(dparams));

                if(arg.repeatability_check)
                {
                    HOST_MEMCHECK(host_vector<T>, hparams_copy, (9, 1));

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
                        DEVICE_MEMCHECK(device_vector<T>, dparams_copy, (9, 1));

                        CHECK_ROCBLAS_ERROR(
                            rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                        for(int runs = 0; runs < arg.iters; runs++)
                        {
                            CHECK_HIP_ERROR(dparams_copy.transfer_from(params));
                            DAPI_CHECK(rocblas_rotmg_fn,
                                       (handle_copy,
                                        dparams_copy,
                                        dparams_copy + 1,
                                        dparams_copy + 2,
                                        dparams_copy + 3,
                                        dparams_copy + 4));
                            CHECK_HIP_ERROR(hparams_copy.transfer_from(dparams_copy));
                            unit_check_general<T>(1, 9, 1, hparams, hparams_copy);
                        }
                    }
                    return;
                }

                if(arg.unit_check)
                    near_check_general<T>(1, 9, 1, hparams_gold, hparams, rel_error);

                if(arg.norm_check)
                    error_device = norm_check_general<T>('F', 1, 9, 1, hparams_gold, hparams);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        host_vector<T> hparams = params;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            hparams = params;
            DAPI_DISPATCH(
                rocblas_rotmg_fn,
                (handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]));
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used) / arg.iters;

        rocblas_cout << "rocblas-us,CPU-us";
        if(arg.norm_check)
            rocblas_cout << ",norm_error_host_ptr,norm_error_dev_ptr";
        rocblas_cout << std::endl;

        rocblas_cout << gpu_time_used << "," << cpu_time_used;
        if(arg.norm_check)
            rocblas_cout << ',' << error_host << ',' << error_device;
        rocblas_cout << std::endl;
    }
}
