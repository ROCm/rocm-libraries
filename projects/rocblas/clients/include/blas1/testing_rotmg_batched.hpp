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
void testing_rotmg_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_rotmg_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_rotmg_batched<T, true>
                                                               : rocblas_rotmg_batched<T, false>;
    auto rocblas_rotmg_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_rotmg_batched_64<T, true>
                                                               : rocblas_rotmg_batched_64<T, false>;

    int64_t batch_count = 5;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_vector<T>, d1, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, d2, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, x1, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, y1, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, param, (5, 1, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_rotmg_batched_fn,
                (nullptr,
                 d1.ptr_on_device(),
                 d2.ptr_on_device(),
                 x1.ptr_on_device(),
                 y1.ptr_on_device(),
                 param.ptr_on_device(),
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_batched_fn,
                (handle,
                 nullptr,
                 d2.ptr_on_device(),
                 x1.ptr_on_device(),
                 y1.ptr_on_device(),
                 param.ptr_on_device(),
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_batched_fn,
                (handle,
                 d1.ptr_on_device(),
                 nullptr,
                 x1.ptr_on_device(),
                 y1.ptr_on_device(),
                 param.ptr_on_device(),
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_batched_fn,
                (handle,
                 d1.ptr_on_device(),
                 d2.ptr_on_device(),
                 nullptr,
                 y1.ptr_on_device(),
                 param.ptr_on_device(),
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_batched_fn,
                (handle,
                 d1.ptr_on_device(),
                 d2.ptr_on_device(),
                 x1.ptr_on_device(),
                 nullptr,
                 param.ptr_on_device(),
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotmg_batched_fn,
                (handle,
                 d1.ptr_on_device(),
                 d2.ptr_on_device(),
                 x1.ptr_on_device(),
                 y1.ptr_on_device(),
                 nullptr,
                 batch_count));
}

template <typename T>
void testing_rotmg_batched(const Arguments& arg)
{

    auto rocblas_rotmg_batched_fn    = arg.api & c_API_FORTRAN ? rocblas_rotmg_batched<T, true>
                                                               : rocblas_rotmg_batched<T, false>;
    auto rocblas_rotmg_batched_fn_64 = arg.api & c_API_FORTRAN ? rocblas_rotmg_batched_64<T, true>
                                                               : rocblas_rotmg_batched_64<T, false>;

    int64_t batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    double gpu_time_used, cpu_time_used;
    double norm_error_host = 0.0, norm_error_device = 0.0;

    T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_rotmg_batched_fn,
                   (handle, nullptr, nullptr, nullptr, nullptr, nullptr, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hd1), `d` is in GPU (device) memory (eg dd1).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_vector<T>, hd1, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hd2, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hy, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hparams, (5, 1, batch_count));

    HOST_MEMCHECK(host_batch_vector<T>, hd1_gold, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hd2_gold, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hx_gold, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hy_gold, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, hparams_gold, (5, 1, batch_count));

    HOST_MEMCHECK(host_batch_vector<T>, rd1, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, rd2, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, rx, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, ry, (1, 1, batch_count));
    HOST_MEMCHECK(host_batch_vector<T>, rparams, (5, 1, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_vector<T>, dd1, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dd2, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dx, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dy, (1, 1, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dparams, (5, 1, batch_count));

    if(arg.unit_check || arg.norm_check)
    {
        const int TEST_COUNT = 100;
        int test_count = !(arg.api & c_API_64) ? TEST_COUNT : 1; // only test 1 64bit API sizes
        for(int i = 0; i < test_count; i++)
        {
            // Initialize data on host memory
            rocblas_init_vector(hd1, arg, rocblas_client_alpha_sets_nan, true);
            rocblas_init_vector(hd2, arg, rocblas_client_alpha_sets_nan, false);
            rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, false);
            rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);
            rocblas_init_vector(hparams, arg, rocblas_client_alpha_sets_nan, false);

            hd1_gold.copy_from(hd1);
            hd2_gold.copy_from(hd2);
            hx_gold.copy_from(hx);
            hy_gold.copy_from(hy);
            hparams_gold.copy_from(hparams);

            cpu_time_used = get_time_us_no_sync();
            for(size_t b = 0; b < batch_count; b++)
            {
                ref_rotmg<T>(hd1_gold[b], hd2_gold[b], hx_gold[b], hy_gold[b], hparams_gold[b]);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;

            if(arg.pointer_mode_host)
            {
                rd1.copy_from(hd1);
                rd2.copy_from(hd2);
                rx.copy_from(hx);
                ry.copy_from(hy);
                rparams.copy_from(hparams);

                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                handle.pre_test(arg);
                DAPI_CHECK(rocblas_rotmg_batched_fn,
                           (handle, rd1, rd2, rx, ry, rparams, batch_count));
                handle.post_test(arg);

                if(arg.unit_check)
                {
                    near_check_general<T>(1, 1, 1, rd1, hd1_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rd2, hd2_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rx, hx_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, ry, hy_gold, batch_count, rel_error);
                    near_check_general<T>(1, 5, 1, rparams, hparams_gold, batch_count, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_host
                        = norm_check_general<T>('F', 1, 1, 1, rd1, hd1_gold, batch_count);
                    norm_error_host
                        += norm_check_general<T>('F', 1, 1, 1, rd2, hd2_gold, batch_count);
                    norm_error_host
                        += norm_check_general<T>('F', 1, 1, 1, rx, hx_gold, batch_count);
                    norm_error_host
                        += norm_check_general<T>('F', 1, 1, 1, ry, hy_gold, batch_count);
                    norm_error_host
                        += norm_check_general<T>('F', 1, 5, 1, rparams, hparams_gold, batch_count);
                }
            }

            if(arg.pointer_mode_device)
            {
                CHECK_HIP_ERROR(dd1.transfer_from(hd1));
                CHECK_HIP_ERROR(dd2.transfer_from(hd2));
                CHECK_HIP_ERROR(dx.transfer_from(hx));
                CHECK_HIP_ERROR(dy.transfer_from(hy));
                CHECK_HIP_ERROR(dparams.transfer_from(hparams));

                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                handle.pre_test(arg);
                DAPI_CHECK(rocblas_rotmg_batched_fn,
                           (handle,
                            dd1.ptr_on_device(),
                            dd2.ptr_on_device(),
                            dx.ptr_on_device(),
                            dy.ptr_on_device(),
                            dparams.ptr_on_device(),
                            batch_count));
                handle.post_test(arg);

                CHECK_HIP_ERROR(rd1.transfer_from(dd1));
                CHECK_HIP_ERROR(rd2.transfer_from(dd2));
                CHECK_HIP_ERROR(rx.transfer_from(dx));
                CHECK_HIP_ERROR(ry.transfer_from(dy));
                CHECK_HIP_ERROR(rparams.transfer_from(dparams));

                if(arg.repeatability_check)
                {
                    HOST_MEMCHECK(host_batch_vector<T>, rd1_copy, (1, 1, batch_count));
                    HOST_MEMCHECK(host_batch_vector<T>, rd2_copy, (1, 1, batch_count));
                    HOST_MEMCHECK(host_batch_vector<T>, rx_copy, (1, 1, batch_count));
                    HOST_MEMCHECK(host_batch_vector<T>, ry_copy, (1, 1, batch_count));
                    HOST_MEMCHECK(host_batch_vector<T>, rparams_copy, (5, 1, batch_count));

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
                        DEVICE_MEMCHECK(device_batch_vector<T>, dd1_copy, (1, 1, batch_count));
                        DEVICE_MEMCHECK(device_batch_vector<T>, dd2_copy, (1, 1, batch_count));
                        DEVICE_MEMCHECK(device_batch_vector<T>, dx_copy, (1, 1, batch_count));
                        DEVICE_MEMCHECK(device_batch_vector<T>, dy_copy, (1, 1, batch_count));
                        DEVICE_MEMCHECK(device_batch_vector<T>, dparams_copy, (5, 1, batch_count));

                        CHECK_ROCBLAS_ERROR(
                            rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                        for(int runs = 0; runs < arg.iters; runs++)
                        {
                            CHECK_HIP_ERROR(dd1_copy.transfer_from(hd1));
                            CHECK_HIP_ERROR(dd2_copy.transfer_from(hd2));
                            CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                            CHECK_HIP_ERROR(dy_copy.transfer_from(hy));
                            CHECK_HIP_ERROR(dparams_copy.transfer_from(hparams));

                            DAPI_CHECK(rocblas_rotmg_batched_fn,
                                       (handle_copy,
                                        dd1_copy.ptr_on_device(),
                                        dd2_copy.ptr_on_device(),
                                        dx_copy.ptr_on_device(),
                                        dy_copy.ptr_on_device(),
                                        dparams_copy.ptr_on_device(),
                                        batch_count));

                            CHECK_HIP_ERROR(rd1_copy.transfer_from(dd1_copy));
                            CHECK_HIP_ERROR(rd2_copy.transfer_from(dd2_copy));
                            CHECK_HIP_ERROR(rx_copy.transfer_from(dx_copy));
                            CHECK_HIP_ERROR(ry_copy.transfer_from(dy_copy));
                            CHECK_HIP_ERROR(rparams_copy.transfer_from(dparams_copy));
                            unit_check_general<T>(1, 1, 1, rd1, rd1_copy, batch_count);
                            unit_check_general<T>(1, 1, 1, rd2, rd2_copy, batch_count);
                            unit_check_general<T>(1, 1, 1, rx, rx_copy, batch_count);
                            unit_check_general<T>(1, 1, 1, ry, ry_copy, batch_count);
                            unit_check_general<T>(1, 5, 1, rparams, rparams_copy, batch_count);
                        }
                    }
                    return;
                }

                if(arg.unit_check)
                {
                    near_check_general<T>(1, 1, 1, rd1, hd1_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rd2, hd2_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, rx, hx_gold, batch_count, rel_error);
                    near_check_general<T>(1, 1, 1, ry, hy_gold, batch_count, rel_error);
                    near_check_general<T>(1, 5, 1, rparams, hparams_gold, batch_count, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_device
                        = norm_check_general<T>('F', 1, 1, 1, rd1, hx_gold, batch_count);
                    norm_error_device
                        += norm_check_general<T>('F', 1, 1, 1, rd2, hd2_gold, batch_count);
                    norm_error_device
                        += norm_check_general<T>('F', 1, 1, 1, rx, hx_gold, batch_count);
                    norm_error_device
                        += norm_check_general<T>('F', 1, 1, 1, ry, hy_gold, batch_count);
                    norm_error_device
                        += norm_check_general<T>('F', 1, 5, 1, rparams, hparams_gold, batch_count);
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dd1.transfer_from(hd1));
        CHECK_HIP_ERROR(dd2.transfer_from(hd2));
        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dparams.transfer_from(hparams));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_rotmg_batched_fn,
                          (handle,
                           dd1.ptr_on_device(),
                           dd2.ptr_on_device(),
                           dx.ptr_on_device(),
                           dy.ptr_on_device(),
                           dparams.ptr_on_device(),
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_batch_count>{}.log_args<T>(rocblas_cout,
                                                   arg,
                                                   gpu_time_used,
                                                   ArgumentLogging::NA_value,
                                                   ArgumentLogging::NA_value,
                                                   cpu_time_used,
                                                   norm_error_host,
                                                   norm_error_device);
    }
}
