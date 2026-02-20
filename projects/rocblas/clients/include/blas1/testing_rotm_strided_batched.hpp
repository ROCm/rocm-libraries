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
void testing_rotm_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_rotm_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_rotm_strided_batched<T, true>
                                                  : rocblas_rotm_strided_batched<T, false>;
    auto rocblas_rotm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_rotm_strided_batched_64<T, true>
                                                  : rocblas_rotm_strided_batched_64<T, false>;

    int64_t             N            = 100;
    int64_t             incx         = 1;
    rocblas_stride      stride_x     = 1;
    int64_t             incy         = 1;
    rocblas_stride      stride_y     = 1;
    rocblas_stride      stride_param = 1;
    int64_t             batch_count  = 5;
    static const size_t safe_size    = 100;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stride_y, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dparam, (5, 1, stride_param, batch_count));

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    DAPI_EXPECT(
        rocblas_status_invalid_handle,
        rocblas_rotm_strided_batched_fn,
        (nullptr, N, dx, incx, stride_x, dy, incy, stride_y, dparam, stride_param, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotm_strided_batched_fn,
                (handle,
                 N,
                 nullptr,
                 incx,
                 stride_x,
                 dy,
                 incy,
                 stride_y,
                 dparam,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_rotm_strided_batched_fn,
                (handle,
                 N,
                 dx,
                 incx,
                 stride_x,
                 nullptr,
                 incy,
                 stride_y,
                 dparam,
                 stride_param,
                 batch_count));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rotm_strided_batched_fn,
        (handle, N, dx, incx, stride_x, dy, incy, stride_y, nullptr, stride_param, batch_count));
}

template <typename T>
void testing_rotm_strided_batched(const Arguments& arg)
{

    auto rocblas_rotm_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_rotm_strided_batched<T, true>
                                                  : rocblas_rotm_strided_batched<T, false>;
    auto rocblas_rotm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_rotm_strided_batched_64<T, true>
                                                  : rocblas_rotm_strided_batched_64<T, false>;

    int64_t        N            = arg.N;
    int64_t        incx         = arg.incx;
    rocblas_stride stride_x     = arg.stride_x;
    int64_t        incy         = arg.incy;
    rocblas_stride stride_y     = arg.stride_y;
    rocblas_stride stride_param = arg.stride_c;
    int64_t        batch_count  = arg.batch_count;

    rocblas_local_handle handle{arg};

    double gpu_time_used, cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;

    T rel_error = std::numeric_limits<T>::epsilon() * 1000;
    // increase relative error for ieee64 bit
    if(std::is_same_v<T, double> || std::is_same_v<T, rocblas_double_complex>)
        rel_error *= 10.0;

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_rotm_strided_batched_fn,
                   (handle,
                    N,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    incy,
                    stride_y,
                    nullptr,
                    stride_param,
                    batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (N, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy, (N, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hparam, (5, 1, stride_param, batch_count));
    HOST_MEMCHECK(host_vector<T>, hdata, (4 * batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stride_y, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dparam, (5, 1, stride_param, batch_count));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);
    rocblas_init_vector(hdata, arg, rocblas_client_alpha_sets_nan, false);

    for(size_t b = 0; b < batch_count; b++)
    {
        T* hparam_ptr = hparam[b];

        // generating simply one set of hparam which will not be appropriate for testing
        // that it zeros out the second element of the rotm vector parameter
        memset(hparam_ptr, 0, 5 * sizeof(T));

        ref_rotmg<T>(
            hdata + b * 4, hdata + b * 4 + 1, hdata + b * 4 + 2, hdata + b * 4 + 3, hparam_ptr);
    }

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    if(arg.unit_check || arg.norm_check)
    {
        // CPU BLAS reference data
        HOST_MEMCHECK(host_strided_batch_vector<T>, hx_gold, (N, incx, stride_x, batch_count));
        HOST_MEMCHECK(host_strided_batch_vector<T>, hy_gold, (N, incy, stride_y, batch_count));

        int flag_count
            = !(arg.api & c_API_64) ? FLAG_COUNT : 1; // only test first flag for 64bit API sizes
        for(int i = 0; i < flag_count; i++)
        {
            for(size_t b = 0; b < batch_count; b++)
                (hparam + b * stride_param)[0] = FLAGS[i];

            hx_gold.copy_from(hx);
            hy_gold.copy_from(hy);

            // Test rocblas_pointer_mode_host
            //if(arg.pointer_mode_device)
            //{
            // TODO: THIS IS NO LONGER SUPPORTED
            // {
            //     CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            //     CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            //     CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            //     CHECK_ROCBLAS_ERROR((rocblas_rotm_strided_batched_fn(
            //         handle, N, dx, incx, stride_x, dy, incy, stride_y, hparam, batch_count)));
            //     HOST_MEMCHECK(host_vector<T>, rx, (size_x));
            //     HOST_MEMCHECK(host_vector<T>, ry, (size_y));
            //     CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            //     CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            //     if(arg.unit_check)
            //     {
            //         T rel_error = std::numeric_limits<T>::epsilon() * 1000;
            //         near_check_general<T,T>(1, N, batch_count, incx, stride_x, hx_gold, rx, rel_error);
            //         near_check_general<T,T>(1, N, batch_count, incy, stride_y, hy_gold, ry, rel_error);
            //     }
            //     if(arg.norm_check)
            //     {
            //         norm_error_host_x
            //             = norm_check_general<T>('F', 1, N, incx, stride_x, batch_count, hx_gold, rx);
            //         norm_error_host_y
            //             = norm_check_general<T>('F', 1, N, incy, stride_x, batch_count, hy_gold, ry);
            //     }
            // }
            //}

            if(arg.pointer_mode_device)
            {
                CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

                CHECK_HIP_ERROR(dx.transfer_from(hx));
                CHECK_HIP_ERROR(dy.transfer_from(hy));
                CHECK_HIP_ERROR(dparam.transfer_from(hparam));

                handle.pre_test(arg);
                DAPI_CHECK(rocblas_rotm_strided_batched_fn,
                           (handle,
                            N,
                            dx,
                            incx,
                            stride_x,
                            dy,
                            incy,
                            stride_y,
                            dparam,
                            stride_param,
                            batch_count));
                handle.post_test(arg);

                CHECK_HIP_ERROR(hx.transfer_from(dx));
                CHECK_HIP_ERROR(hy.transfer_from(dy));

                if(arg.repeatability_check)
                {
                    HOST_MEMCHECK(
                        host_strided_batch_vector<T>, hx_copy, (N, incx, stride_x, batch_count));
                    HOST_MEMCHECK(
                        host_strided_batch_vector<T>, hy_copy, (N, incy, stride_y, batch_count));

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
                        DEVICE_MEMCHECK(device_strided_batch_vector<T>,
                                        dx_copy,
                                        (N, incx, stride_x, batch_count));
                        DEVICE_MEMCHECK(device_strided_batch_vector<T>,
                                        dy_copy,
                                        (N, incy, stride_y, batch_count));
                        DEVICE_MEMCHECK(device_strided_batch_vector<T>,
                                        dparam_copy,
                                        (5, 1, stride_param, batch_count));

                        CHECK_ROCBLAS_ERROR(
                            rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                        for(int runs = 0; runs < arg.iters; runs++)
                        {
                            CHECK_HIP_ERROR(dx_copy.transfer_from(hx_gold));
                            CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                            CHECK_HIP_ERROR(dparam_copy.transfer_from(hparam));
                            DAPI_CHECK(rocblas_rotm_strided_batched_fn,
                                       (handle_copy,
                                        N,
                                        dx_copy,
                                        incx,
                                        stride_x,
                                        dy_copy,
                                        incy,
                                        stride_y,
                                        dparam_copy,
                                        stride_param,
                                        batch_count));
                            CHECK_HIP_ERROR(hx_copy.transfer_from(dx_copy));
                            CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));
                            unit_check_general<T>(1, N, incx, stride_x, hx, hx_copy, batch_count);
                            unit_check_general<T>(1, N, incy, stride_y, hy, hy_copy, batch_count);
                        }
                    }
                    return;
                }

                cpu_time_used = get_time_us_no_sync();
                for(size_t b = 0; b < batch_count; b++)
                {
                    ref_rotm<T>(N, hx_gold[b], incx, hy_gold[b], incy, hparam[b]);
                }
                cpu_time_used = get_time_us_no_sync() - cpu_time_used;

                if(arg.unit_check)
                {
                    near_check_general<T>(
                        1, N, incx, stride_x, hx_gold, hx, batch_count, rel_error);
                    near_check_general<T>(
                        1, N, incy, stride_y, hy_gold, hy, batch_count, rel_error);
                }

                if(arg.norm_check)
                {
                    norm_error_device_x += norm_check_general<T>(
                        'F', 1, N, incx, stride_x, hx_gold, hx, batch_count);
                    norm_error_device_y += norm_check_general<T>(
                        'F', 1, N, incy, stride_y, hy_gold, hy, batch_count);
                }
            }
        }
    }
    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        // Initializing flag value to -1 for all the batches of hparam
        for(size_t b = 0; b < batch_count; b++)
        {
            (hparam + b * stride_param)[0] = FLAGS[0];
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dparam.transfer_from(hparam));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_rotm_strided_batched_fn,
                          (handle,
                           N,
                           dx,
                           incx,
                           stride_x,
                           dy,
                           incy,
                           stride_y,
                           dparam,
                           stride_param,
                           batch_count));
        }
        gpu_time_used = (get_time_us_sync(stream) - gpu_time_used);

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            rotm_gflop_count<T>(N, (hparam + 0 * stride_param)[0]),
            rotm_gbyte_count<T>(N, (hparam + 0 * stride_param)[0]),
            cpu_time_used,
            norm_error_device_x,
            norm_error_device_y);
    }
}
