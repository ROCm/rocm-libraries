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

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs>
void testing_rot_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_rot_ex_fn = arg.api & c_API_FORTRAN ? rocblas_rot_ex_fortran : rocblas_rot_ex;
    auto rocblas_rot_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_rot_ex_64_fortran : rocblas_rot_ex_64;

    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype y_type         = rocblas_type2datatype<Ty>();
    rocblas_datatype cs_type        = rocblas_type2datatype<Tcs>();
    rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

    int64_t              N    = 100;
    int64_t              incx = 1;
    int64_t              incy = 1;
    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Ty>, dy, (N, incy));
    DEVICE_MEMCHECK(device_vector<Tcs>, dc, (1, 1));
    DEVICE_MEMCHECK(device_vector<Tcs>, ds, (1, 1));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_rot_ex_fn,
                (nullptr, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rot_ex_fn,
        (handle, N, nullptr, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rot_ex_fn,
        (handle, N, dx, x_type, incx, nullptr, y_type, incy, dc, ds, cs_type, execution_type));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rot_ex_fn,
        (handle, N, dx, x_type, incx, dy, y_type, incy, nullptr, ds, cs_type, execution_type));

    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_rot_ex_fn,
        (handle, N, dx, x_type, incx, dy, y_type, incy, dc, nullptr, cs_type, execution_type));
}

template <typename Tx, typename Ty = Tx, typename Tcs = Ty, typename Tex = Tcs>
void testing_rot_ex(const Arguments& arg)
{
    auto rocblas_rot_ex_fn = arg.api & c_API_FORTRAN ? rocblas_rot_ex_fortran : rocblas_rot_ex;

    auto rocblas_rot_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_rot_ex_64_fortran : rocblas_rot_ex_64;

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype cs_type        = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    rocblas_local_handle handle{arg};
    double               cpu_time_used;
    double norm_error_host_x = 0.0, norm_error_host_y = 0.0, norm_error_device_x = 0.0,
           norm_error_device_y = 0.0;

    // check to prevent undefined memory allocation error
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_rot_ex_fn,
                   (handle,
                    N,
                    nullptr,
                    x_type,
                    incx,
                    nullptr,
                    y_type,
                    incy,
                    nullptr,
                    nullptr,
                    cs_type,
                    execution_type));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<Tx>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<Ty>, hy, (N, incy));
    HOST_MEMCHECK(host_vector<Tcs>, hc, (1, 1));
    HOST_MEMCHECK(host_vector<Tcs>, hs, (1, 1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Ty>, dy, (N, incy));
    DEVICE_MEMCHECK(device_vector<Tcs>, dc, (1, 1));
    DEVICE_MEMCHECK(device_vector<Tcs>, ds, (1, 1));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);
    rocblas_init_vector(hc, arg, rocblas_client_alpha_sets_nan, false);
    rocblas_init_vector(hs, arg, rocblas_client_alpha_sets_nan, false);

    // CPU BLAS reference data
    host_vector<Tx> hx_gold = hx;
    host_vector<Ty> hy_gold = hy;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dx.transfer_from(hx));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            handle.pre_test(arg);
            DAPI_CHECK(
                rocblas_rot_ex_fn,
                (handle, N, dx, x_type, incx, dy, y_type, incy, hc, hs, cs_type, execution_type));
            handle.post_test(arg);

            CHECK_HIP_ERROR(hx.transfer_from(dx));
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dx.transfer_from(hx_gold));
            CHECK_HIP_ERROR(dy.transfer_from(hy_gold));
            CHECK_HIP_ERROR(dc.transfer_from(hc));
            CHECK_HIP_ERROR(ds.transfer_from(hs));

            DAPI_CHECK(
                rocblas_rot_ex_fn,
                (handle, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<Tx>, hx_copy, (N, incx));
                HOST_MEMCHECK(host_vector<Ty>, hy_copy, (N, incy));

                CHECK_HIP_ERROR(hx.transfer_from(dx));
                CHECK_HIP_ERROR(hy.transfer_from(dy));

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
                    DEVICE_MEMCHECK(device_vector<Tx>, dx_copy, (N, incx));
                    DEVICE_MEMCHECK(device_vector<Ty>, dy_copy, (N, incy));
                    DEVICE_MEMCHECK(device_vector<Tcs>, dc_copy, (1, 1));
                    DEVICE_MEMCHECK(device_vector<Tcs>, ds_copy, (1, 1));

                    CHECK_HIP_ERROR(dc_copy.transfer_from(hc));
                    CHECK_HIP_ERROR(ds_copy.transfer_from(hs));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dx_copy.transfer_from(hx_gold));
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_rot_ex_fn,
                                   (handle_copy,
                                    N,
                                    dx_copy,
                                    x_type,
                                    incx,
                                    dy_copy,
                                    y_type,
                                    incy,
                                    dc_copy,
                                    ds_copy,
                                    cs_type,
                                    execution_type));
                        CHECK_HIP_ERROR(hx_copy.transfer_from(dx_copy));
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));

                        unit_check_general<Tx>(1, N, incx, hx, hx_copy);
                        unit_check_general<Ty>(1, N, incy, hy, hy_copy);
                    }
                }
                return;
            }
        }

        cpu_time_used = get_time_us_no_sync();
        ref_rot<Tx, Ty, Tcs, Tcs>(N, hx_gold, incx, hy_gold, incy, hc, hs);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, incx, hx_gold, hx);
                unit_check_general<Ty>(1, N, incy, hy_gold, hy);
            }
            if(arg.norm_check)
            {
                norm_error_host_x = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx);
                norm_error_host_y = norm_check_general<Ty>('F', 1, N, incy, hy_gold, hy);
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hx.transfer_from(dx));
            CHECK_HIP_ERROR(hy.transfer_from(dy));
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, incx, hx_gold, hx);
                unit_check_general<Ty>(1, N, incy, hy_gold, hy);
            }
            if(arg.norm_check)
            {
                norm_error_device_x = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx);
                norm_error_device_y = norm_check_general<Ty>('F', 1, N, incy, hy_gold, hy);
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dx.transfer_from(hx));
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        CHECK_HIP_ERROR(dc.transfer_from(hc));
        CHECK_HIP_ERROR(ds.transfer_from(hs));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(
                rocblas_rot_ex_fn,
                (handle, N, dx, x_type, incx, dy, y_type, incy, dc, ds, cs_type, execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<Tx>(rocblas_cout,
                                                          arg,
                                                          gpu_time_used,
                                                          rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
                                                          rot_gbyte_count<Tx>(N),
                                                          cpu_time_used,
                                                          norm_error_host_x,
                                                          norm_error_device_x,
                                                          norm_error_host_y,
                                                          norm_error_device_y);
    }
}
