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

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_scal_ex_fn = arg.api & c_API_FORTRAN ? rocblas_scal_ex_fortran : rocblas_scal_ex;
    auto rocblas_scal_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_scal_ex_64_fortran : rocblas_scal_ex_64;

    rocblas_datatype alpha_type     = rocblas_type2datatype<Ta>();
    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

    int64_t N    = 100;
    int64_t incx = 1;
    Ta      alpha(0.6);

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_scal_ex_fn,
                (nullptr, N, &alpha, alpha_type, dx, x_type, incx, execution_type));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_scal_ex_fn,
                (handle, N, nullptr, alpha_type, dx, x_type, incx, execution_type));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_scal_ex_fn,
                (handle, N, &alpha, alpha_type, nullptr, x_type, incx, execution_type));

    DAPI_EXPECT(rocblas_status_not_implemented,
                rocblas_scal_ex_fn,
                (handle,
                 N,
                 nullptr,
                 rocblas_datatype_f32_c,
                 nullptr,
                 rocblas_datatype_f64_r,
                 incx,
                 rocblas_datatype_f32_c));
}

template <typename Ta, typename Tx = Ta, typename Tex = Tx>
void testing_scal_ex(const Arguments& arg)
{
    auto rocblas_scal_ex_fn = arg.api & c_API_FORTRAN ? rocblas_scal_ex_fortran : rocblas_scal_ex;
    auto rocblas_scal_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_scal_ex_64_fortran : rocblas_scal_ex_64;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype execution_type = arg.compute_type;

    int64_t N       = arg.N;
    int64_t incx    = arg.incx;
    Ta      h_alpha = arg.get_alpha<Ta>();

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0 || incx <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_scal_ex_fn,
                   (handle, N, nullptr, alpha_type, nullptr, x_type, incx, execution_type));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<Tx>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<Tx>, hx_gold, (N, incx));
    HOST_MEMCHECK(host_vector<Ta>, halpha, (1));
    halpha[0] = h_alpha;

    // Initial Data on CPU
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);

    // allocate memory on device
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Ta>, d_alpha, (1));

    // copy vector is easy in STL; hx_gold = hx: save a copy in hx_gold which will be output of CPU
    // BLAS
    hx_gold = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used, cpu_time_used;
    double rocblas_error_host   = 0.0;
    double rocblas_error_device = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // GPU BLAS, rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_scal_ex_fn,
                       (handle, N, &h_alpha, alpha_type, dx, x_type, incx, execution_type));
            handle.post_test(arg);

            // Transfer output from device to CPU
            CHECK_HIP_ERROR(hx.transfer_from(dx));
        }

        if(arg.pointer_mode_device)
        {
            // copy data from CPU to device for rocblas_pointer_mode_device test
            CHECK_HIP_ERROR(dx.transfer_from(hx_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(halpha));

            // GPU BLAS, rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_scal_ex_fn,
                       (handle, N, d_alpha, alpha_type, dx, x_type, incx, execution_type));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<Tx>, hx_copy, (N, incx));

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

                    //Allocate device memory in new device
                    DEVICE_MEMCHECK(device_vector<Tx>, dx_copy, (N, incx));
                    DEVICE_MEMCHECK(device_vector<Ta>, d_alpha_copy, (1));

                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(halpha));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dx_copy.transfer_from(hx_gold));

                        DAPI_CHECK(rocblas_scal_ex_fn,
                                   (handle_copy,
                                    N,
                                    d_alpha_copy,
                                    alpha_type,
                                    dx_copy,
                                    x_type,
                                    incx,
                                    execution_type));
                        CHECK_HIP_ERROR(hx_copy.transfer_from(dx_copy));
                        unit_check_general<Tx>(1, N, incx, hx, hx_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        ref_scal(N, h_alpha, (Tx*)hx_gold, incx);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, incx, hx_gold, hx);
            }
            if(arg.norm_check)
            {
                rocblas_error_host = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx);
            }
        }

        if(arg.pointer_mode_device)
        {
            // Transfer output from device to CPU
            CHECK_HIP_ERROR(hx.transfer_from(dx));

            if(arg.unit_check)
            {
                unit_check_general<Tx>(1, N, incx, hx_gold, hx);
            }
            if(arg.norm_check)
            {
                rocblas_error_device = norm_check_general<Tx>('F', 1, N, incx, hx_gold, hx);
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_scal_ex_fn,
                          (handle, N, &h_alpha, alpha_type, dx, x_type, incx, execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx>{}.log_args<Tx>(rocblas_cout,
                                                           arg,
                                                           gpu_time_used,
                                                           scal_gflop_count<Tx, Ta>(N),
                                                           scal_gbyte_count<Tx>(N),
                                                           cpu_time_used,
                                                           rocblas_error_host,
                                                           rocblas_error_device);
    }
}
