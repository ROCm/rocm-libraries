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

template <typename Tx, typename Tr = Tx>
void testing_nrm2_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_nrm2_ex_fn = arg.api & c_API_FORTRAN ? rocblas_nrm2_ex_fortran : rocblas_nrm2_ex;
    auto rocblas_nrm2_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_nrm2_ex_64_fortran : rocblas_nrm2_ex_64;

    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype result_type    = rocblas_type2datatype<Tr>();
    rocblas_datatype execution_type = arg.compute_type;

    int64_t N    = 100;
    int64_t incx = 1;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result, (1));

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_nrm2_ex_fn,
                (handle, N, nullptr, x_type, incx, d_rocblas_result, result_type, execution_type));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_nrm2_ex_fn,
                (handle, N, dx, x_type, incx, nullptr, result_type, execution_type));
    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_nrm2_ex_fn,
                (nullptr, N, dx, x_type, incx, d_rocblas_result, result_type, execution_type));
}

template <typename Tx, typename Tr = Tx>
void testing_nrm2_ex(const Arguments& arg)
{
    auto rocblas_nrm2_ex_fn = arg.api & c_API_FORTRAN ? rocblas_nrm2_ex_fortran : rocblas_nrm2_ex;
    auto rocblas_nrm2_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_nrm2_ex_64_fortran : rocblas_nrm2_ex_64;

    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype result_type    = rocblas_type2datatype<Tr>();
    rocblas_datatype execution_type = arg.compute_type;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;

    double rocblas_error_1;
    double rocblas_error_2;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result_0, (1));
        HOST_MEMCHECK(host_vector<Tr>, h_rocblas_result_0, (1));

        rocblas_init_nan(h_rocblas_result_0, 1, 1, 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_rocblas_result_0, h_rocblas_result_0, sizeof(Tr), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(
            rocblas_nrm2_ex_fn,
            (handle, N, nullptr, x_type, incx, d_rocblas_result_0, result_type, execution_type));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(
            rocblas_nrm2_ex_fn,
            (handle, N, nullptr, x_type, incx, h_rocblas_result_0, result_type, execution_type));

        HOST_MEMCHECK(host_vector<Tr>, cpu_0, (1));
        HOST_MEMCHECK(host_vector<Tr>, gpu_0, (1));

        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_rocblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);
        unit_check_general<Tr>(1, 1, 1, cpu_0, h_rocblas_result_0);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<Tx>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<Tr>, rocblas_result_1, (1, 1));
    HOST_MEMCHECK(host_vector<Tr>, rocblas_result_2, (1, 1));
    HOST_MEMCHECK(host_vector<Tr>, cpu_result, (1, 1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result_2, (1, 1));

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
            DAPI_CHECK(
                rocblas_nrm2_ex_fn,
                (handle, N, dx, x_type, incx, rocblas_result_1, result_type, execution_type));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(
                rocblas_nrm2_ex_fn,
                (handle, N, dx, x_type, incx, d_rocblas_result_2, result_type, execution_type));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<Tr>, rocblas_result_copy, (1, 1));
                CHECK_HIP_ERROR(rocblas_result_2.transfer_from(d_rocblas_result_2));

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
                    DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result_2_copy, (1, 1));

                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        DAPI_CHECK(rocblas_nrm2_ex_fn,
                                   (handle_copy,
                                    N,
                                    dx_copy,
                                    x_type,
                                    incx,
                                    d_rocblas_result_2_copy,
                                    result_type,
                                    execution_type));
                        CHECK_HIP_ERROR(rocblas_result_copy.transfer_from(d_rocblas_result_2_copy));
                        unit_check_general<Tr, Tr>(1, 1, 1, rocblas_result_2, rocblas_result_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        ref_nrm2<Tx>(N, hx, incx, cpu_result);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        double abs_result = cpu_result[0] > 0 ? cpu_result[0] : -cpu_result[0];
        double abs_error;

        abs_error = abs_result > 0 ? std::numeric_limits<Tr>::epsilon() * N * abs_result
                                   : std::numeric_limits<Tr>::epsilon() * N;

        double tolerance = 2.0; //  accounts for rounding in reduction sum. depends on n.
            //  If test fails, try decreasing n or increasing tolerance.
        abs_error *= tolerance;

        if(arg.pointer_mode_host)
        {
            if(!rocblas_isnan(arg.alpha))
            {
                if(arg.unit_check)
                {
                    near_check_general<Tr, Tr>(1, 1, 1, cpu_result, rocblas_result_1, abs_error);
                }
            }

            if(arg.norm_check)
            {
                rocblas_error_1
                    = rocblas_abs((cpu_result[0] - rocblas_result_1[0]) / cpu_result[0]);
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(rocblas_result_2.transfer_from(d_rocblas_result_2));

            if(!rocblas_isnan(arg.alpha))
            {
                if(arg.unit_check)
                {
                    near_check_general<Tr, Tr>(1, 1, 1, cpu_result, rocblas_result_2, abs_error);
                }
            }

            if(arg.norm_check)
            {
                rocblas_error_2
                    = rocblas_abs((cpu_result[0] - rocblas_result_2[0]) / cpu_result[0]);
            }
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
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            DAPI_DISPATCH(
                rocblas_nrm2_ex_fn,
                (handle, N, dx, x_type, incx, d_rocblas_result_2, result_type, execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<Tx>(rocblas_cout,
                                                  arg,
                                                  gpu_time_used,
                                                  nrm2_gflop_count<Tx>(N),
                                                  nrm2_gbyte_count<Tx>(N),
                                                  cpu_time_used,
                                                  rocblas_error_1,
                                                  rocblas_error_2);
    }
}
