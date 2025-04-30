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

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_dot_ex_fn = arg.api & c_API_FORTRAN
                                 ? (CONJ ? rocblas_dotc_ex_fortran : rocblas_dot_ex_fortran)
                                 : (CONJ ? rocblas_dotc_ex : rocblas_dot_ex);
    auto rocblas_dot_ex_fn_64
        = arg.api & c_API_FORTRAN ? (CONJ ? rocblas_dotc_ex_64_fortran : rocblas_dot_ex_64_fortran)
                                  : (CONJ ? rocblas_dotc_ex_64 : rocblas_dot_ex_64);

    rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
    rocblas_datatype y_type         = rocblas_type2datatype<Ty>();
    rocblas_datatype result_type    = rocblas_type2datatype<Tr>();
    rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

    int64_t N    = 100;
    int64_t incx = 1;
    int64_t incy = 1;

    rocblas_local_handle handle{arg};

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Ty>, dy, (N, incy));
    DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result, (1));

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_dot_ex_fn,
                (nullptr,
                 N,
                 dx,
                 x_type,
                 incx,
                 dy,
                 y_type,
                 incy,
                 d_rocblas_result,
                 result_type,
                 execution_type));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dot_ex_fn,
                (handle,
                 N,
                 nullptr,
                 x_type,
                 incx,
                 dy,
                 y_type,
                 incy,
                 d_rocblas_result,
                 result_type,
                 execution_type));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dot_ex_fn,
                (handle,
                 N,
                 dx,
                 x_type,
                 incx,
                 nullptr,
                 y_type,
                 incy,
                 d_rocblas_result,
                 result_type,
                 execution_type));
    DAPI_EXPECT(
        rocblas_status_invalid_pointer,
        rocblas_dot_ex_fn,
        (handle, N, dx, x_type, incx, dy, y_type, incy, nullptr, result_type, execution_type));
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_ex(const Arguments& arg)
{
    auto rocblas_dot_ex_fn = arg.api & c_API_FORTRAN
                                 ? (CONJ ? rocblas_dotc_ex_fortran : rocblas_dot_ex_fortran)
                                 : (CONJ ? rocblas_dotc_ex : rocblas_dot_ex);
    auto rocblas_dot_ex_fn_64
        = arg.api & c_API_FORTRAN ? (CONJ ? rocblas_dotc_ex_64_fortran : rocblas_dot_ex_64_fortran)
                                  : (CONJ ? rocblas_dotc_ex_64 : rocblas_dot_ex_64);

    rocblas_datatype x_type         = arg.a_type;
    rocblas_datatype y_type         = arg.b_type;
    rocblas_datatype result_type    = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    double               rocblas_error_host;
    double               rocblas_error_device;
    rocblas_local_handle handle{arg};

    // check to prevent undefined memmory allocation error
    if(N <= 0)
    {
        DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result, (1));

        HOST_MEMCHECK(host_vector<Tr>, h_rocblas_result, (1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_dot_ex_fn,
                   (handle,
                    N,
                    nullptr,
                    x_type,
                    incx,
                    nullptr,
                    y_type,
                    incy,
                    d_rocblas_result,
                    result_type,
                    execution_type));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_dot_ex_fn,
                   (handle,
                    N,
                    nullptr,
                    x_type,
                    incx,
                    nullptr,
                    y_type,
                    incy,
                    h_rocblas_result,
                    result_type,
                    execution_type));

        Tr cpu_0 = Tr(0);
        Tr gpu_0, gpu_1;
        CHECK_HIP_ERROR(hipMemcpy(&gpu_0, d_rocblas_result, sizeof(Tr), hipMemcpyDeviceToHost));
        gpu_1 = h_rocblas_result[0];
        unit_check_general<Tr>(1, 1, 1, &cpu_0, &gpu_0);
        unit_check_general<Tr>(1, 1, 1, &cpu_0, &gpu_1);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<Tx>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<Ty>, hy, (N, incy));
    HOST_MEMCHECK(host_vector<Tr>, cpu_result, (1, 1));
    HOST_MEMCHECK(host_vector<Tr>, rocblas_result, (1, 1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Ty>, dy, (N, incy));
    DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result_device, (1));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double gpu_time_used, cpu_time_used;

    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
    auto dy_ptr = (arg.algo) ? (Tx*)(dx) : (Ty*)(dy);
    auto hy_ptr = (arg.algo) ? &hx[0] : &hy[0];
    if(arg.algo)
        incy = incx;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            DAPI_CHECK(rocblas_dot_ex_fn,
                       (handle,
                        N,
                        dx,
                        x_type,
                        incx,
                        dy_ptr,
                        y_type,
                        incy,
                        rocblas_result,
                        result_type,
                        execution_type));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_dot_ex_fn,
                       (handle,
                        N,
                        dx,
                        x_type,
                        incx,
                        dy_ptr,
                        y_type,
                        incy,
                        d_rocblas_result_device,
                        result_type,
                        execution_type));
            handle.post_test(arg);
            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<Tr>, rocblas_result_device_copy, (1, 1));

                CHECK_HIP_ERROR(rocblas_result.transfer_from(d_rocblas_result_device));

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
                    DEVICE_MEMCHECK(device_vector<Tr>, d_rocblas_result_device_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(dy_copy.transfer_from(hy));

                    auto dy_ptr_copy = (arg.algo) ? (Tx*)(dx_copy) : (Ty*)(dy_copy);

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        DAPI_CHECK(rocblas_dot_ex_fn,
                                   (handle_copy,
                                    N,
                                    dx_copy,
                                    x_type,
                                    incx,
                                    dy_ptr_copy,
                                    y_type,
                                    incy,
                                    d_rocblas_result_device_copy,
                                    result_type,
                                    execution_type));
                        CHECK_HIP_ERROR(
                            rocblas_result_device_copy.transfer_from(d_rocblas_result_device_copy));
                        unit_check_general<Tr>(1, 1, 1, rocblas_result, rocblas_result_device_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        (CONJ ? ref_dotc<Tx, Tr> : ref_dot<Tx, Tr>)(N, hx, incx, hy_ptr, incy, cpu_result);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_to_gold = [&] {
            bool near_check = arg.initialization == rocblas_initialization::hpl
                              || (sizeof(Tr) == 2 && N > 10000);

            if(arg.unit_check)
            {
                if(near_check)
                {
                    double tol = dot_near_tolerance<Tex, Tx, Tr>(
                        rocblas_handle(handle)->getArchMajor(), N, cpu_result[0]);
                    near_check_general<Tr>(1, 1, 1, cpu_result, rocblas_result, tol);
                }
                else
                {
                    unit_check_general<Tr>(1, 1, 1, cpu_result, rocblas_result);
                }
            }

            double error = 0.0;
            if(arg.norm_check)
            {
                error = double(rocblas_abs((cpu_result[0] - rocblas_result[0]) / cpu_result[0]));
            }
            return error;
        };

        if(arg.pointer_mode_host)
        {
            rocblas_error_host = compare_to_gold();
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(rocblas_result.transfer_from(d_rocblas_result_device));
            rocblas_error_device = compare_to_gold();
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

            DAPI_DISPATCH(rocblas_dot_ex_fn,
                          (handle,
                           N,
                           dx,
                           x_type,
                           incx,
                           dy_ptr,
                           y_type,
                           incy,
                           d_rocblas_result_device,
                           result_type,
                           execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_algo>{}.log_args<Tx>(rocblas_cout,
                                                                  arg,
                                                                  gpu_time_used,
                                                                  dot_gflop_count<CONJ, Tx>(N),
                                                                  dot_gbyte_count<Tx>(N),
                                                                  cpu_time_used,
                                                                  rocblas_error_host,
                                                                  rocblas_error_device);
    }
}
