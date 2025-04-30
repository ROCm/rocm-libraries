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

#include "blas1/rocblas_dot.hpp" // internal API

template <typename T, bool CONJ = false>
void testing_dot_bad_arg(const Arguments& arg)
{
    auto rocblas_dot_fn    = arg.api & c_API_FORTRAN
                                 ? (CONJ ? rocblas_dotc<T, true> : rocblas_dot<T, true>)
                                 : (CONJ ? rocblas_dotc<T, false> : rocblas_dot<T, false>);
    auto rocblas_dot_fn_64 = arg.api & c_API_FORTRAN
                                 ? (CONJ ? rocblas_dotc_64<T, true> : rocblas_dot_64<T, true>)
                                 : (CONJ ? rocblas_dotc_64<T, false> : rocblas_dot_64<T, false>);

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        // Allocate device memory
        DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx));
        DEVICE_MEMCHECK(device_vector<T>, dy, (N, incy));
        DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result, (1, 1));

        // don't write to result so device pointer fine for both host and device mode

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_dot_fn,
                    (nullptr, N, dx, incx, dy, incy, d_rocblas_result));
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_dot_fn,
                    (handle, N, nullptr, incx, dy, incy, d_rocblas_result));
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_dot_fn,
                    (handle, N, dx, incx, nullptr, incy, d_rocblas_result));
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_dot_fn,
                    (handle, N, dx, incx, dy, incy, nullptr));
    }
}

template <typename T, bool CONJ = false>
void testing_dot(const Arguments& arg)
{
    auto rocblas_dot_fn    = arg.api & c_API_FORTRAN
                                 ? (CONJ ? rocblas_dotc<T, true> : rocblas_dot<T, true>)
                                 : (CONJ ? rocblas_dotc<T, false> : rocblas_dot<T, false>);
    auto rocblas_dot_fn_64 = arg.api & c_API_FORTRAN
                                 ? (CONJ ? rocblas_dotc_64<T, true> : rocblas_dot_64<T, true>)
                                 : (CONJ ? rocblas_dotc_64<T, false> : rocblas_dot_64<T, false>);

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    T cpu_result;
    T rocblas_result_host;
    T rocblas_result_device;

    double rocblas_error_host   = 0.0;
    double rocblas_error_device = 0.0;
    bool   HMM                  = arg.HMM;

    rocblas_local_handle handle{arg};

    // check to prevent undefined memmory allocation error
    if(N <= 0)
    {
        DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result, (1));

        HOST_MEMCHECK(host_vector<T>, h_rocblas_result, (1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        DAPI_CHECK(rocblas_dot_fn, (handle, N, nullptr, incx, nullptr, incy, d_rocblas_result));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_dot_fn, (handle, N, nullptr, incx, nullptr, incy, h_rocblas_result));

        T cpu_0 = T(0);
        T gpu_0, gpu_host;
        CHECK_HIP_ERROR(hipMemcpy(&gpu_0, d_rocblas_result, sizeof(T), hipMemcpyDeviceToHost));
        gpu_host = h_rocblas_result[0];
        unit_check_general<T>(1, 1, 1, &cpu_0, &gpu_0);
        unit_check_general<T>(1, 1, 1, &cpu_0, &gpu_host);

        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<T>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<T>, hy, (N, incy));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, dx, (N, incx, HMM));
    DEVICE_MEMCHECK(device_vector<T>, dy, (N, incy, HMM));
    DEVICE_MEMCHECK(device_vector<T>, d_rocblas_result_device, (1, 1, HMM));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;

    // arg.algo indicates to force optimized x dot x kernel algorithm with equal inc
    auto dy_ptr = (arg.algo) ? (T*)(dx) : (T*)(dy);
    auto hy_ptr = (arg.algo) ? &hx[0] : &hy[0];
    if(arg.algo)
        incy = incx;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            // GPU BLAS, rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            DAPI_CHECK(rocblas_dot_fn, (handle, N, dx, incx, dy_ptr, incy, &rocblas_result_host));
        }

        if(arg.pointer_mode_device)
        {
            // GPU BLAS, rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            handle.pre_test(arg);
            if(arg.api != INTERNAL)
            {
                DAPI_CHECK(rocblas_dot_fn,
                           (handle, N, dx, incx, dy_ptr, incy, d_rocblas_result_device));

                if(arg.repeatability_check)
                {
                    T rocblas_result_device_copy;

                    CHECK_HIP_ERROR(hipMemcpy(&rocblas_result_device,
                                              d_rocblas_result_device,
                                              sizeof(T),
                                              hipMemcpyDeviceToHost));

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
                        DEVICE_MEMCHECK(device_vector<T>, dx_copy, (N, incx, HMM));
                        DEVICE_MEMCHECK(device_vector<T>, dy_copy, (N, incy, HMM));
                        DEVICE_MEMCHECK(
                            device_vector<T>, d_rocblas_result_device_copy, (1, 1, HMM));

                        CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy));

                        auto dy_ptr_copy = (arg.algo) ? (T*)(dx_copy) : (T*)(dy_copy);

                        CHECK_ROCBLAS_ERROR(
                            rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                        for(int runs = 0; runs < arg.iters; runs++)
                        {
                            DAPI_CHECK(rocblas_dot_fn,
                                       (handle_copy,
                                        N,
                                        dx_copy,
                                        incx,
                                        dy_ptr_copy,
                                        incy,
                                        d_rocblas_result_device_copy));
                            CHECK_HIP_ERROR(hipMemcpy(&rocblas_result_device_copy,
                                                      d_rocblas_result_device_copy,
                                                      sizeof(T),
                                                      hipMemcpyDeviceToHost));
                            unit_check_general<T>(
                                1, 1, 1, &rocblas_result_device, &rocblas_result_device_copy);
                        }
                    }
                    return;
                }
            }
            else if constexpr(std::is_same_v<T, float>)
            {
                rocblas_stride offset_x = arg.lda;
                rocblas_stride offset_y = arg.ldb;
                CHECK_ROCBLAS_ERROR(
                    (rocblas_internal_dot_template<T, T>)(handle,
                                                          N,
                                                          dx + offset_x,
                                                          -offset_x,
                                                          incx,
                                                          arg.stride_x,
                                                          dy_ptr + offset_y,
                                                          -offset_y,
                                                          incy,
                                                          arg.stride_y,
                                                          1,
                                                          d_rocblas_result_device,
                                                          nullptr)); // N must be small
            }
            handle.post_test(arg);
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        (CONJ ? ref_dotc<T, T> : ref_dot<T, T>)(N, hx, incx, hy_ptr, incy, &cpu_result);
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // For large N, rocblas_half tends to diverge proportional to N
        // Tolerance is slightly greater than 1 / 1024.0
        bool near_check = arg.initialization == rocblas_initialization::hpl
                          || (std::is_same_v<T, rocblas_half> && N > 10000);

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(near_check)
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_host, tol);
                }
                else
                {
                    unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_host);
                }
            }

            if(arg.norm_check)
            {
                rocblas_error_host
                    = double(rocblas_abs((cpu_result - rocblas_result_host) / cpu_result));
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(
                &rocblas_result_device, d_rocblas_result_device, sizeof(T), hipMemcpyDeviceToHost));

            if(arg.unit_check)
            {
                if(near_check)
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_device, tol);
                }
                else
                {
                    unit_check_general<T>(1, 1, 1, &cpu_result, &rocblas_result_device);
                }
            }

            if(arg.norm_check)
            {
                rocblas_error_device
                    = double(rocblas_abs((cpu_result - rocblas_result_device) / cpu_result));
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
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_dot_fn,
                          (handle, N, dx, incx, dy_ptr, incy, d_rocblas_result_device));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_algo>{}.log_args<T>(rocblas_cout,
                                                                 arg,
                                                                 gpu_time_used,
                                                                 dot_gflop_count<CONJ, T>(N),
                                                                 dot_gbyte_count<T>(N),
                                                                 cpu_time_used,
                                                                 rocblas_error_host,
                                                                 rocblas_error_device);
    }
}
