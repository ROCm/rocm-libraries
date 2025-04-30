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

/* ============================================================================================ */
template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_axpy_batched_ex_fn
        = arg.api & c_API_FORTRAN ? rocblas_axpy_batched_ex_fortran : rocblas_axpy_batched_ex;
    auto rocblas_axpy_batched_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_axpy_batched_ex_64_fortran : rocblas_axpy_batched_ex_64;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_datatype alpha_type     = rocblas_type2datatype<Ta>();
        rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
        rocblas_datatype y_type         = rocblas_type2datatype<Ty>();
        rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

        int64_t N = 100, incx = 1, incy = 1, batch_count = 2;

        DEVICE_MEMCHECK(device_vector<Ta>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<Ta>, zero_d, (1));

        const Ta alpha_h(1), zero_h(0);

        const Ta* alpha = &alpha_h;
        const Ta* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_vector<Tx>, dx, (N, incx, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<Ty>, dy, (N, incy, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_axpy_batched_ex_fn,
                    (nullptr,
                     N,
                     &alpha,
                     alpha_type,
                     dx.ptr_on_device(),
                     x_type,
                     incx,
                     dy.ptr_on_device(),
                     y_type,
                     incy,
                     batch_count,
                     execution_type));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_axpy_batched_ex_fn,
                    (handle,
                     N,
                     nullptr,
                     alpha_type,
                     dx.ptr_on_device(),
                     x_type,
                     incx,
                     dy.ptr_on_device(),
                     y_type,
                     incy,
                     batch_count,
                     execution_type));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_axpy_batched_ex_fn,
                        (handle,
                         N,
                         alpha,
                         alpha_type,
                         nullptr,
                         x_type,
                         incx,
                         dy.ptr_on_device(),
                         y_type,
                         incy,
                         batch_count,
                         execution_type));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_axpy_batched_ex_fn,
                        (handle,
                         N,
                         alpha,
                         alpha_type,
                         dx.ptr_on_device(),
                         x_type,
                         incx,
                         nullptr,
                         y_type,
                         incy,
                         batch_count,
                         execution_type));
        }

        // If N == 0, then X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_batched_ex_fn,
                    (handle,
                     0,
                     nullptr,
                     alpha_type,
                     nullptr,
                     x_type,
                     incx,
                     nullptr,
                     y_type,
                     incy,
                     batch_count,
                     execution_type));

        // If alpha == 0, then X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_batched_ex_fn,
                    (handle,
                     N,
                     zero,
                     alpha_type,
                     nullptr,
                     x_type,
                     incx,
                     nullptr,
                     y_type,
                     incy,
                     batch_count,
                     execution_type));

        // If batch_count == 0, then X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_batched_ex_fn,
                    (handle,
                     N,
                     nullptr,
                     alpha_type,
                     nullptr,
                     x_type,
                     incx,
                     nullptr,
                     y_type,
                     incy,
                     0,
                     execution_type));
    }
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_batched_ex(const Arguments& arg)
{
    auto rocblas_axpy_batched_ex_fn
        = arg.api & c_API_FORTRAN ? rocblas_axpy_batched_ex_fortran : rocblas_axpy_batched_ex;
    auto rocblas_axpy_batched_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_axpy_batched_ex_64_fortran : rocblas_axpy_batched_ex_64;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype y_type         = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    rocblas_local_handle handle{arg};
    int64_t              N = arg.N, incx = arg.incx, incy = arg.incy, batch_count = arg.batch_count;

    Ta  h_alpha    = arg.get_alpha<Ta>();
    Tex h_alpha_ex = (Tex)h_alpha;

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_batched_ex_fn,
                    (handle,
                     N,
                     nullptr,
                     alpha_type,
                     nullptr,
                     x_type,
                     incx,
                     nullptr,
                     y_type,
                     incy,
                     batch_count,
                     execution_type));
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;
    size_t size_x   = N * (abs_incx ? abs_incx : 1);
    size_t size_y   = N * (abs_incy ? abs_incy : 1);

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_vector<Tx>, hx, (N, incx, batch_count));
    HOST_MEMCHECK(host_batch_vector<Ty>, hy, (N, incy, batch_count));
    HOST_MEMCHECK(host_batch_vector<Ty>, hy1, (N, incy, batch_count));
    HOST_MEMCHECK(host_batch_vector<Ty>, hy2, (N, incy, batch_count));
    HOST_MEMCHECK(host_batch_vector<Tex>, hy_ex, (N, incy, batch_count));
    HOST_MEMCHECK(host_batch_vector<Tex>, hx_ex, (N, incx, batch_count));
    HOST_MEMCHECK(host_vector<Ta>, halpha, (1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_vector<Tx>, dx, (N, incx, batch_count));
    DEVICE_MEMCHECK(device_batch_vector<Ty>, dy, (N, incy, batch_count));
    DEVICE_MEMCHECK(device_vector<Ta>, dalpha, (1));

    // Assign host alpha.
    halpha[0] = h_alpha;

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);

    for(int64_t b = 0; b < batch_count; b++)
    {
        for(size_t i = 0, idx = 0; i < N; i++, idx += abs_incy)
            hy_ex[b][idx] = (Tex)hy[b][idx];
        for(size_t i = 0, idx = 0; i < N; i++, idx += abs_incx)
            hx_ex[b][idx] = (Tex)hx[b][idx];
    }

    // Device memory.
    double cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        // Transfer host to device
        CHECK_HIP_ERROR(dx.transfer_from(hx));

        // Call routine with pointer mode on host.

        // Pointer mode.
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            // Call routine.
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_axpy_batched_ex_fn,
                       (handle,
                        N,
                        halpha,
                        alpha_type,
                        dx.ptr_on_device(),
                        x_type,
                        incx,
                        dy.ptr_on_device(),
                        y_type,
                        incy,
                        batch_count,
                        execution_type));
            handle.post_test(arg);
            // Transfer from device to host.
            CHECK_HIP_ERROR(hy1.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            // Call routine.
            CHECK_HIP_ERROR(dalpha.transfer_from(halpha));
            CHECK_HIP_ERROR(dy.transfer_from(hy));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_axpy_batched_ex_fn,
                       (handle,
                        N,
                        dalpha,
                        alpha_type,
                        dx.ptr_on_device(),
                        x_type,
                        incx,
                        dy.ptr_on_device(),
                        y_type,
                        incy,
                        batch_count,
                        execution_type));
            handle.post_test(arg);

            // Transfer from device to host.
            CHECK_HIP_ERROR(hy2.transfer_from(dy));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_batch_vector<Ty>, hy_copy, (N, incy, batch_count));

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
                    DEVICE_MEMCHECK(device_batch_vector<Tx>, dx_copy, (N, incx, batch_count));
                    DEVICE_MEMCHECK(device_batch_vector<Ty>, dy_copy, (N, incy, batch_count));
                    DEVICE_MEMCHECK(device_vector<Ta>, dalpha_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(dalpha_copy.transfer_from(halpha));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy));

                        DAPI_CHECK(rocblas_axpy_batched_ex_fn,
                                   (handle_copy,
                                    N,
                                    dalpha_copy,
                                    alpha_type,
                                    dx_copy.ptr_on_device(),
                                    x_type,
                                    incx,
                                    dy_copy.ptr_on_device(),
                                    y_type,
                                    incy,
                                    batch_count,
                                    execution_type));

                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));

                        unit_check_general<Ty>(1, N, incy, hy2, hy_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        {
            cpu_time_used = get_time_us_no_sync();

            // Compute the host solution.
            for(int64_t b = 0; b < batch_count; ++b)
            {
                ref_axpy<Tex>(N, h_alpha_ex, hx_ex[b], incx, hy_ex[b], incy);
            }
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;

            for(int64_t b = 0; b < batch_count; b++)
            {
                for(size_t i = 0, idx = 0; i < N; i++, idx += abs_incy)
                    hy[b][idx] = (Ty)hy_ex[b][idx];
            }
        }

        if(arg.pointer_mode_host)
        {
            // Compare with with hsolution.
            if(arg.unit_check)
            {
                unit_check_general<Ty>(1, N, incy, hy, hy1, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<Ty>('I', 1, N, incy, hy, hy1, batch_count);
            }
        }
        if(arg.pointer_mode_device)
        {
            // Compare with with hsolution.
            if(arg.unit_check)
            {

                unit_check_general<Ty>(1, N, incy, hy, hy2, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_2 = norm_check_general<Ty>('I', 1, N, incy, hy, hy2, batch_count);
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        // Transfer from host to device.
        CHECK_HIP_ERROR(dy.transfer_from(hy));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_axpy_batched_ex_fn,
                          (handle,
                           N,
                           &h_alpha,
                           alpha_type,
                           dx.ptr_on_device(),
                           x_type,
                           incx,
                           dy.ptr_on_device(),
                           y_type,
                           incy,
                           batch_count,
                           execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy, e_batch_count>{}.log_args<Ta>(
            rocblas_cout,
            arg,
            gpu_time_used,
            axpy_gflop_count<Ta>(N),
            axpy_gbyte_count<Ta>(N),
            cpu_time_used,
            rocblas_error_1,
            rocblas_error_2);
    }
}
