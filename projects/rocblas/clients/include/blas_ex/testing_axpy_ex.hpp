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
void testing_axpy_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_axpy_ex_fn = arg.api & c_API_FORTRAN ? rocblas_axpy_ex_fortran : rocblas_axpy_ex;
    auto rocblas_axpy_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_axpy_ex_64_fortran : rocblas_axpy_ex_64;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        rocblas_datatype alpha_type     = rocblas_type2datatype<Ta>();
        rocblas_datatype x_type         = rocblas_type2datatype<Tx>();
        rocblas_datatype y_type         = rocblas_type2datatype<Ty>();
        rocblas_datatype execution_type = rocblas_type2datatype<Tex>();

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

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

        DEVICE_MEMCHECK(device_vector<Tx>, dx, (N));
        DEVICE_MEMCHECK(device_vector<Ty>, dy, (N));

        DAPI_EXPECT(
            rocblas_status_invalid_handle,
            rocblas_axpy_ex_fn,
            (nullptr, N, alpha, alpha_type, dx, x_type, incx, dy, y_type, incy, execution_type));

        DAPI_EXPECT(
            rocblas_status_invalid_pointer,
            rocblas_axpy_ex_fn,
            (handle, N, nullptr, alpha_type, dx, x_type, incx, dy, y_type, incy, execution_type));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_axpy_ex_fn,
                        (handle,
                         N,
                         alpha,
                         alpha_type,
                         nullptr,
                         x_type,
                         incx,
                         dy,
                         y_type,
                         incy,
                         execution_type));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_axpy_ex_fn,
                        (handle,
                         N,
                         alpha,
                         alpha_type,
                         dx,
                         x_type,
                         incx,
                         nullptr,
                         y_type,
                         incy,
                         execution_type));
        }

        // If N == 0, then X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_ex_fn,
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
                     execution_type));

        // If alpha == 0, then X and Y can be nullptr without error
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_axpy_ex_fn,
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
                     execution_type));
    }
}

template <typename Ta, typename Tx = Ta, typename Ty = Tx, typename Tex = Ty>
void testing_axpy_ex(const Arguments& arg)
{
    auto rocblas_axpy_ex_fn = arg.api & c_API_FORTRAN ? rocblas_axpy_ex_fortran : rocblas_axpy_ex;
    auto rocblas_axpy_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_axpy_ex_64_fortran : rocblas_axpy_ex_64;

    rocblas_datatype alpha_type     = arg.a_type;
    rocblas_datatype x_type         = arg.b_type;
    rocblas_datatype y_type         = arg.c_type;
    rocblas_datatype execution_type = arg.compute_type;

    int64_t              N       = arg.N;
    int64_t              incx    = arg.incx;
    int64_t              incy    = arg.incy;
    Ta                   h_alpha = arg.get_alpha<Ta>();
    rocblas_local_handle handle{arg};

    bool special_compute_test = N == 1 && h_alpha == -1.001;

    // argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        DAPI_CHECK(rocblas_axpy_ex_fn,
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
                    execution_type));
        return;
    }

    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_vector<Tx>, hx, (N, incx));
    HOST_MEMCHECK(host_vector<Tex>, hx_ex, (N, incx));
    HOST_MEMCHECK(host_vector<Ty>, hy, (N, incy));
    HOST_MEMCHECK(host_vector<Ty>, hy_gold, (N, incy));
    HOST_MEMCHECK(host_vector<Tex>, hy_gold_ex, (N, incy));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tx>, dx, (N, incx));
    DEVICE_MEMCHECK(device_vector<Ty>, dy, (N, incy));
    DEVICE_MEMCHECK(device_vector<Ta>, d_alpha, (1));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false, true);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    // BLAS
    hy_gold = hy;

    for(size_t i = 0, idx = 0; i < N; i++, idx += abs_incy)
        hy_gold_ex[idx] = (Tex)hy_gold[idx];

    for(size_t i = 0, idx = 0; i < N; i++, idx += abs_incx)
        hx_ex[idx] = (Tex)hx[idx];

    Tex h_alpha_ex = (Tex)h_alpha;

    // This is to test that we are using the correct
    // compute type (avoiding overflow in this case)
    if(special_compute_test)
    {
        // max half value
        hx[0]   = (Tx)65504;
        hy[0]   = (Ty)65504;
        hy_gold = hy;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;
    double rocblas_error_1 = 0.0;
    double rocblas_error_2 = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_HIP_ERROR(dy.transfer_from(hy));

            handle.pre_test(arg);
            // ROCBLAS pointer mode host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            DAPI_CHECK(rocblas_axpy_ex_fn,
                       (handle,
                        N,
                        &h_alpha,
                        alpha_type,
                        dx,
                        x_type,
                        incx,
                        dy,
                        y_type,
                        incy,
                        execution_type));
            handle.post_test(arg);

            // copy output from device to CPU
            CHECK_HIP_ERROR(hy.transfer_from(dy));
        }

        if(arg.pointer_mode_device)
        {
            // ROCBLAS pointer mode device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dy.transfer_from(hy_gold)); // hy_gold not computed yet so still hy
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_axpy_ex_fn,
                       (handle,
                        N,
                        d_alpha,
                        alpha_type,
                        dx,
                        x_type,
                        incx,
                        dy,
                        y_type,
                        incy,
                        execution_type));
            handle.post_test(arg);

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_vector<Ty>, hy_copy, (N, incy));
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
                    DEVICE_MEMCHECK(device_vector<Ta>, dalpha_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                    CHECK_HIP_ERROR(
                        hipMemcpy(dalpha_copy, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dy_copy.transfer_from(hy_gold));
                        DAPI_CHECK(rocblas_axpy_ex_fn,
                                   (handle_copy,
                                    N,
                                    dalpha_copy,
                                    alpha_type,
                                    dx_copy,
                                    x_type,
                                    incx,
                                    dy_copy,
                                    y_type,
                                    incy,
                                    execution_type));
                        CHECK_HIP_ERROR(hy_copy.transfer_from(dy_copy));
                        unit_check_general<Ty>(1, N, incy, hy, hy_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        // ref_axpy<Ta>(N, h_alpha, hx, incx, hy_gold, incy);
        ref_axpy<Tex>(N, h_alpha_ex, hx_ex, incx, hy_gold_ex, incy);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        for(size_t i = 0, idx = 0; i < N; i++, idx += abs_incy)
            hy_gold[idx] = (Ty)hy_gold_ex[idx];

        if(special_compute_test)
            hy_gold[0] = Ty(Tex(h_alpha + 1) * Tex(65504));

        // No accumulation in axpy, hard to check if we're using the right
        // compute_type

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<Ty>(1, N, incy, hy_gold, hy);
            }

            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<Ty>('F', 1, N, incy, hy_gold, hy);
            }
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hy.transfer_from(dy));

            if(arg.unit_check)
            {
                unit_check_general<Ty>(1, N, incy, hy_gold, hy);
            }

            if(arg.norm_check)
            {
                rocblas_error_2 = norm_check_general<Ty>('F', 1, N, incy, hy_gold, hy);
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

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_axpy_ex_fn,
                          (handle,
                           N,
                           &h_alpha,
                           alpha_type,
                           dx,
                           x_type,
                           incx,
                           dy,
                           y_type,
                           incy,
                           execution_type));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_incy>{}.log_args<Ta>(rocblas_cout,
                                                                   arg,
                                                                   gpu_time_used,
                                                                   axpy_gflop_count<Ta>(N),
                                                                   axpy_gbyte_count<Ta>(N),
                                                                   cpu_time_used,
                                                                   rocblas_error_1,
                                                                   rocblas_error_2);
    }
}
