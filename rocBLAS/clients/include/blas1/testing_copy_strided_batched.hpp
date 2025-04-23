/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
void testing_copy_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_copy_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_copy_strided_batched<T, true>
                                                  : rocblas_copy_strided_batched<T, false>;
    auto rocblas_copy_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_copy_strided_batched_64<T, true>
                                                  : rocblas_copy_strided_batched_64<T, false>;

    rocblas_local_handle handle{arg};

    int64_t        N           = 100;
    int64_t        incx        = 1;
    int64_t        incy        = 1;
    rocblas_stride stride_x    = incx * N;
    rocblas_stride stride_y    = incy * N;
    int64_t        batch_count = 2;

    size_t size_x = stride_x * batch_count;
    size_t size_y = stride_y * batch_count;

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stride_y, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_copy_strided_batched_fn,
                (nullptr, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_copy_strided_batched_fn,
                (handle, N, nullptr, incx, stride_x, dy, incy, stride_y, batch_count));
    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_copy_strided_batched_fn,
                (handle, N, dx, incx, stride_x, nullptr, incy, stride_y, batch_count));
}

template <typename T>
void testing_copy_strided_batched(const Arguments& arg)
{
    auto rocblas_copy_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_copy_strided_batched<T, true>
                                                  : rocblas_copy_strided_batched<T, false>;
    auto rocblas_copy_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_copy_strided_batched_64<T, true>
                                                  : rocblas_copy_strided_batched_64<T, false>;

    int64_t N           = arg.N;
    int64_t incx        = arg.incx;
    int64_t incy        = arg.incy;
    int64_t stride_x    = arg.stride_x;
    int64_t stride_y    = arg.stride_y;
    int64_t batch_count = arg.batch_count;

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(rocblas_copy_strided_batched_fn,
                   (handle, N, nullptr, incx, stride_x, nullptr, incy, stride_y, batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hx), `d` is in GPU (device) memory (eg dx).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (N, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy, (N, incy, stride_y, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hy_gold, (N, incy, stride_y, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (N, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dy, (N, incy, stride_y, batch_count));

    // Initialize data on host memory
    rocblas_init_vector(hx, arg, rocblas_client_alpha_sets_nan, true);
    rocblas_init_vector(hy, arg, rocblas_client_alpha_sets_nan, false);

    hy_gold.copy_from(hy);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    double cpu_time_used;
    double rocblas_error = 0.0;

    if(arg.unit_check || arg.norm_check)
    {
        handle.pre_test(arg);
        // GPU BLAS
        DAPI_CHECK(rocblas_copy_strided_batched_fn,
                   (handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        handle.post_test(arg);

        CHECK_HIP_ERROR(hy.transfer_from(dy));

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < batch_count; ++b)
        {
            ref_copy<T>(N, hx[b], incx, hy_gold[b], incy);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incy, stride_y, hy_gold, hy, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, N, incy, stride_y, hy_gold, hy, batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(rocblas_copy_strided_batched_fn,
                          (handle, N, dx, incx, stride_x, dy, incy, stride_y, batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            copy_gbyte_count<T>(N),
            cpu_time_used,
            rocblas_error);
    }
}
