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
void testing_set_get_vector_async(const Arguments& arg)
{
    auto rocblas_set_vector_async_fn    = rocblas_set_vector_async;
    auto rocblas_set_vector_async_fn_64 = rocblas_set_vector_async_64;
    auto rocblas_get_vector_async_fn    = rocblas_get_vector_async;
    auto rocblas_get_vector_async_fn_64 = rocblas_get_vector_async_64;

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;
    int64_t ldd  = arg.ldd;

    rocblas_local_handle handle{arg};

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory

    if(arg.algo == 1) // use arg.algo == 1 to test bad device pointer
    {
#ifndef ASAN_BUILD
        T  host_data;
        T* host_data_ptr = &host_data;

        rocblas_status status
            = rocblas_set_vector_async_fn(1, sizeof(T), host_data_ptr, 1, host_data_ptr, 1, stream);
        hipError_t h_error = hipGetLastError(); // clear HIP error
        GTEST_ASSERT_TRUE(rocblas_status_internal_error == status);

        status
            = rocblas_get_vector_async_fn(1, sizeof(T), host_data_ptr, 1, host_data_ptr, 1, stream);
        h_error = hipGetLastError(); // clear HIP error
        GTEST_ASSERT_TRUE(rocblas_status_internal_error == status);
#else
        GTEST_SKIP() << "ASAN_BUILD";
#endif
        return;
    }
    else if(N <= 0 || incx <= 0 || incy <= 0 || ldd <= 0)
    {
        rocblas_status expected_status
            = N == 0 ? rocblas_status_success : rocblas_status_invalid_size;
        DAPI_EXPECT(expected_status,
                    rocblas_set_vector_async_fn,
                    (N, sizeof(T), nullptr, incx, nullptr, ldd, stream));
        DAPI_EXPECT(expected_status,
                    rocblas_get_vector_async_fn,
                    (N, sizeof(T), nullptr, ldd, nullptr, incy, stream));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    HOST_MEMCHECK(host_pinned_vector<T>, hx, (N, incx));
    HOST_MEMCHECK(host_pinned_vector<T>, hy, (N, incy));
    HOST_MEMCHECK(host_vector<T>, hy_gold, (N, incy));

    double cpu_time_used;
    cpu_time_used        = 0.0;
    double rocblas_error = 0.0;

    // allocate memory on device
    DEVICE_MEMCHECK(device_vector<T>, db, (N, ldd));

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hx, 1, N, incx);
    rocblas_init<T>(hy, 1, N, incy);

    if(arg.unit_check || arg.norm_check)
    {
        // set device memory to be zero
        CHECK_HIP_ERROR(hipMemsetAsync(db, 0, sizeof(T) * (1 + size_t(ldd) * (N - 1)), stream));

        DAPI_CHECK(rocblas_set_vector_async_fn, (N, sizeof(T), hx, incx, db, ldd, stream));
        DAPI_CHECK(rocblas_get_vector_async_fn, (N, sizeof(T), db, ldd, hy, incy, stream));

        cpu_time_used = get_time_us_no_sync();

        // reference calculation
        for(size_t i = 0; i < N; i++)
        {
            hy_gold[i * incy] = hx[i * incx];
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        hipStreamSynchronize(stream);

        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, incy, hy, hy_gold);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, N, incy, hy, hy_gold);
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

            DAPI_DISPATCH(rocblas_set_vector_async_fn, (N, sizeof(T), hx, incx, db, ldd, stream));
            DAPI_DISPATCH(rocblas_get_vector_async_fn, (N, sizeof(T), db, ldd, hy, incy, stream));
        }

        hipStreamSynchronize(stream);
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_ldd>{}.log_args<T>(rocblas_cout,
                                                                arg,
                                                                gpu_time_used,
                                                                ArgumentLogging::NA_value,
                                                                set_get_vector_gbyte_count<T>(N),
                                                                cpu_time_used,
                                                                rocblas_error);
    }
}
