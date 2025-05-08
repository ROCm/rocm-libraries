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
void testing_set_get_matrix(const Arguments& arg)
{
    auto rocblas_set_matrix_fn    = rocblas_set_matrix;
    auto rocblas_set_matrix_fn_64 = rocblas_set_matrix_64;
    auto rocblas_get_matrix_fn    = rocblas_get_matrix;
    auto rocblas_get_matrix_fn_64 = rocblas_get_matrix_64;

    int64_t rows = arg.M;
    int64_t cols = arg.N;
    int64_t lda  = arg.lda;
    int64_t ldb  = arg.ldb;
    int64_t ldd  = arg.ldd;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalidGPUMatrix = rows < 0 || cols < 0 || ldd <= 0 || ldd < rows;
    bool invalidSet       = invalidGPUMatrix || lda <= 0 || lda < rows;
    bool invalidGet       = invalidGPUMatrix || ldb <= 0 || ldb < rows;

    if(arg.algo == 1) // use arg.algo == 1 to test bad device pointer
    {
#ifndef ASAN_BUILD
        T  host_data;
        T* host_data_ptr = &host_data;

        rocblas_status status
            = rocblas_get_matrix_fn(1, 1, sizeof(T), host_data_ptr, 1, host_data_ptr, 1);
        hipError_t h_error = hipGetLastError(); // clear HIP error
        GTEST_ASSERT_TRUE(rocblas_status_internal_error == status);

        status  = rocblas_set_matrix_fn(1, 1, sizeof(T), host_data_ptr, 1, host_data_ptr, 1);
        h_error = hipGetLastError(); // clear HIP error
        GTEST_ASSERT_TRUE(rocblas_status_internal_error == status);
#else
        GTEST_SKIP() << "ASAN_BUILD";
#endif
        return;
    }
    else if(rows == 0 || cols == 0)
    {
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_set_matrix_fn,
                    (rows, cols, sizeof(T), nullptr, lda, nullptr, ldd));
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_get_matrix_fn,
                    (rows, cols, sizeof(T), nullptr, lda, nullptr, ldd));
        return;
    }
    else if(invalidSet || invalidGet)
    {
        DAPI_EXPECT(invalidSet ? rocblas_status_invalid_size : rocblas_status_invalid_pointer,
                    rocblas_set_matrix_fn,
                    (rows, cols, sizeof(T), nullptr, lda, nullptr, ldd));

        DAPI_EXPECT(invalidGet ? rocblas_status_invalid_size : rocblas_status_invalid_pointer,
                    rocblas_get_matrix_fn,
                    (rows, cols, sizeof(T), nullptr, ldd, nullptr, ldb));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    HOST_MEMCHECK(host_matrix<T>, hA, (rows, cols, lda));
    HOST_MEMCHECK(host_matrix<T>, hB, (rows, cols, ldb));
    HOST_MEMCHECK(host_matrix<T>, hB_gold, (rows, cols, ldb));

    double cpu_time_used;
    double rocblas_error = 0.0;

    // allocate memory on device
    DEVICE_MEMCHECK(device_matrix<T>, dD, (rows, cols, ldd));

    // Initial Data on CPU
    rocblas_seedrand();
    rocblas_init<T>(hA, rows, cols, lda);
    rocblas_init<T>(hB, rows, cols, ldb);

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        CHECK_HIP_ERROR(hipMemset(dD, 0, sizeof(T) * ldd * cols));

        DAPI_CHECK(rocblas_set_matrix_fn, (rows, cols, sizeof(T), hA, lda, dD, ldd));
        DAPI_CHECK(rocblas_get_matrix_fn, (rows, cols, sizeof(T), dD, ldd, hB, ldb));

        // reference calculation
        cpu_time_used = get_time_us_no_sync();

        for(size_t i1 = 0; i1 < rows; i1++)
            for(size_t i2 = 0; i2 < cols; i2++)
                *(hB_gold + i1 + i2 * ldb) = *(hA + i1 + i2 * lda);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(rows, cols, ldb, hB, hB_gold);
        }

        if(arg.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', rows, cols, ldb, (T*)hB, hB_gold);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        for(int iter = 0; iter < total_calls; iter++)
        {
            if(iter == number_cold_calls)
                gpu_time_used = get_time_us_no_sync();

            DAPI_DISPATCH(rocblas_set_matrix_fn, (rows, cols, sizeof(T), hA, lda, dD, ldd));
            DAPI_DISPATCH(rocblas_get_matrix_fn, (rows, cols, sizeof(T), dD, ldd, hB, ldb));
        }

        gpu_time_used = get_time_us_no_sync() - gpu_time_used;

        ArgumentModel<e_M, e_N, e_lda, e_ldb, e_ldd>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            ArgumentLogging::NA_value,
            set_get_matrix_gbyte_count<T>(rows, cols),
            cpu_time_used,
            rocblas_error);
    }
}
