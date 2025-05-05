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

#include "cblas_interface.hpp"
#include "client_utility.hpp"
#include "flops.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"

#include "blas3/rocblas_trtri.hpp"

template <typename T>
void testing_trtri_bad_arg(const Arguments& arg)
{
    auto rocblas_trtri_fn
        = arg.api & c_API_FORTRAN ? rocblas_trtri<T, true> : rocblas_trtri<T, false>;

    rocblas_local_handle handle{arg};

    const rocblas_int N   = 100;
    const rocblas_int lda = 100;

    const rocblas_fill     uplo = rocblas_fill_upper;
    const rocblas_diagonal diag = rocblas_diagonal_non_unit;

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (N, N, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dinvA, (N, N, lda));

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, rocblas_fill_full, diag, N, dA, lda, dinvA, lda),
                          rocblas_status_invalid_value);

    EXPECT_ROCBLAS_STATUS(
        rocblas_trtri_fn(handle, uplo, (rocblas_diagonal)rocblas_side_both, N, dA, lda, dinvA, lda),
        rocblas_status_invalid_value);

    // check for invalid sizes
    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, uplo, diag, -1, dA, lda, dinvA, lda),
                          rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, uplo, diag, N, dA, lda - 1, dinvA, lda),
                          rocblas_status_invalid_size);

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, uplo, diag, N, dA, lda, dinvA, lda - 1),
                          rocblas_status_invalid_size);

    // nullptr tests
    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(nullptr, uplo, diag, N, dA, lda, dinvA, lda),
                          rocblas_status_invalid_handle);

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, uplo, diag, N, nullptr, lda, dinvA, lda),
                          rocblas_status_invalid_pointer);

    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, uplo, diag, N, dA, lda, nullptr, lda),
                          rocblas_status_invalid_pointer);

    // quick return: If N==0, then all pointers can be nullptr without error
    EXPECT_ROCBLAS_STATUS(rocblas_trtri_fn(handle, uplo, diag, 0, nullptr, lda, nullptr, lda),
                          rocblas_status_success);
}

template <typename T>
void testing_trtri(const Arguments& arg)
{
    auto rocblas_trtri_fn
        = arg.api & c_API_FORTRAN ? rocblas_trtri<T, true> : rocblas_trtri<T, false>;

    rocblas_int N = arg.N;
    rocblas_int lda;
    rocblas_int ldinvA;
    ldinvA = lda = arg.lda;

    size_t size_A = size_t(lda) * N;

    char char_uplo = arg.uplo;
    char char_diag = arg.diag;

    rocblas_fill     uplo = char2rocblas_fill(char_uplo);
    rocblas_diagonal diag = char2rocblas_diagonal(char_diag);

    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = lda < 0 || lda < N;
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocblas_trtri_fn(handle, uplo, diag, N, nullptr, lda, nullptr, ldinvA),
            rocblas_status_invalid_size);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (N, N, lda));
    HOST_MEMCHECK(host_matrix<T>, hB, (N, N, lda));

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (N, N, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dinvA, (N, N, lda));

    // Initial Data on CPU
    //Explicitly set the unused side of matrix `hA` to 0 when using it for temp storage.
    //Used rocblas_client_triangular_matrix type initialization, which will ensure the unused side is set to 0 or could be done manually
    rocblas_init_matrix(
        hA, arg, rocblas_client_never_set_nan, rocblas_client_triangular_matrix, true);

    for(size_t i = 0; i < N; i++)
    {
        for(size_t j = 0; j < N; j++)
        {
            T* A = (T*)hA;
            A[i + j * lda] *= 0.01;

            if(j % 2)
                A[i + j * lda] *= -1;
            if(i == j)
            {
                if(diag == rocblas_diagonal_unit)
                    A[i + j * lda] = 1.0; // need to preprocess matrix for clbas_trtri
                else
                    A[i + j * lda] *= 100.0;
            }
        }
    }

    hB = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dinvA.transfer_from(hA));

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error_out, rocblas_error_in;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(arg.unit_check || arg.norm_check)
    {
        handle.pre_test(arg);
        if(arg.api != INTERNAL)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trtri_fn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));
            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_matrix<T>, hA_copy, (N, N, lda));
                CHECK_HIP_ERROR(hA.transfer_from(dinvA));
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
                    DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (N, N, lda));
                    DEVICE_MEMCHECK(device_matrix<T>, dinvA_copy, (N, N, lda));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dA_copy.transfer_from(hB));
                        CHECK_HIP_ERROR(dinvA_copy.transfer_from(hB));
                        CHECK_ROCBLAS_ERROR(rocblas_trtri_fn(
                            handle_copy, uplo, diag, N, dA_copy, lda, dinvA_copy, ldinvA));
                        CHECK_HIP_ERROR(hA_copy.transfer_from(dinvA_copy));
                        unit_check_general<T>(N, N, lda, hA, hA_copy);
                    }
                }
                return;
            }
        }
        else
        {
            rocblas_stride offsetA         = arg.ldc;
            rocblas_stride offsetinvA      = arg.ldd;
            rocblas_stride strideA         = arg.stride_a;
            rocblas_stride strideinvA      = arg.stride_b;
            rocblas_stride subStride       = 0;
            rocblas_int    batch_count     = 1;
            rocblas_int    sub_batch_count = 1;

            size_t work_el = rocblas_internal_trtri_temp_elements(N, 1);
            DEVICE_MEMCHECK(device_vector<T>, workspace, (work_el));

            // Note: sub_strides and sub_batch_count don't seem to be used anywhere in rocBLAS or rocSOLVER,
            //       and should be able to be removed in a future release. They are not tested.
            CHECK_ROCBLAS_ERROR(rocblas_internal_trtri_template(handle,
                                                                uplo,
                                                                diag,
                                                                N,
                                                                (const T*)dA + offsetA,
                                                                -offsetA,
                                                                lda,
                                                                strideA,
                                                                subStride,
                                                                (T*)dinvA + offsetinvA,
                                                                -offsetinvA,
                                                                ldinvA,
                                                                strideinvA,
                                                                subStride,
                                                                batch_count,
                                                                sub_batch_count,
                                                                (T*)workspace));
        }

        // test in-place
        handle.post_test(arg);
        if(arg.api != INTERNAL)
        {
            CHECK_ROCBLAS_ERROR(rocblas_trtri_fn(handle, uplo, diag, N, dA, lda, dA, lda));
        }
        else
        {
            rocblas_stride offsetA         = arg.ldc;
            rocblas_stride strideA         = arg.stride_a;
            rocblas_stride subStride       = 0;
            rocblas_int    batch_count     = 1;
            rocblas_int    sub_batch_count = 1;

            size_t work_el = rocblas_internal_trtri_temp_elements(N, 1);
            DEVICE_MEMCHECK(device_vector<T>, workspace, (work_el));

            CHECK_ROCBLAS_ERROR(rocblas_internal_trtri_template(handle,
                                                                uplo,
                                                                diag,
                                                                N,
                                                                (const T*)dA + offsetA,
                                                                -offsetA,
                                                                lda,
                                                                strideA,
                                                                subStride,
                                                                (T*)dA + offsetA,
                                                                -offsetA,
                                                                lda,
                                                                strideA,
                                                                subStride,
                                                                batch_count,
                                                                sub_batch_count,
                                                                (T*)workspace));
        }

        // copy output from device to CPU
        CHECK_HIP_ERROR(hA.transfer_from(dinvA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        if(arg.timing)
            cpu_time_used = get_time_us_no_sync();

        // CBLAS doesn't have trtri implementation so using the LAPACK trtri
        ref_lapack_xtrtri<T>(char_uplo, char_diag, N, hB, lda);

        if(arg.timing)
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // test out-of-place
        const double rel_error = trtri_tolerance<T>(N);
        if(arg.unit_check)
            near_check_general<T>(N, N, lda, hB, hA, rel_error);

        if(arg.norm_check)
            rocblas_error_out = norm_check_symmetric<T>('F', char_uplo, N, lda, hB, hA);

        // test in-place
        CHECK_HIP_ERROR(hA.transfer_from(dA));
        if(arg.unit_check)
            near_check_general<T>(N, N, lda, hB, hA, rel_error);

        if(arg.norm_check)
            rocblas_error_in = norm_check_symmetric<T>('F', char_uplo, N, lda, hB, hA);
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        for(int i = 0; i < total_calls; i++)
        {
            if(i == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream);

            rocblas_trtri_fn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA);
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        ArgumentModel<e_uplo, e_diag, e_N, e_lda>{}.log_args<T>(rocblas_cout,
                                                                arg,
                                                                gpu_time_used,
                                                                trtri_gflop_count<T>(N),
                                                                ArgumentLogging::NA_value,
                                                                cpu_time_used,
                                                                rocblas_error_out,
                                                                rocblas_error_in);
    }
}
