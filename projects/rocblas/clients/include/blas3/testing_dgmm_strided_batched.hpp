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

template <typename T>
void testing_dgmm_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_dgmm_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_dgmm_strided_batched<T, true>
                                               : rocblas_dgmm_strided_batched<T, false>;

    auto rocblas_dgmm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_dgmm_strided_batched_64<T, true>
                                                  : rocblas_dgmm_strided_batched_64<T, false>;

    const int64_t M = 100;
    const int64_t N = 101;

    const int64_t lda  = 100;
    const int64_t incx = 1;
    const int64_t ldc  = 100;

    const int64_t      batch_count = 2;
    const rocblas_side side        = rocblas_side_left;

    // no device/host loop required as no difference
    rocblas_local_handle handle{arg};

    rocblas_int K = rocblas_side_right == side ? N : M;

    const rocblas_stride stride_a = N * size_t(lda);
    const rocblas_stride stride_x = K * size_t(incx);
    const rocblas_stride stride_c = N * size_t(ldc);

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (M, N, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (K, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (M, N, ldc, stride_a, batch_count));

    DAPI_EXPECT(rocblas_status_invalid_handle,
                rocblas_dgmm_strided_batched_fn,
                (nullptr,
                 side,
                 M,
                 N,
                 dA,
                 lda,
                 stride_a,
                 dx,
                 incx,
                 stride_x,
                 dC,
                 ldc,
                 stride_c,
                 batch_count));

    DAPI_EXPECT(rocblas_status_invalid_value,
                rocblas_dgmm_strided_batched_fn,
                (handle,
                 (rocblas_side)rocblas_fill_full,
                 M,
                 N,
                 dA,
                 lda,
                 stride_a,
                 dx,
                 incx,
                 stride_x,
                 dC,
                 ldc,
                 stride_c,
                 batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_strided_batched_fn,
                (handle,
                 side,
                 M,
                 N,
                 nullptr,
                 lda,
                 stride_a,
                 dx,
                 incx,
                 stride_x,
                 dC,
                 ldc,
                 stride_c,
                 batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_strided_batched_fn,
                (handle,
                 side,
                 M,
                 N,
                 dA,
                 lda,
                 stride_a,
                 nullptr,
                 incx,
                 stride_x,
                 dC,
                 ldc,
                 stride_c,
                 batch_count));

    DAPI_EXPECT(rocblas_status_invalid_pointer,
                rocblas_dgmm_strided_batched_fn,
                (handle,
                 side,
                 M,
                 N,
                 dA,
                 lda,
                 stride_a,
                 dx,
                 incx,
                 stride_x,
                 nullptr,
                 ldc,
                 stride_c,
                 batch_count));
}

template <typename T>
void testing_dgmm_strided_batched(const Arguments& arg)
{
    auto rocblas_dgmm_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_dgmm_strided_batched<T, true>
                                               : rocblas_dgmm_strided_batched<T, false>;

    auto rocblas_dgmm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_dgmm_strided_batched_64<T, true>
                                                  : rocblas_dgmm_strided_batched_64<T, false>;

    rocblas_side side = char2rocblas_side(arg.side);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = rocblas_side_right == side ? N : M;

    int64_t lda         = arg.lda;
    int64_t incx        = arg.incx;
    int64_t ldc         = arg.ldc;
    int64_t batch_count = arg.batch_count;

    rocblas_stride stride_a = arg.stride_a;
    rocblas_stride stride_x = arg.stride_x;
    if(!stride_x)
        stride_x = 1; // incx = 0 case
    rocblas_stride stride_c = arg.stride_c;

    int64_t abs_incx = incx > 0 ? incx : -incx;

    double gpu_time_used, cpu_time_used;

    double rocblas_error = std::numeric_limits<double>::max();

    if((stride_a > 0) && (stride_a < int64_t(lda) * N))
    {
        rocblas_cout << "WARNING: stride_a < lda * N, setting stride_a = lda * N " << std::endl;
        stride_a = N * int64_t(lda);
    }
    if((stride_c > 0) && (stride_c < int64_t(ldc) * N))
    {
        rocblas_cout << "WARNING: stride_c < ldc * N, setting stride_c = ldc * N" << std::endl;
        stride_c = N * int64_t(ldc);
    }
    if((stride_x > 0) && (stride_x < int64_t(abs_incx) * K))
    {
        rocblas_cout << "WARNING: stride_x < incx * (rocblas_side_right == side ? N : M)),\n"
                        "setting stride_x = incx * (rocblas_side_right == side ? N : M))"
                     << std::endl;
        stride_x = K * int64_t(abs_incx);
    }

    rocblas_local_handle handle{arg};

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0;
    if(invalid_size || M == 0 || N == 0 || batch_count == 0)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_dgmm_strided_batched_fn,
                    (handle,
                     side,
                     M,
                     N,
                     nullptr,
                     lda,
                     stride_a,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     ldc,
                     stride_c,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (M, N, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_vector<T>, hx, (K, incx, stride_x, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC, (M, N, ldc, stride_c, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC_gold, (M, N, ldc, stride_c, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (M, N, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_vector<T>, dx, (K, incx, stride_x, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (M, N, ldc, stride_c, batch_count));

    // Initialize data on host memory
    rocblas_init_matrix(hA, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
    rocblas_init_vector(hx, arg, rocblas_client_never_set_nan, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        handle.pre_test(arg);
        DAPI_CHECK(rocblas_dgmm_strided_batched_fn,
                   (handle,
                    side,
                    M,
                    N,
                    dA,
                    lda,
                    stride_a,
                    dx,
                    incx,
                    stride_x,
                    dC,
                    ldc,
                    stride_c,
                    batch_count));
        handle.post_test(arg);

        // fetch GPU result
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        if(arg.repeatability_check)
        {
            HOST_MEMCHECK(
                host_strided_batch_matrix<T>, hC_copy, (M, N, ldc, stride_c, batch_count));
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
                DEVICE_MEMCHECK(
                    device_strided_batch_matrix<T>, dA_copy, (M, N, lda, stride_a, batch_count));
                DEVICE_MEMCHECK(
                    device_strided_batch_vector<T>, dx_copy, (K, incx, stride_x, batch_count));
                DEVICE_MEMCHECK(
                    device_strided_batch_matrix<T>, dC_copy, (M, N, ldc, stride_c, batch_count));

                // copy data from CPU to device
                CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                CHECK_HIP_ERROR(dx_copy.transfer_from(hx));
                CHECK_HIP_ERROR(dC_copy.transfer_from(hC));

                CHECK_ROCBLAS_ERROR(
                    rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                for(int runs = 0; runs < arg.iters; runs++)
                {
                    DAPI_CHECK(rocblas_dgmm_strided_batched_fn,
                               (handle_copy,
                                side,
                                M,
                                N,
                                dA_copy,
                                lda,
                                stride_a,
                                dx_copy,
                                incx,
                                stride_x,
                                dC_copy,
                                ldc,
                                stride_c,
                                batch_count));
                    // fetch GPU result
                    CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                    unit_check_general<T>(M, N, ldc, stride_c, hC, hC_copy, batch_count);
                }
            }
            return;
        }

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(size_t b = 0; b < batch_count; b++)
            ref_dgmm<T>(side, M, N, hA[b], lda, hx[b], incx, hC_gold[b], ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
        }

        if(arg.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count);
        }

    } // end of if unit/norm check

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

            DAPI_DISPATCH(rocblas_dgmm_strided_batched_fn,
                          (handle,
                           side,
                           M,
                           N,
                           dA,
                           lda,
                           stride_a,
                           dx,
                           incx,
                           stride_x,
                           dC,
                           ldc,
                           stride_c,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_side,
                      e_M,
                      e_N,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         dgmm_gflop_count<T>(M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error);
    }
}
