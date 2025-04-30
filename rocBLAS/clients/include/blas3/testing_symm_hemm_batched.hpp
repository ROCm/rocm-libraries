/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas3/rocblas_symm_hemm.hpp"
#include "testing_common.hpp"

template <typename T, bool HERM>
void testing_symm_hemm_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_fn = HERM ? (arg.api & c_API_FORTRAN ? rocblas_hemm_batched<T, true>
                                                      : rocblas_hemm_batched<T, false>)
                           : (arg.api & c_API_FORTRAN ? rocblas_symm_batched<T, true>
                                                      : rocblas_symm_batched<T, false>);

    auto rocblas_fn_64 = HERM ? (arg.api & c_API_FORTRAN ? rocblas_hemm_batched_64<T, true>
                                                         : rocblas_hemm_batched_64<T, false>)
                              : (arg.api & c_API_FORTRAN ? rocblas_symm_batched_64<T, true>
                                                         : rocblas_symm_batched_64<T, false>);

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_side side        = rocblas_side_left;
        const rocblas_fill uplo        = rocblas_fill_upper;
        const int64_t      M           = 100;
        const int64_t      N           = 100;
        const int64_t      lda         = 100;
        const int64_t      ldb         = 100;
        const int64_t      ldc         = 100;
        rocblas_int        batch_count = 2;

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, one_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));

        const T alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* beta  = &beta_h;
        const T* one   = &one_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(one_d, one, sizeof(*one), hipMemcpyHostToDevice));
            one = one_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        size_t rows = (side == rocblas_side_left ? std::max(M, static_cast<int64_t>(1))
                                                 : std::max(N, static_cast<int64_t>(1)));
        size_t cols = rows;

        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_matrix<T>, dA, (rows, cols, lda, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<T>, dB, (M, N, ldb, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<T>, dC, (M, N, ldc, batch_count));

        DAPI_EXPECT(
            rocblas_status_invalid_handle,
            rocblas_fn,
            (nullptr, side, uplo, M, N, alpha, dA, lda, dB, ldb, beta, dC, ldc, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_fn,
                    (handle,
                     rocblas_side_both,
                     uplo,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_fn,
                    (handle,
                     side,
                     rocblas_fill_full,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     dB,
                     ldb,
                     beta,
                     dC,
                     ldc,
                     batch_count));

        DAPI_EXPECT(
            rocblas_status_invalid_pointer,
            rocblas_fn,
            (handle, side, uplo, M, N, nullptr, dA, lda, dB, ldb, beta, dC, ldc, batch_count));

        DAPI_EXPECT(
            rocblas_status_invalid_pointer,
            rocblas_fn,
            (handle, side, uplo, M, N, alpha, dA, lda, dB, ldb, nullptr, dC, ldc, batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_fn,
                        (handle,
                         side,
                         uplo,
                         M,
                         N,
                         alpha,
                         nullptr,
                         lda,
                         dB,
                         ldb,
                         beta,
                         dC,
                         ldc,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_fn,
                        (handle,
                         side,
                         uplo,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         nullptr,
                         ldb,
                         beta,
                         dC,
                         ldc,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_fn,
                        (handle,
                         side,
                         uplo,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         dB,
                         ldb,
                         beta,
                         nullptr,
                         ldc,
                         batch_count));
        }

        // quick return with invalid pointers
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_fn,
                    (handle,
                     side,
                     uplo,
                     0,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     nullptr,
                     ldc,
                     batch_count));

        // alpha==0 and beta==1 all pointers may be null
        DAPI_EXPECT(rocblas_status_success,
                    rocblas_fn,
                    (handle,
                     side,
                     uplo,
                     M,
                     N,
                     zero,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     one,
                     nullptr,
                     ldc,
                     batch_count));
    }
}

template <typename T, bool HERM>
void testing_symm_hemm_batched(const Arguments& arg)
{
    auto rocblas_fn    = HERM ? (arg.api & c_API_FORTRAN ? rocblas_hemm_batched<T, true>
                                                         : rocblas_hemm_batched<T, false>)
                              : (arg.api & c_API_FORTRAN ? rocblas_symm_batched<T, true>
                                                         : rocblas_symm_batched<T, false>);
    auto rocblas_fn_64 = HERM ? (arg.api & c_API_FORTRAN ? rocblas_hemm_batched_64<T, true>
                                                         : rocblas_hemm_batched_64<T, false>)
                              : (arg.api & c_API_FORTRAN ? rocblas_symm_batched_64<T, true>
                                                         : rocblas_symm_batched_64<T, false>);

    auto gflop_count_fn = HERM ? hemm_gflop_count<T> : symm_gflop_count<T>;

    rocblas_local_handle handle{arg};
    rocblas_side         side        = char2rocblas_side(arg.side);
    rocblas_fill         uplo        = char2rocblas_fill(arg.uplo);
    int64_t              M           = arg.M;
    int64_t              N           = arg.N;
    int64_t              lda         = arg.lda;
    int64_t              ldb         = arg.ldb;
    int64_t              ldc         = arg.ldc;
    T                    alpha       = arg.get_alpha<T>();
    T                    beta        = arg.get_beta<T>();
    int64_t              batch_count = arg.batch_count;

    double gpu_time_used, cpu_time_used;
    double err_host   = 0.0;
    double err_device = 0.0;

    // Note: N==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = batch_count < 0 || M < 0 || N < 0 || ldc < M || ldb < M
                        || (side == rocblas_side_left && (lda < M))
                        || (side != rocblas_side_left && (lda < N));
    if(M == 0 || N == 0 || batch_count == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check

        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_fn,
                    (handle,
                     side,
                     uplo,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     ldb,
                     nullptr,
                     nullptr,
                     ldc,
                     batch_count));

        return;
    }

    size_t rows = (side == rocblas_side_left ? M : N);
    size_t cols = rows;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<T>, hA, (rows, cols, lda, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hB, (M, N, ldb, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hC, (M, N, ldc, batch_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hC_gold, (M, N, ldc, batch_count));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<T>, h_beta, (1));

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<T>, dA, (rows, cols, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_matrix<T>, dB, (M, N, ldb, batch_count));
    DEVICE_MEMCHECK(device_batch_matrix<T>, dC, (M, N, ldc, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    if(HERM)
    {
        rocblas_init_matrix(
            hA, arg, rocblas_client_never_set_nan, rocblas_client_hermitian_matrix, true);
    }
    else
    {
        rocblas_init_matrix(
            hA, arg, rocblas_client_never_set_nan, rocblas_client_symmetric_matrix, true);
    }
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    hC_gold.copy_from(hC);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_fn,
                       (handle,
                        side,
                        uplo,
                        M,
                        N,
                        &h_alpha[0],
                        dA.ptr_on_device(),
                        lda,
                        dB.ptr_on_device(),
                        ldb,
                        &h_beta[0],
                        dC.ptr_on_device(),
                        ldc,
                        batch_count));
            handle.post_test(arg);

            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

            DAPI_CHECK(rocblas_fn,
                       (handle,
                        side,
                        uplo,
                        M,
                        N,
                        d_alpha,
                        dA.ptr_on_device(),
                        lda,
                        dB.ptr_on_device(),
                        ldb,
                        d_beta,
                        dC.ptr_on_device(),
                        ldc,
                        batch_count));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_batch_matrix<T>, hC_copy, (M, N, ldc, batch_count));
                CHECK_HIP_ERROR(hC.transfer_from(dC));
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
                        device_batch_matrix<T>, dA_copy, (rows, cols, lda, batch_count));
                    DEVICE_MEMCHECK(device_batch_matrix<T>, dB_copy, (M, N, ldb, batch_count));
                    DEVICE_MEMCHECK(device_batch_matrix<T>, dC_copy, (M, N, ldc, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(h_beta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));
                        DAPI_CHECK(rocblas_fn,
                                   (handle_copy,
                                    side,
                                    uplo,
                                    M,
                                    N,
                                    d_alpha_copy,
                                    dA_copy.ptr_on_device(),
                                    lda,
                                    dB_copy.ptr_on_device(),
                                    ldb,
                                    d_beta_copy,
                                    dC_copy.ptr_on_device(),
                                    ldc,
                                    batch_count));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(M, N, ldc, hC, hC_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(size_t b = 0; b < batch_count; b++)
        {
            if(HERM)
            {
                ref_hemm<T>(
                    side, uplo, M, N, h_alpha, hA[b], lda, hB[b], ldb, h_beta, hC_gold[b], ldc);
            }
            else
            {
                ref_symm<T>(side,
                            uplo,
                            M,
                            N,
                            h_alpha[0],
                            hA[b],
                            lda,
                            hB[b],
                            ldb,
                            h_beta[0],
                            hC_gold[b],
                            ldc);
            }
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<
                       T,
                       rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC, batch_count);
                }
            }

            if(arg.norm_check)
            {
                err_host
                    = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC, batch_count));
            }
        }
        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));

            if(arg.unit_check)
            {
                if(std::is_same_v<
                       T,
                       rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = N * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC, batch_count);
                }
            }

            if(arg.norm_check)
            {
                err_device
                    = std::abs(norm_check_general<T>('F', M, N, ldc, hC_gold, hC, batch_count));
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used, cpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        for(int i = 0; i < total_calls; i++)
        {
            if(i == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds
            DAPI_DISPATCH(rocblas_fn,
                          (handle,
                           side,
                           uplo,
                           M,
                           N,
                           h_alpha,
                           dA.ptr_on_device(),
                           lda,
                           dB.ptr_on_device(),
                           ldb,
                           h_beta,
                           dC.ptr_on_device(),
                           ldc,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side,
                      e_uplo,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_ldb,
                      e_beta,
                      e_ldc,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gflop_count_fn(side, M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         err_host,
                         err_device);
    }
}
