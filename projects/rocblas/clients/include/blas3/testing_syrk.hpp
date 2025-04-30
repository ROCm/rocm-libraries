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

#include "testing_common.hpp"

template <typename T>
void testing_syrk_bad_arg(const Arguments& arg)
{
    auto rocblas_syrk_fn = arg.api & c_API_FORTRAN ? rocblas_syrk<T, true> : rocblas_syrk<T, false>;

    auto rocblas_syrk_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_syrk_64<T, true> : rocblas_syrk_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const int64_t           N      = 100;
        const int64_t           K      = 99;
        const int64_t           lda    = 100;
        const int64_t           ldc    = 100;

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

        size_t rows = (transA == rocblas_operation_none ? N : std::max(K, int64_t(1)));
        size_t cols = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (rows, cols, lda));
        DEVICE_MEMCHECK(device_matrix<T>, dC, (N, N, ldc));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_syrk_fn,
                    (nullptr, uplo, transA, N, K, alpha, dA, lda, beta, dC, ldc));

        // invalid values
        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_syrk_fn,
                    (handle, rocblas_fill_full, transA, N, K, alpha, dA, lda, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_syrk_fn,
                    (handle,
                     (rocblas_fill)rocblas_operation_none,
                     transA,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     beta,
                     dC,
                     ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_syrk_fn,
                    (handle,
                     uplo,
                     (rocblas_operation)rocblas_fill_full,
                     N,
                     K,
                     alpha,
                     dA,
                     lda,
                     beta,
                     dC,
                     ldc));

        // conjugate transpose supported in ssyrk and dsyrk
        if(rocblas_is_complex<T>)
        {
            DAPI_EXPECT(rocblas_status_invalid_value,
                        rocblas_syrk_fn,
                        (handle,
                         uplo,
                         rocblas_operation_conjugate_transpose,
                         N,
                         K,
                         alpha,
                         dA,
                         lda,
                         beta,
                         dC,
                         ldc));
        }

        // size
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_syrk_fn,
                    (handle, uplo, transA, N, K, alpha, dA, lda - 1, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_syrk_fn,
                    (handle, uplo, transA, N, K, alpha, dA, lda, beta, dC, ldc - 1));

        // invalid alpha/beta pointers
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_syrk_fn,
                    (handle, uplo, transA, N, K, nullptr, dA, lda, beta, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_syrk_fn,
                    (handle, uplo, transA, N, K, alpha, dA, lda, nullptr, dC, ldc));

        // invalid pointers
        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syrk_fn,
                        (handle, uplo, transA, N, K, alpha, nullptr, lda, beta, dC, ldc));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_syrk_fn,
                        (handle, uplo, transA, N, K, alpha, dA, lda, beta, nullptr, ldc));
        }

        // N == 0 quick return with invalid pointers
        DAPI_CHECK(rocblas_syrk_fn,
                   (handle, uplo, transA, 0, K, nullptr, nullptr, lda, nullptr, nullptr, ldc));

        // k==0 and beta==1 all other pointers may be null
        DAPI_CHECK(rocblas_syrk_fn,
                   (handle, uplo, transA, N, 0, nullptr, nullptr, lda, one, nullptr, ldc));

        // alpha==0 and beta==1 all other pointers may be null
        DAPI_CHECK(rocblas_syrk_fn,
                   (handle, uplo, transA, N, K, zero, nullptr, lda, one, nullptr, ldc));
    }
}

template <typename T>
void testing_syrk(const Arguments& arg)
{
    auto rocblas_syrk_fn = arg.api & c_API_FORTRAN ? rocblas_syrk<T, true> : rocblas_syrk<T, false>;
    auto rocblas_syrk_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_syrk_64<T, true> : rocblas_syrk_64<T, false>;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    int64_t              N      = arg.N;
    int64_t              K      = arg.K;
    int64_t              lda    = arg.lda;
    int64_t              ldc    = arg.ldc;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    double cpu_time_used;
    double error_host   = 0.0;
    double error_device = 0.0;

    // Note: K==0 is not an early exit, since C still needs to be multiplied by beta
    bool invalid_size = N < 0 || K < 0 || ldc < N || (transA == rocblas_operation_none && lda < N)
                        || (transA != rocblas_operation_none && lda < K);
    if(N == 0 || invalid_size)
    {
        // ensure invalid sizes checked before pointer check
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_syrk_fn,
                    (handle, uplo, transA, N, K, nullptr, nullptr, lda, nullptr, nullptr, ldc));

        return;
    }

    size_t rows = (transA == rocblas_operation_none ? N : std::max(K, int64_t(1)));
    size_t cols = (transA == rocblas_operation_none ? std::max(K, int64_t(1)) : N);

    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (rows, cols, lda));
    HOST_MEMCHECK(host_matrix<T>, hC, (N, N, ldc));
    HOST_MEMCHECK(host_matrix<T>, hC_gold, (N, N, ldc));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<T>, h_beta, (1));

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true, true);
    rocblas_init_matrix(
        hC, arg, rocblas_client_beta_sets_nan, rocblas_client_symmetric_matrix, false, true);
    hC_gold = hC;

    if(arg.unit_check || arg.norm_check)
    {
        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (rows, cols, lda));
        DEVICE_MEMCHECK(device_matrix<T>, dC, (N, N, ldc));

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        if(arg.pointer_mode_host)
        {
            // host alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            CHECK_HIP_ERROR(dC.transfer_from(hC));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_syrk_fn,
                       (handle, uplo, transA, N, K, &h_alpha[0], dA, lda, &h_beta[0], dC, ldc));
            handle.post_test(arg);
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            // device alpha/beta
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));

            DAPI_CHECK(rocblas_syrk_fn,
                       (handle, uplo, transA, N, K, d_alpha, dA, lda, d_beta, dC, ldc));

            if(arg.repeatability_check)
            {
                CHECK_HIP_ERROR(hC.transfer_from(dC));

                HOST_MEMCHECK(host_matrix<T>, hC_copy, (N, N, ldc));

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
                    DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (rows, cols, lda));
                    DEVICE_MEMCHECK(device_matrix<T>, dC_copy, (N, N, ldc));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(h_beta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));
                        DAPI_CHECK(rocblas_syrk_fn,
                                   (handle_copy,
                                    uplo,
                                    transA,
                                    N,
                                    K,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    d_beta_copy,
                                    dC_copy,
                                    ldc));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(N, N, ldc, hC, hC_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        ref_syrk<T>(uplo, transA, N, K, h_alpha[0], hA, lda, h_beta[0], hC_gold, ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<
                       T,
                       rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>)
                {
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, hC_gold, hC);
                }
            }

            if(arg.norm_check)
            {
                error_host = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC));
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
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(N, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(N, N, ldc, hC_gold, hC);
                }
            }

            if(arg.norm_check)
            {
                error_device = std::abs(norm_check_general<T>('F', N, N, ldc, hC_gold, hC));
            }
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        int    number_cold_calls = arg.cold_iters;
        int    number_hot_calls  = arg.iters;

        // Information on flush_memory_size and flush_batch_count
        // - To time syrk it is called number_hot_calls times.
        // - if the size of dA and dC are small enough they will be cached
        //   and reused number_hot_calls-1 times.
        // - This "hot-cache" timing will give higher performance than if the
        //   cache is flushed
        // - arg.flush_batch_count or arg.flush_memory_size can be used to avoid
        //   caching of dA and dC.
        // - if arg.flush_memory_size is specified, then flush_batch_count is calculated.
        // - only one of arg.flush_memory_size or arg.flush_batch_count can be
        //   used, not both.
        // - Note that this is only used in timing code, not in testing code.
        // - The method is as outlined in
        //   "Achieving accurate and context-sensitive timing for code optimization" by Whaley and Castaldo.
        // - In the number_hot_calls timing loop it cycles through the arg.flush_batch_count copies
        //   of dA and dC, and if flush_memory_size is large enough they will be evicted
        //   from cache before they are reused.
        // - The individual matrices are aligned on byte boundaries used by hipMalloc
        size_t stride_a = lda * cols;
        size_t stride_c = ldc * N;

        size_t aligned_stride_a = align_stride<T>(stride_a);
        size_t aligned_stride_c = align_stride<T>(stride_c);

        size_t flush_batch_count = 1;
        size_t a_size            = rows * cols * sizeof(T);
        size_t c_size            = N * N * sizeof(T);
        size_t a_c_cached_size   = a_size + c_size;

        flush_batch_count = calculate_flush_batch_count(
            arg.flush_batch_count, arg.flush_memory_size, a_c_cached_size);

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                        dA,
                        (rows, cols, lda, aligned_stride_a, flush_batch_count));
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dC, (N, N, ldc, aligned_stride_c, flush_batch_count));

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.broadcast_one_matrix_from(hA));
        CHECK_HIP_ERROR(dC.broadcast_one_matrix_from(hC_gold));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dC.broadcast_one_matrix_from(hC));

        for(int i = 0; i < number_cold_calls; i++)
        {
            DAPI_DISPATCH(rocblas_syrk_fn,
                          (handle, uplo, transA, N, K, h_alpha, dA[0], lda, h_beta, dC[0], ldc));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            int flush_index = (i + 1) % flush_batch_count;
            DAPI_DISPATCH(rocblas_syrk_fn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           h_alpha,
                           dA[flush_index],
                           lda,
                           h_beta,
                           dC[flush_index],
                           ldc));
            ;
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc>{}.log_args<T>(
            rocblas_cout,
            arg,
            gpu_time_used,
            syrk_gflop_count<T>(N, K),
            ArgumentLogging::NA_value,
            cpu_time_used,
            error_host,
            error_device);
    }
}
