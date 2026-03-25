/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_matrix.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */
template <typename Ti, typename To, typename Tex>
void testing_herk_ex_bad_arg(const Arguments& arg)
{
    using Tab_ex = real_t<Tex>;

    auto rocblas_herk_ex_fn = arg.api & c_API_FORTRAN ? rocblas_herk_ex_fortran : rocblas_herk_ex;
    auto rocblas_herk_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_herk_ex_fortran : rocblas_herk_ex;
    // TODO
    //auto rocblas_herk_ex_fn_64
    //    = arg.api & c_API_FORTRAN ? rocblas_herk_ex_64_fortran : rocblas_herk_ex_64;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const int64_t           N      = 100;
        const int64_t           K      = 99;
        const int64_t           lda    = 100;
        const int64_t           ldc    = 100;

        rocblas_datatype a_type       = arg.a_type;
        rocblas_datatype c_type       = arg.c_type;
        rocblas_datatype compute_type = arg.compute_type;

        DEVICE_MEMCHECK(device_vector<Tab_ex>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<Tab_ex>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<Tab_ex>, one_d, (1));
        DEVICE_MEMCHECK(device_vector<Tab_ex>, zero_d, (1));

        const Tab_ex alpha_h(1), beta_h(2), one_h(1), zero_h(0);

        const Tab_ex* alpha = &alpha_h;
        const Tab_ex* beta  = &beta_h;
        const Tab_ex* one   = &one_h;
        const Tab_ex* zero  = &zero_h;

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

        int64_t Kmax  = std::max(K, int64_t(1));
        int64_t A_row = transA == rocblas_operation_none ? N : Kmax;
        int64_t A_col = transA == rocblas_operation_none ? Kmax : N;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<Ti>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<To>, dC, (N, N, ldc));

        // host
        HOST_MEMCHECK(host_matrix<Ti>, hA, (A_row, A_col, lda));
        HOST_MEMCHECK(host_matrix<To>, hC, (N, N, ldc));

        rocblas_seedrand();

        // Initialize data on host memory
        rocblas_init_matrix(
            hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true, true);
        rocblas_init_matrix(
            hC, arg, rocblas_client_beta_sets_nan, rocblas_client_symmetric_matrix, false);

        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dC.transfer_from(hC));

        // clang-format off

// check for invalid enum
DAPI_EXPECT(rocblas_status_invalid_value, rocblas_herk_ex_fn, (handle, (rocblas_fill) rocblas_side_both, transA, N, K, nullptr,
nullptr, a_type, lda, nullptr, nullptr, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_value, rocblas_herk_ex_fn, (handle, uplo, (rocblas_operation) rocblas_side_both, N, K, nullptr,
nullptr, a_type, lda, nullptr, nullptr, c_type, ldc, compute_type));


// check for invalid size
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_herk_ex_fn, (handle, uplo, transA, -1, K, nullptr,
nullptr, a_type, lda, nullptr, nullptr, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_herk_ex_fn, (handle, uplo, transA, N, -1, nullptr,
nullptr, a_type, lda, nullptr, nullptr, c_type, ldc, compute_type));

// check for invalid leading dimension
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, nullptr,
nullptr, a_type, N-1, nullptr, nullptr, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, nullptr,
nullptr, a_type, N, nullptr, nullptr, c_type, N-1, compute_type));

// checks that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer

DAPI_EXPECT(rocblas_status_invalid_handle, rocblas_herk_ex_fn, (nullptr, uplo, transA, N, K, alpha,
dA, a_type, lda, beta, dC, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, nullptr,
dA, a_type, lda, beta, dC, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, alpha,
nullptr, a_type, lda, beta, dC, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, alpha,
dA, a_type, lda, nullptr, dC, c_type, ldc, compute_type));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, alpha,
dA, a_type, lda, beta, nullptr, c_type, ldc, compute_type));

// If N==0, then all pointers can be nullptr without issue
DAPI_CHECK(rocblas_herk_ex_fn, (handle, uplo, transA, 0, K, nullptr,
nullptr, a_type, lda, nullptr, nullptr, c_type, ldc, compute_type));

// If alpha==0 then A can be nullptr without issue.
DAPI_CHECK(rocblas_herk_ex_fn, (handle, uplo, transA, N, K, zero,
nullptr, a_type, lda, beta, dC, c_type, ldc, compute_type));

// k==0 and beta==1 all other pointers may be null
DAPI_CHECK(rocblas_herk_ex_fn, (handle, uplo, transA, N, 0, nullptr,
nullptr, a_type, lda, one, nullptr, c_type, ldc, compute_type));

// alpha==0 and beta==1 all other pointers may be null
DAPI_CHECK(rocblas_herk_ex_fn, (handle, uplo, transA, N, K, zero,
nullptr, a_type, lda, one, nullptr, c_type, ldc, compute_type));

        // clang-format on
    }
}

template <typename Ti, typename To, typename Tex>
void testing_herk_ex(const Arguments& arg)
{
    auto rocblas_herk_ex_fn = arg.api & c_API_FORTRAN ? rocblas_herk_ex_fortran : rocblas_herk_ex;
    auto rocblas_herk_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_herk_ex_fortran : rocblas_herk_ex;
    // TODO
    //auto rocblas_herk_ex_fn_64
    //    = arg.api & c_API_FORTRAN ? rocblas_herk_ex_64_fortran : rocblas_herk_ex_64;

    using Tab_ex = real_t<Tex>;

    bool alpha_isnan = arg.alpha_isnan<Tab_ex>();
    bool beta_isnan  = arg.beta_isnan<Tab_ex>();
    if(!std::is_same_v<
           To,
           float> && !std::is_same_v<To, double> && !std::is_same_v<To, rocblas_half> && !rocblas_is_complex<To> && (alpha_isnan || beta_isnan))
        return; // Exclude integers or other types which don't support NaN

    Tab_ex h_alpha_Tc = arg.get_alpha<Tab_ex>();
    Tab_ex h_beta_Tc  = arg.get_beta<Tab_ex>();

    double cpu_time_used;
    double error_host   = 0.0;
    double error_device = 0.0;

    rocblas_local_handle handle{arg};
    rocblas_fill         uplo   = char2rocblas_fill(arg.uplo);
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    int64_t              N = arg.N, K = arg.K;
    int64_t              lda = arg.lda, ldc = arg.ldc;
    int64_t              Kmax  = std::max(K, int64_t(1));
    int64_t              A_row = transA == rocblas_operation_none ? N : Kmax;
    int64_t              A_col = transA == rocblas_operation_none ? Kmax : N;

    rocblas_datatype a_type       = arg.a_type;
    rocblas_datatype c_type       = arg.c_type;
    rocblas_datatype compute_type = arg.compute_type;

    // check for invalid sizes
    bool invalid_size = N < 0 || K < 0 || lda < A_row || ldc < N;
    if(invalid_size)
    {
        // clang-format off
        DAPI_EXPECT(rocblas_status_invalid_size, rocblas_herk_ex_fn, (handle, uplo, transA, N, K, nullptr,
                                                 nullptr, a_type, lda,
                                                 nullptr,
                                                 nullptr, c_type, ldc,
                                                 compute_type));
        // clang-format on
        return;
    }

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tab_ex>, d_alpha_Tc, (1));
    DEVICE_MEMCHECK(device_vector<Tab_ex>, d_beta_Tc, (1));

    // Allocate host memory
    HOST_MEMCHECK(host_matrix<Ti>, hA, (A_row, A_col, lda));
    HOST_MEMCHECK(host_matrix<To>, hC, (N, N, ldc));

    if(arg.unit_check || arg.norm_check)
    {
        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<Ti>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<To>, dC, (N, N, ldc));
        DEVICE_MEMCHECK(device_vector<Tab_ex>, d_alpha_Tc, (1));
        DEVICE_MEMCHECK(device_vector<Tab_ex>, d_beta_Tc, (1));

        // Initialize data on host memory
        rocblas_init_matrix(
            hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true, true);
        rocblas_init_matrix(
            hC, arg, rocblas_client_beta_sets_nan, rocblas_client_hermitian_matrix, false);

        HOST_MEMCHECK(host_matrix<To>, hC_copy, (N, N, ldc));
        HOST_MEMCHECK(host_matrix<To>, hC_orig, (N, N, ldc));
        HOST_MEMCHECK(host_matrix<To>, hC_gold, (N, N, ldc));
        hC_orig = hC;
        hC_gold = hC;

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dC.transfer_from(hC));

        //using To = std::conditional_t<std::is_same_v<To, rocblas_bfloat16>, float, To>;
        //rocblas_init_nan<To_hpa>(hC_gold, N, N, ldc);

        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_herk_ex_fn,
                       (handle,
                        uplo,
                        transA,
                        N,
                        K,
                        &h_alpha_Tc,
                        dA,
                        a_type,
                        lda,
                        &h_beta_Tc,
                        dC,
                        c_type,
                        ldc,
                        compute_type));
            handle.post_test(arg);
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_orig));
            CHECK_HIP_ERROR(
                hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tab_ex), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tab_ex), hipMemcpyHostToDevice));
            DAPI_CHECK(rocblas_herk_ex_fn,
                       (handle,
                        uplo,
                        transA,
                        N,
                        K,
                        d_alpha_Tc,
                        dA,
                        a_type,
                        lda,
                        d_beta_Tc,
                        dC,
                        c_type,
                        ldc,
                        compute_type));

            if(arg.repeatability_check)
            {
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
                    DEVICE_MEMCHECK(device_matrix<Ti>, dA_copy, (A_row, A_col, lda));
                    DEVICE_MEMCHECK(device_matrix<To>, dC_copy, (N, N, ldc));
                    DEVICE_MEMCHECK(device_vector<Tab_ex>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<Tab_ex>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_alpha_copy, &h_alpha_Tc, sizeof(Tab_ex), hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(d_beta_copy, &h_beta_Tc, sizeof(Tab_ex), hipMemcpyHostToDevice));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC_orig));
                        DAPI_CHECK(rocblas_herk_ex_fn,
                                   (handle_copy,
                                    uplo,
                                    transA,
                                    N,
                                    K,
                                    d_alpha_copy,
                                    dA_copy,
                                    a_type,
                                    lda,
                                    d_beta_copy,
                                    dC_copy,
                                    c_type,
                                    ldc,
                                    compute_type));

                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<To>(N, N, ldc, hC, hC_copy);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        ref_herk_ex<Ti, To, Tab_ex>(
            uplo, transA, N, K, h_alpha_Tc, hA, lda, h_beta_Tc, hC_gold, ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        auto compare_hC_to_gold = [&] {
            if(arg.unit_check)
            {
                bool near_check = arg.initialization == rocblas_initialization::hpl
                                  || (sizeof(To) < 4
                                      && (((K > 1000) || (rocblas_is_complex<To>))
                                          || (rocblas_handle(handle)->getArchMajor() == 11)));
                if(near_check)
                {
                    // reference is computed on floats
                    double tol = rocblas_handle(handle)->getArchMajor() == 11
                                     ? sum_error_tolerance_for_gfx11<Tex, Ti, To>
                                     : 4 * sum_error_tolerance<Ti>;
                    tol = tol * K + 2 * sum_error_tolerance<To>; // add To conversion rounding error
                    near_check_general<To>(N, N, ldc, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<To>(N, N, ldc, hC_gold, hC);
                }
            }

            double error = 0;
            if(arg.norm_check)
            {
                error = std::abs(norm_check_general<To>('F', N, N, ldc, (To*)hC_gold, (To*)hC));
            }
            return error;
        };

        if(arg.pointer_mode_host)
        {
            error_host = compare_hC_to_gold();
        }

        if(arg.pointer_mode_device)
        {
            // copy output from device to CPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));
            error_device = compare_hC_to_gold();
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
        size_t stride_a = lda * A_col;
        size_t stride_c = ldc * N;

        size_t aligned_stride_a = align_stride<Ti>(stride_a);
        size_t aligned_stride_c = align_stride<To>(stride_c);

        size_t flush_batch_count = 1;
        size_t a_size            = A_row * A_col * sizeof(Ti);
        size_t c_size            = N * N * sizeof(To);
        size_t a_c_cached_size   = a_size + c_size;

        flush_batch_count = calculate_flush_batch_count(
            arg.flush_batch_count, arg.flush_memory_size, a_c_cached_size);

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<Ti>,
                        dA,
                        (A_row, A_col, lda, aligned_stride_a, flush_batch_count));
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<To>, dC, (N, N, ldc, aligned_stride_c, flush_batch_count));

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.broadcast_one_matrix_from(hA));
        CHECK_HIP_ERROR(dC.broadcast_one_matrix_from(hC));
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            DAPI_DISPATCH(rocblas_herk_ex_fn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           (void*)&h_alpha_Tc,
                           (void*)dA[0],
                           a_type,
                           lda,
                           (void*)&h_beta_Tc,
                           (void*)dC[0],
                           c_type,
                           ldc,
                           compute_type));
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            int flush_index = (i + 1) % flush_batch_count;
            DAPI_DISPATCH(rocblas_herk_ex_fn,
                          (handle,
                           uplo,
                           transA,
                           N,
                           K,
                           (void*)&h_alpha_Tc,
                           (void*)dA[flush_index],
                           a_type,
                           lda,
                           (void*)&h_beta_Tc,
                           (void*)dC[flush_index],
                           c_type,
                           ldc,
                           compute_type));
            ;
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo, e_transA, e_N, e_K, e_alpha, e_lda, e_beta, e_ldc>{}.log_args<To>(
            rocblas_cout,
            arg,
            gpu_time_used,
            herk_gflop_count<To>(N, K),
            ArgumentLogging::NA_value,
            cpu_time_used,
            error_host,
            error_device);
    }
}
