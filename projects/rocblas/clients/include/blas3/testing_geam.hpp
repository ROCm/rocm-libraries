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
void testing_geam_bad_arg(const Arguments& arg)
{
    auto rocblas_geam_fn = arg.api & c_API_FORTRAN ? rocblas_geam<T, true> : rocblas_geam<T, false>;
    auto rocblas_geam_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_geam_64<T, true> : rocblas_geam_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const int64_t M = 100;
        const int64_t N = 99;

        const int64_t lda = 100;
        const int64_t ldb = 100;
        const int64_t ldc = 100;

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

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        int64_t A_row = transA == rocblas_operation_none ? M : N;
        int64_t A_col = transA == rocblas_operation_none ? N : M;
        int64_t B_row = transB == rocblas_operation_none ? M : N;
        int64_t B_col = transB == rocblas_operation_none ? N : M;

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<T>, dB, (B_row, B_col, ldb));
        DEVICE_MEMCHECK(device_matrix<T>, dC, (M, N, ldc));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_geam_fn,
                    (nullptr, transA, transB, M, N, alpha, dA, lda, beta, dB, ldb, dC, ldc));

        // invalid values
        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_geam_fn,
                    (handle,
                     (rocblas_operation)rocblas_fill_full,
                     transB,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     beta,
                     dB,
                     ldb,
                     dC,
                     ldc));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_geam_fn,
                    (handle,
                     transA,
                     (rocblas_operation)rocblas_fill_full,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     beta,
                     dB,
                     ldb,
                     dC,
                     ldc));

        // invalid sizes not done in yaml test

        // A == C leading dims must match
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_geam_fn,
                    (handle, transA, transB, M, N, alpha, dC, ldc + 1, beta, dB, ldb, dC, ldc));

        // B == C leading dims must match
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_geam_fn,
                    (handle, transA, transB, M, N, alpha, dA, lda, beta, dC, ldc + 1, dC, ldc));

        // alpha/beta
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_geam_fn,
                    (handle, transA, transB, M, N, nullptr, dA, lda, beta, dB, ldb, dC, ldc));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_geam_fn,
                    (handle, transA, transB, M, N, alpha, dA, lda, nullptr, dB, ldb, dC, ldc));

        // invalid pointers
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_geam_fn,
                    (handle, transA, transB, M, N, alpha, dA, lda, beta, dB, ldb, nullptr, ldc));

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            DAPI_EXPECT(
                rocblas_status_invalid_pointer,
                rocblas_geam_fn,
                (handle, transA, transB, M, N, alpha, nullptr, lda, beta, dB, ldb, dC, ldc));

            DAPI_EXPECT(
                rocblas_status_invalid_pointer,
                rocblas_geam_fn,
                (handle, transA, transB, M, N, alpha, dA, lda, beta, nullptr, ldb, dC, ldc));
        }

        // M==0 then all may be nullptr
        DAPI_CHECK(rocblas_geam_fn,
                   (handle,
                    transA,
                    transB,
                    0,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    nullptr,
                    ldb,
                    nullptr,
                    ldc));

        // N==0 then all may be nullptr
        DAPI_CHECK(rocblas_geam_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    nullptr,
                    nullptr,
                    ldb,
                    nullptr,
                    ldc));

        // alpha==0 then A may be nullptr
        DAPI_CHECK(rocblas_geam_fn,
                   (handle, transA, transB, M, N, zero, nullptr, lda, beta, dB, ldb, dC, ldc));

        // beta==0 then B may be nullptr
        DAPI_CHECK(rocblas_geam_fn,
                   (handle, transA, transB, M, N, alpha, dA, lda, zero, nullptr, ldb, dC, ldc));
    }
}

template <typename T>
void testing_geam(const Arguments& arg)
{
    auto rocblas_geam_fn = arg.api & c_API_FORTRAN ? rocblas_geam<T, true> : rocblas_geam<T, false>;
    auto rocblas_geam_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_geam_64<T, true> : rocblas_geam_64<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    int64_t M = arg.M;
    int64_t N = arg.N;

    int64_t lda = arg.lda;
    int64_t ldb = arg.ldb;
    int64_t ldc = arg.ldc;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    T* dC_in_place;

    int64_t A_row = transA == rocblas_operation_none ? M : N;
    int64_t A_col = transA == rocblas_operation_none ? N : M;
    int64_t B_row = transB == rocblas_operation_none ? M : N;
    int64_t B_col = transB == rocblas_operation_none ? N : M;

    double cpu_time_used = 0.0;

    double rocblas_error_1 = std::numeric_limits<double>::max();
    double rocblas_error_2 = std::numeric_limits<double>::max();
    double rocblas_error   = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    size_t size_C = ldc * N;

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < A_row || ldb < B_row || ldc < M;
    if(invalid_size || !M || !N)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_geam_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     nullptr,
                     nullptr,
                     ldb,
                     (T*)0x1, // defeat C==A or B leading dim invalid checks, C nullptr in bad_arg
                     ldc));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (A_row, A_col, lda));
    HOST_MEMCHECK(host_matrix<T>, hA_copy, (A_row, A_col, lda));
    HOST_MEMCHECK(host_matrix<T>, hB, (B_row, B_col, ldb));
    HOST_MEMCHECK(host_matrix<T>, hB_copy, (B_row, B_col, ldb));
    HOST_MEMCHECK(host_matrix<T>, hC, (M, N, ldc));
    HOST_MEMCHECK(host_matrix<T>, hC_gold, (M, N, ldc));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<T>, h_beta, (1));

    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (A_row, A_col, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dB, (B_row, B_col, ldb));
    DEVICE_MEMCHECK(device_matrix<T>, dC, (M, N, ldc));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(hB, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    hA_copy = hA;
    hB_copy = hB;

    // copy data from CPU to device

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_geam_fn,
                       (handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC, ldc));
            handle.post_test(arg);
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
            DAPI_CHECK(rocblas_geam_fn,
                       (handle, transA, transB, M, N, d_alpha, dA, lda, d_beta, dB, ldb, dC, ldc));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_matrix<T>, hC_copy, (M, N, ldc));
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
                    DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (A_row, A_col, lda));
                    DEVICE_MEMCHECK(device_matrix<T>, dB_copy, (B_row, B_col, ldb));
                    DEVICE_MEMCHECK(device_matrix<T>, dC_copy, (M, N, ldc));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));
                    DEVICE_MEMCHECK(device_vector<T>, d_beta_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(dC_copy.transfer_from(hC_gold));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));
                    CHECK_HIP_ERROR(d_beta_copy.transfer_from(h_beta));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        DAPI_CHECK(rocblas_geam_fn,
                                   (handle_copy,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    d_beta_copy,
                                    dB_copy,
                                    ldb,
                                    dC_copy,
                                    ldc));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(M, N, ldc, hC, hC_copy);
                    }
                }
                return;
            }
        }

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        ref_geam(transA,
                 transB,
                 M,
                 N,
                 (T*)h_alpha,
                 (T*)hA,
                 lda,
                 (T*)h_beta,
                 (T*)hB,
                 ldb,
                 (T*)hC_gold,
                 ldc);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC);
            }

            if(arg.norm_check)
            {
                rocblas_error_1 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC);
            }
        }

        if(arg.pointer_mode_device)
        {
            // fetch GPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));

            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldc, hC_gold, hC);
            }

            if(arg.norm_check)
            {
                rocblas_error_2 = norm_check_general<T>('F', M, N, ldc, hC_gold, hC);
            }
        }

        // extra host mode checks

        // inplace check for dC == dA
        if(arg.pointer_mode_host)
        {
            bool invalid_size_in_place = lda != ldc || transA != rocblas_operation_none;

            dC_in_place = dA;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            DAPI_EXPECT(
                invalid_size_in_place ? rocblas_status_invalid_size : rocblas_status_success,
                rocblas_geam_fn,
                (handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC_in_place, ldc));

            if(!invalid_size_in_place)
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(hC, dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(dA.transfer_from(hA));

                // reference calculation
                ref_geam(transA,
                         transB,
                         M,
                         N,
                         (T*)h_alpha,
                         (T*)hA_copy,
                         lda,
                         (T*)h_beta,
                         (T*)hB,
                         ldb,
                         (T*)hC_gold,
                         ldc);

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC);
                }
            }
        }

        // inplace check for dC == dB
        if(arg.pointer_mode_host)
        {
            bool invalid_size_in_place = ldb != ldc || transB != rocblas_operation_none;

            dC_in_place = dB;

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            DAPI_EXPECT(
                invalid_size_in_place ? rocblas_status_invalid_size : rocblas_status_success,
                rocblas_geam_fn,
                (handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC_in_place, ldc));

            if(!invalid_size_in_place)
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(hC, dC_in_place, sizeof(T) * size_C, hipMemcpyDeviceToHost));

                // reference calculation
                ref_geam(transA,
                         transB,
                         M,
                         N,
                         (T*)h_alpha,
                         (T*)hA_copy,
                         lda,
                         (T*)h_beta,
                         (T*)hB_copy,
                         ldb,
                         (T*)hC_gold,
                         ldc);

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, hC_gold, hC);
                }

                if(arg.norm_check)
                {
                    rocblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC);
                }
            }
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

            DAPI_DISPATCH(rocblas_geam_fn,
                          (handle, transA, transB, M, N, &alpha, dA, lda, &beta, dB, ldb, dC, ldc));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_alpha, e_lda, e_beta, e_ldb, e_ldc>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         geam_gflop_count<T>(M, N),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         rocblas_error_1,
                         rocblas_error_2);
    }
}
