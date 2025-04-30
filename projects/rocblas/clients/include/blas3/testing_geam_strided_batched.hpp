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
void testing_geam_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_geam_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_geam_strided_batched<T, true>
                                               : rocblas_geam_strided_batched<T, false>;

    auto rocblas_geam_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_geam_strided_batched_64<T, true>
                                                  : rocblas_geam_strided_batched_64<T, false>;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const int64_t M = 100;
        const int64_t N = 99;

        const int64_t lda = 100;
        const int64_t ldb = 100;
        const int64_t ldc = 100;

        const int64_t batch_count = 2;

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

        const rocblas_stride stride_a = lda * A_col;
        const rocblas_stride stride_b = ldb * B_col;
        const rocblas_stride stride_c = ldc * N;

        // Allocate device memory
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dA, (A_row, A_col, lda, stride_a, batch_count));
        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dB, (B_row, B_col, ldb, stride_b, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (M, N, ldc, stride_c, batch_count));

        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_geam_strided_batched_fn,
                    (nullptr,
                     transA,
                     transB,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     beta,
                     dB,
                     ldb,
                     stride_b,
                     dC,
                     ldc,
                     stride_c,
                     batch_count));

        // invalid values
        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_geam_strided_batched_fn,
                    (handle,
                     (rocblas_operation)rocblas_fill_full,
                     transB,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     beta,
                     dB,
                     ldb,
                     stride_b,
                     dC,
                     ldc,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_geam_strided_batched_fn,
                    (handle,
                     transA,
                     (rocblas_operation)rocblas_fill_full,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     beta,
                     dB,
                     ldb,
                     stride_b,
                     dC,
                     ldc,
                     stride_c,
                     batch_count));

        // size in regular tests

        // alpha/beta
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_geam_strided_batched_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     nullptr,
                     dA,
                     lda,
                     stride_a,
                     beta,
                     dB,
                     ldb,
                     stride_b,
                     dC,
                     ldc,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_geam_strided_batched_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     nullptr,
                     dB,
                     ldb,
                     stride_b,
                     dC,
                     ldc,
                     stride_c,
                     batch_count));

        // invalid pointers
        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_geam_strided_batched_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     beta,
                     dB,
                     ldb,
                     stride_b,
                     nullptr,
                     ldc,
                     stride_c,
                     batch_count));

        if(pointer_mode == rocblas_pointer_mode_host)
        {

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_geam_strided_batched_fn,
                        (handle,
                         transA,
                         transB,
                         M,
                         N,
                         alpha,
                         nullptr,
                         lda,
                         stride_a,
                         beta,
                         dB,
                         ldb,
                         stride_b,
                         dC,
                         ldc,
                         stride_c,
                         batch_count));

            DAPI_EXPECT(rocblas_status_invalid_pointer,
                        rocblas_geam_strided_batched_fn,
                        (handle,
                         transA,
                         transB,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         stride_a,
                         beta,
                         nullptr,
                         ldb,
                         stride_b,
                         dC,
                         ldc,
                         stride_c,
                         batch_count));
        }

        // batch_count==0 then all may be nullptr
        DAPI_CHECK(rocblas_geam_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    nullptr,
                    ldb,
                    stride_b,
                    nullptr,
                    ldc,
                    stride_c,
                    0));

        // M==0 then all may be nullptr
        DAPI_CHECK(rocblas_geam_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    0,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    nullptr,
                    ldb,
                    stride_b,
                    nullptr,
                    ldc,
                    stride_c,
                    0));

        // N==0 then all may be nullptr
        DAPI_CHECK(rocblas_geam_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    nullptr,
                    ldb,
                    stride_b,
                    nullptr,
                    ldc,
                    stride_c,
                    0));

        // alpha==0 then A may be nullptr
        DAPI_CHECK(rocblas_geam_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    zero,
                    nullptr,
                    lda,
                    stride_a,
                    beta,
                    dB,
                    ldb,
                    stride_b,
                    dC,
                    ldc,
                    stride_c,
                    batch_count));

        // beta==0 then B may be nullptr
        DAPI_CHECK(rocblas_geam_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    M,
                    N,
                    alpha,
                    dA,
                    lda,
                    stride_a,
                    zero,
                    nullptr,
                    ldb,
                    stride_b,
                    dC,
                    ldc,
                    stride_c,
                    batch_count));
    }
}

template <typename T>
void testing_geam_strided_batched(const Arguments& arg)
{
    auto rocblas_geam_strided_batched_fn = arg.api & c_API_FORTRAN
                                               ? rocblas_geam_strided_batched<T, true>
                                               : rocblas_geam_strided_batched<T, false>;

    auto rocblas_geam_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_geam_strided_batched_64<T, true>
                                                  : rocblas_geam_strided_batched_64<T, false>;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    int64_t M = arg.M;
    int64_t N = arg.N;

    int64_t        lda         = arg.lda;
    int64_t        ldb         = arg.ldb;
    int64_t        ldc         = arg.ldc;
    rocblas_stride stride_a    = arg.stride_a;
    rocblas_stride stride_b    = arg.stride_b;
    rocblas_stride stride_c    = arg.stride_c;
    int64_t        batch_count = arg.batch_count;

    T alpha = arg.get_alpha<T>();
    T beta  = arg.get_beta<T>();

    double cpu_time_used = 0.0;

    double rocblas_error_1 = std::numeric_limits<double>::max();
    double rocblas_error_2 = std::numeric_limits<double>::max();
    double rocblas_error   = std::numeric_limits<double>::max();

    rocblas_local_handle handle{arg};

    int64_t A_row = transA == rocblas_operation_none ? M : N;
    int64_t A_col = transA == rocblas_operation_none ? N : M;
    int64_t B_row = transB == rocblas_operation_none ? M : N;
    int64_t B_col = transB == rocblas_operation_none ? N : M;

    if(stride_a < int64_t(lda) * A_col)
    {
        rocblas_cout << "WARNING: stride_a < lda * A_col, \n"
                        "setting stride_a = int64_t(lda) * A_col"
                     << std::endl;
        stride_a = int64_t(lda) * A_col;
    }
    if(stride_b < int64_t(ldb) * B_col)
    {
        rocblas_cout << "WARNING: stride_b < ldb * B_col, \n"
                        "setting stride_b = int64_t(ldb) * B_col"
                     << std::endl;
        stride_b = int64_t(ldb) * B_col;
    }
    if(stride_c < int64_t(ldc) * N)
    {
        rocblas_cout << "WARNING: stride_c < ldc * N), setting stride_c = int64_t(ldc) * N"
                     << std::endl;
        stride_c = int64_t(ldc) * N;
    }

    // argument sanity check before allocating invalid memory
    bool invalid_size = M < 0 || N < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_geam_strided_batched_fn,
                    (handle,
                     transA,
                     transB,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     stride_a,
                     nullptr,
                     nullptr,
                     ldb,
                     stride_b,
                     (T*)0x1, // defeat C==A or B leading dim invalid checks, C nullptr in bad_arg
                     ldc,
                     stride_c,
                     batch_count));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (A_row, A_col, lda, stride_a, batch_count));
    HOST_MEMCHECK(
        host_strided_batch_matrix<T>, hA_copy, (A_row, A_col, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hB, (B_row, B_col, ldb, stride_b, batch_count));
    HOST_MEMCHECK(
        host_strided_batch_matrix<T>, hB_copy, (B_row, B_col, ldb, stride_b, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC, (M, N, ldc, stride_c, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC_gold, (M, N, ldc, stride_c, batch_count));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));
    HOST_MEMCHECK(host_vector<T>, h_beta, (1));

    // Initial Data on CPU
    h_alpha[0] = alpha;
    h_beta[0]  = beta;

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (A_row, A_col, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dB, (B_row, B_col, ldb, stride_b, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dC, (M, N, ldc, stride_c, batch_count));
    DEVICE_MEMCHECK(
        device_strided_batch_matrix<T>, dC_in_place, (M, N, ldc, stride_c, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix);

    hA_copy.copy_from(hA);
    hB_copy.copy_from(hB);
    hC_gold.copy_from(hC);

    // copy data from CPU to device

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));

    if(arg.unit_check || arg.norm_check)
    {
        // ROCBLAS
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            DAPI_CHECK(rocblas_geam_strided_batched_fn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        &alpha,
                        dA,
                        lda,
                        stride_a,
                        &beta,
                        dB,
                        ldb,
                        stride_b,
                        dC,
                        ldc,
                        stride_c,
                        batch_count));
            handle.post_test(arg);
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
            DAPI_CHECK(rocblas_geam_strided_batched_fn,
                       (handle,
                        transA,
                        transB,
                        M,
                        N,
                        d_alpha,
                        dA,
                        lda,
                        stride_a,
                        d_beta,
                        dB,
                        ldb,
                        stride_b,
                        dC,
                        ldc,
                        stride_c,
                        batch_count));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(
                    host_strided_batch_matrix<T>, hC_copy, (M, N, ldc, stride_c, batch_count));
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
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dA_copy,
                                    (A_row, A_col, lda, stride_a, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dB_copy,
                                    (B_row, B_col, ldb, stride_b, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dC_copy,
                                    (M, N, ldc, stride_c, batch_count));
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
                        DAPI_CHECK(rocblas_geam_strided_batched_fn,
                                   (handle_copy,
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    stride_a,
                                    d_beta_copy,
                                    dB_copy,
                                    ldb,
                                    stride_b,
                                    dC_copy,
                                    ldc,
                                    stride_c,
                                    batch_count));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(dC_copy));
                        unit_check_general<T>(M, N, ldc, stride_c, hC, hC_copy, batch_count);
                    }
                }
                return;
            }
        }

        // reference calculation for golden result
        cpu_time_used = get_time_us_no_sync();

        for(size_t b = 0; b < batch_count; b++)
        {
            ref_geam(transA,
                     transB,
                     M,
                     N,
                     (T*)h_alpha,
                     hA[b],
                     lda,
                     (T*)h_beta,
                     hB[b],
                     ldb,
                     hC_gold[b],
                     ldc);
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_1
                    = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count);
            }
        }

        if(arg.pointer_mode_device)
        {
            // fetch GPU
            CHECK_HIP_ERROR(hC.transfer_from(dC));

            if(arg.unit_check)
            {
                unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
            }

            if(arg.norm_check)
            {
                rocblas_error_2
                    = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count);
            }
        }

        // inplace check for dC == dA
        if(arg.pointer_mode_host)
        {
            bool invalid_size_in_place = lda != ldc || transA != rocblas_operation_none;

            if((lda == ldc) && (transA == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hA));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            DAPI_EXPECT(invalid_size_in_place ? rocblas_status_invalid_size
                                              : rocblas_status_success,
                        rocblas_geam_strided_batched_fn,
                        (handle,
                         transA,
                         transB,
                         M,
                         N,
                         &alpha,
                         dC_in_place,
                         lda,
                         stride_a,
                         &beta,
                         dB,
                         ldb,
                         stride_b,
                         dC_in_place,
                         ldc,
                         stride_c,
                         batch_count));

            if(!invalid_size_in_place)
            {
                CHECK_HIP_ERROR(hC.transfer_from(dC_in_place));
                // dA was clobbered by dC_in_place, so copy hA back to dA
                CHECK_HIP_ERROR(dA.transfer_from(hA));

                // reference calculation
                for(size_t b = 0; b < batch_count; b++)
                {
                    ref_geam(transA,
                             transB,
                             M,
                             N,
                             (T*)h_alpha,
                             hA_copy[b],
                             lda,
                             (T*)h_beta,
                             hB_copy[b],
                             ldb,
                             hC_gold[b],
                             ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error
                        = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count);
                }
            }
        }

        // inplace check for dC == dB
        if(arg.pointer_mode_host)
        {
            bool invalid_size_in_place = ldb != ldc || transB != rocblas_operation_none;

            if((ldb == ldc) && (transB == rocblas_operation_none))
                CHECK_HIP_ERROR(dC_in_place.transfer_from(hB));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            DAPI_EXPECT(invalid_size_in_place ? rocblas_status_invalid_size
                                              : rocblas_status_success,
                        rocblas_geam_strided_batched_fn,
                        (handle,
                         transA,
                         transB,
                         M,
                         N,
                         &alpha,
                         dA,
                         lda,
                         stride_a,
                         &beta,
                         dC_in_place,
                         ldb,
                         stride_b,
                         dC_in_place,
                         ldc,
                         stride_c,
                         batch_count));

            if(!invalid_size_in_place)
            {
                CHECK_HIP_ERROR(hC.transfer_from(dC_in_place));
                // reference calculation
                for(size_t b = 0; b < batch_count; b++)
                {
                    ref_geam(transA,
                             transB,
                             M,
                             N,
                             (T*)h_alpha,
                             hA_copy[b],
                             lda,
                             (T*)h_beta,
                             hB_copy[b],
                             ldb,
                             hC_gold[b],
                             ldc);
                }

                if(arg.unit_check)
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
                }

                if(arg.norm_check)
                {
                    rocblas_error
                        = norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count);
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

            DAPI_DISPATCH(rocblas_geam_strided_batched_fn,
                          (handle,
                           transA,
                           transB,
                           M,
                           N,
                           &alpha,
                           dA,
                           lda,
                           stride_a,
                           &beta,
                           dB,
                           ldb,
                           stride_b,
                           dC,
                           ldc,
                           stride_c,
                           batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
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
