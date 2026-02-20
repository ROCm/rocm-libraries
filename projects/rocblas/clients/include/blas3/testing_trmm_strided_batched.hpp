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
#include "rocblas_datatype2string.hpp"
#include "rocblas_init.hpp"
#include "rocblas_math.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_random.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "unit.hpp"

template <typename T>
void testing_trmm_strided_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_trmm_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_trmm_strided_batched<T, true>
                                                  : rocblas_trmm_strided_batched<T, false>;
    auto rocblas_trmm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_trmm_strided_batched_64<T, true>
                                                  : rocblas_trmm_strided_batched_64<T, false>;
    // trmm has both inplace and outofplace versions.
    // inplace == true for inplace, inplace == false for outofplace
    bool inplace = !arg.outofplace;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const int64_t M           = 100;
        const int64_t N           = 101;
        const int64_t lda         = 101;
        const int64_t ldb         = 101;
        const int64_t ldc         = 101;
        const int64_t ldOut       = inplace ? ldb : ldc;
        const int64_t batch_count = 2;

        DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<T>, zero_d, (1));

        const T alpha_h(1), zero_h(0);

        const T* alpha = &alpha_h;
        const T* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_side      side   = rocblas_side_left;
        const rocblas_fill      uplo   = rocblas_fill_upper;
        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_diagonal  diag   = rocblas_diagonal_non_unit;

        int64_t              K        = side == rocblas_side_left ? M : N;
        const rocblas_stride stride_a = lda * K;
        const rocblas_stride stride_b = ldb * N;
        const rocblas_stride stride_c = ldc * N;

        // Allocate device memory
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (K, K, lda, stride_a, batch_count));
        DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dB, (M, N, ldb, stride_b, batch_count));

        int64_t dC_M   = inplace ? 1 : M;
        int64_t dC_N   = inplace ? 1 : N;
        int64_t dC_ldc = inplace ? 1 : ldc;

        DEVICE_MEMCHECK(
            device_strided_batch_matrix<T>, dC, (dC_M, dC_N, dC_ldc, stride_c, batch_count));

        device_strided_batch_matrix<T>* dOut = inplace ? &dB : &dC;

        // check for invalid enum
        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     rocblas_side_both,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     (rocblas_fill)rocblas_side_both,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     (rocblas_operation)rocblas_side_both,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_value,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     (rocblas_diagonal)rocblas_side_both,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        // check for invalid size
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     -1,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     -1,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        // check for invalid leading dimension
        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     M - 1,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     M - 1,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     rocblas_side_left,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     M - 1,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_size,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     rocblas_side_right,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     N - 1,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        // check that nullpointer gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
        DAPI_EXPECT(rocblas_status_invalid_handle,
                    rocblas_trmm_strided_batched_fn,
                    (nullptr,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     nullptr,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     nullptr,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     nullptr,
                     ldb,
                     stride_b,
                     *dOut,
                     ldOut,
                     stride_c,
                     batch_count));

        DAPI_EXPECT(rocblas_status_invalid_pointer,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     alpha,
                     dA,
                     lda,
                     stride_a,
                     dB,
                     ldb,
                     stride_b,
                     nullptr,
                     ldOut,
                     stride_c,
                     batch_count));

        // quick return: If alpha==0, then A and B can be nullptr without error
        DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    N,
                    zero,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    ldb,
                    stride_b,
                    *dOut,
                    ldOut,
                    stride_c,
                    batch_count));

        // quick return: If M==0, then all pointers can be nullptr without error
        DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    0,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    ldb,
                    stride_b,
                    nullptr,
                    ldOut,
                    stride_c,
                    batch_count));

        // quick return: If N==0, then all pointers can be nullptr without error
        DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    0,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    ldb,
                    stride_b,
                    nullptr,
                    ldOut,
                    stride_c,
                    batch_count));

        // quick return: If batch_count==0, then all pointers can be nullptr without error
        DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                   (handle,
                    side,
                    uplo,
                    transA,
                    diag,
                    M,
                    N,
                    nullptr,
                    nullptr,
                    lda,
                    stride_a,
                    nullptr,
                    ldb,
                    stride_b,
                    nullptr,
                    ldOut,
                    stride_c,
                    0));

        // in-place only checks
        if(inplace)
        {
            // if inplace, must have ldb == ldc
            DAPI_EXPECT(rocblas_status_invalid_value,
                        rocblas_trmm_strided_batched_fn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         alpha,
                         dA,
                         lda,
                         stride_a,
                         dB,
                         ldb,
                         stride_b,
                         dB,
                         ldb + 1,
                         stride_b,
                         batch_count));
        }
    }
}

template <typename T>
void testing_trmm_strided_batched(const Arguments& arg)
{
    auto rocblas_trmm_strided_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_trmm_strided_batched<T, true>
                                                  : rocblas_trmm_strided_batched<T, false>;
    auto rocblas_trmm_strided_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_trmm_strided_batched_64<T, true>
                                                  : rocblas_trmm_strided_batched_64<T, false>;
    // trmm has both inplace and outofplace versions.
    // inplace == true for inplace, inplace == false for outofplace
    bool inplace = !arg.outofplace;

    int64_t        M           = arg.M;
    int64_t        N           = arg.N;
    int64_t        lda         = arg.lda;
    int64_t        ldb         = arg.ldb;
    int64_t        ldc         = arg.ldc;
    int64_t        ldOut       = inplace ? ldb : ldc;
    rocblas_stride stride_a    = arg.stride_a;
    rocblas_stride stride_b    = arg.stride_b;
    rocblas_stride stride_c    = arg.stride_c;
    int64_t        batch_count = arg.batch_count;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    alpha       = arg.get_alpha<T>();

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    int64_t K = side == rocblas_side_left ? M : N;

    if(stride_a < lda * K)
    {
        rocblas_cout << "WARNING: setting stride_a = lda * (side == rocblas_side_left ? M : N)"
                     << std::endl;
        stride_a = lda * (side == rocblas_side_left ? M : N);
    }
    if(stride_b < ldb * N)
    {
        rocblas_cout << "WARNING: setting stride_b = ldb * N" << std::endl;
        stride_b = ldb * N;
    }
    if(stride_c < ldc * N)
    {
        rocblas_cout << "WARNING: setting stride_c = ldc * N" << std::endl;
        stride_c = ldc * N;
    }

    rocblas_local_handle handle{arg};

    // ensure invalid sizes and quick return checked before pointer check
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M || ldc < M || batch_count < 0;
    if(M == 0 || N == 0 || batch_count == 0 || invalid_size)
    {
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success,
                    rocblas_trmm_strided_batched_fn,
                    (handle,
                     side,
                     uplo,
                     transA,
                     diag,
                     M,
                     N,
                     nullptr,
                     nullptr,
                     lda,
                     stride_a,
                     nullptr,
                     ldb,
                     stride_b,
                     nullptr,
                     ldc,
                     stride_c,
                     batch_count));
        return;
    }

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double err_host = 0.0, err_device = 0.0;

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hA, (K, K, lda, stride_a, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hB, (M, N, ldb, stride_b, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC, (M, N, ldc, stride_c, batch_count));
    HOST_MEMCHECK(host_strided_batch_matrix<T>, hC_gold, (M, N, ldc, stride_c, batch_count));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (1));

    // Allocate device memory
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dA, (K, K, lda, stride_a, batch_count));
    DEVICE_MEMCHECK(device_strided_batch_matrix<T>, dB, (M, N, ldb, stride_b, batch_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));

    //  initialize full random matrix hA and hB
    h_alpha[0] = alpha;

    // Initialize data on host memory
    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_triangular_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));

    // inplace    trmm is given by B <- alpha * op(A) * B so  matrix C is not used
    // outofplace trmm is given by C <- alpha * op(A) * B and matrix C is used
    int64_t dC_M   = inplace ? 1 : M;
    int64_t dC_N   = inplace ? 1 : N;
    int64_t dC_ldc = inplace ? 1 : ldc;

    DEVICE_MEMCHECK(
        device_strided_batch_matrix<T>, dC, (dC_M, dC_N, dC_ldc, stride_c, batch_count));

    device_strided_batch_matrix<T>* dOut = inplace ? &dB : &dC;

    if(inplace)
    {
        // if stride_b != stride_c inplace will fail
        if(stride_b != stride_c)
        {
            return;
        }
        // if ldc != ldb inplace returns rocblas_status_invalid_value
        if(ldb != ldc)
        {
            DAPI_EXPECT(rocblas_status_invalid_value,
                        rocblas_trmm_strided_batched_fn,
                        (handle,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         &h_alpha[0],
                         dA,
                         lda,
                         stride_a,
                         dB,
                         ldb,
                         stride_b,
                         dC,
                         ldc,
                         stride_c,
                         batch_count));
            return;
        }

        hC_gold.copy_from(hB);
    }
    else
    {
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        hC_gold.copy_from(hC);
    }

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

            handle.pre_test(arg);
            DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                       (handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        M,
                        N,
                        &h_alpha[0],
                        dA,
                        lda,
                        stride_a,
                        dB,
                        ldb,
                        stride_b,
                        *dOut,
                        ldOut,
                        stride_c,
                        batch_count));
            handle.post_test(arg);
            CHECK_HIP_ERROR(hC.transfer_from(*dOut));
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR((*dOut).transfer_from(hC_gold));

            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

            DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                       (handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        M,
                        N,
                        d_alpha,
                        dA,
                        lda,
                        stride_a,
                        dB,
                        ldb,
                        stride_b,
                        *dOut,
                        ldOut,
                        stride_c,
                        batch_count));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(
                    host_strided_batch_matrix<T>, hC_copy, (M, N, ldc, stride_c, batch_count));
                CHECK_HIP_ERROR(hC.transfer_from(*dOut));
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
                                    (K, K, lda, stride_a, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dB_copy,
                                    (M, N, ldb, stride_b, batch_count));
                    DEVICE_MEMCHECK(device_strided_batch_matrix<T>,
                                    dC_copy,
                                    (dC_M, dC_N, dC_ldc, stride_c, batch_count));
                    DEVICE_MEMCHECK(device_vector<T>, d_alpha_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(d_alpha_copy.transfer_from(h_alpha));

                    device_strided_batch_matrix<T>* dOut_copy = inplace ? &dB_copy : &dC_copy;

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR((*dOut_copy).transfer_from(hC_gold));
                        DAPI_CHECK(rocblas_trmm_strided_batched_fn,
                                   (handle_copy,
                                    side,
                                    uplo,
                                    transA,
                                    diag,
                                    M,
                                    N,
                                    d_alpha_copy,
                                    dA_copy,
                                    lda,
                                    stride_a,
                                    dB_copy,
                                    ldb,
                                    stride_b,
                                    *dOut_copy,
                                    ldOut,
                                    stride_c,
                                    batch_count));
                        CHECK_HIP_ERROR(hC_copy.transfer_from(*dOut_copy));
                        unit_check_general<T>(M, N, ldc, stride_c, hC, hC_copy, batch_count);
                    }
                }
                return;
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();
        for(int64_t b = 0; b < batch_count; b++)
        {
            ref_trmm<T>(side, uplo, transA, diag, M, N, alpha, hA[b], lda, hB[b], ldb);
        }
        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        // copy B matrix into C matrix
        copy_matrix_with_different_leading_dimensions(hB, hC_gold);

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
                }
            }
            if(arg.norm_check)
            {
                err_host = std::abs(
                    norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count));
            }
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hC.transfer_from(*dOut));

            if(arg.unit_check)
            {
                if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, stride_c, hC_gold, hC, batch_count);
                }
            }
            if(arg.norm_check)
            {
                err_device = std::abs(
                    norm_check_general<T>('F', M, N, ldc, stride_c, hC_gold, hC, batch_count));
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        for(int i = 0; i < total_calls; i++)
        {
            if(i == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            CHECK_ROCBLAS_ERROR(rocblas_trmm_strided_batched_fn(handle,
                                                                side,
                                                                uplo,
                                                                transA,
                                                                diag,
                                                                M,
                                                                N,
                                                                &h_alpha[0],
                                                                dA,
                                                                lda,
                                                                stride_a,
                                                                dB,
                                                                ldb,
                                                                stride_b,
                                                                *dOut,
                                                                ldOut,
                                                                stride_c,
                                                                batch_count));
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side,
                      e_uplo,
                      e_transA,
                      e_diag,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, side),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         err_host,
                         err_device);
    }
}
