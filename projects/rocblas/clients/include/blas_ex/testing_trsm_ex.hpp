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

#define ERROR_EPS_MULTIPLIER 40
#define RESIDUAL_EPS_MULTIPLIER 40
#define TRSM_BLOCK 128

template <typename T>
void printMatrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
{
    rocblas_cout << "---------- " << name << " ----------\n";
    int max_size = 3;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            rocblas_cout << A[i + j * lda] << " ";
        rocblas_cout << "\n";
    }
}

template <typename T>
void testing_trsm_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_trsm_ex_fn = arg.api & c_API_FORTRAN ? rocblas_trsm_ex_fortran : rocblas_trsm_ex;

    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        const rocblas_int M   = 100;
        const rocblas_int N   = 100;
        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;

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

        rocblas_int K        = side == rocblas_side_left ? M : N;
        size_t      sizeInvA = TRSM_BLOCK * K;

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<T>, dA, (K, K, lda));
        DEVICE_MEMCHECK(device_matrix<T>, dB, (M, N, ldb));
        DEVICE_MEMCHECK(device_vector<T>, dinvA, (TRSM_BLOCK, TRSM_BLOCK, K));

        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(nullptr,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 N,
                                                 alpha,
                                                 dA,
                                                 lda,
                                                 dB,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_f32_r),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 N,
                                                 nullptr,
                                                 dA,
                                                 lda,
                                                 dB,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_f32_r),
                              rocblas_status_invalid_pointer);

        if(pointer_mode == rocblas_pointer_mode_host)
        {
            EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     N,
                                                     alpha,
                                                     nullptr,
                                                     lda,
                                                     dB,
                                                     ldb,
                                                     dinvA,
                                                     sizeInvA,
                                                     rocblas_datatype_f32_r),
                                  rocblas_status_invalid_pointer);
        }

        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 N,
                                                 alpha,
                                                 dA,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_f32_r),
                              rocblas_status_invalid_pointer);

        // If M==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 0,
                                                 N,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_f32_r),
                              rocblas_status_success);

        // If N==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_f32_r),
                              rocblas_status_success);

        // If alpha==0, then A can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 N,
                                                 zero,
                                                 nullptr,
                                                 lda,
                                                 dB,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_f32_r),
                              rocblas_status_success);

        // Unsupported datatype
        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 N,
                                                 alpha,
                                                 dA,
                                                 lda,
                                                 dB,
                                                 ldb,
                                                 dinvA,
                                                 sizeInvA,
                                                 rocblas_datatype_bf16_r),
                              rocblas_status_not_implemented);
    }
}

template <typename T>
void testing_trsm_ex(const Arguments& arg)
{
    auto rocblas_trsm_ex_fn = arg.api & c_API_FORTRAN ? rocblas_trsm_ex_fortran : rocblas_trsm_ex;

    rocblas_int M   = arg.M;
    rocblas_int N   = arg.N;
    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;

    char char_side   = arg.side;
    char char_uplo   = arg.uplo;
    char char_transA = arg.transA;
    char char_diag   = arg.diag;
    T    alpha_h     = arg.alpha;

    rocblas_side      side   = char2rocblas_side(char_side);
    rocblas_fill      uplo   = char2rocblas_fill(char_uplo);
    rocblas_operation transA = char2rocblas_operation(char_transA);
    rocblas_diagonal  diag   = char2rocblas_diagonal(char_diag);

    rocblas_int K      = side == rocblas_side_left ? M : N;
    size_t      size_A = lda * size_t(K);
    size_t      size_B = ldb * size_t(N);

    rocblas_local_handle handle{arg};

    // check here to prevent undefined memory allocation error
    bool invalid_size = M < 0 || N < 0 || lda < K || ldb < M;
    if(invalid_size)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        EXPECT_ROCBLAS_STATUS(rocblas_trsm_ex_fn(handle,
                                                 side,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 N,
                                                 nullptr,
                                                 nullptr,
                                                 lda,
                                                 nullptr,
                                                 ldb,
                                                 nullptr,
                                                 TRSM_BLOCK * K,
                                                 arg.compute_type),
                              rocblas_status_invalid_size);
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_matrix<T>, hA, (K, K, lda));
    HOST_MEMCHECK(host_matrix<T>, hAAT, (K, K, lda));
    HOST_MEMCHECK(host_matrix<T>, hB, (M, N, ldb));
    HOST_MEMCHECK(host_matrix<T>, hX, (M, N, ldb));
    HOST_MEMCHECK(host_matrix<T>, hXorB_1, (M, N, ldb));
    HOST_MEMCHECK(host_matrix<T>, hXorB_2, (M, N, ldb));
    HOST_MEMCHECK(host_matrix<T>, cpuXorB, (M, N, ldb));
    HOST_MEMCHECK(host_matrix<T>, invATemp1, (TRSM_BLOCK, TRSM_BLOCK, K));
    HOST_MEMCHECK(host_matrix<T>, hinvAI, (TRSM_BLOCK, TRSM_BLOCK, K));

    // Allocate device memory
    DEVICE_MEMCHECK(device_matrix<T>, dA, (K, K, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dXorB, (M, N, ldb));
    DEVICE_MEMCHECK(device_matrix<T>, dinvA, (TRSM_BLOCK, TRSM_BLOCK, K));
    DEVICE_MEMCHECK(device_vector<T>, alpha_d, (1));

    // Initialize data on host memory
    rocblas_init_matrix(hA,
                        arg,
                        rocblas_client_never_set_nan,
                        rocblas_client_diagonally_dominant_triangular_matrix,
                        true);
    rocblas_init_matrix(
        hX, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, false, true);
    hB = hX;

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(diag == rocblas_diagonal_unit)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, K);
    }

    // Calculate hB = hA*hX;
    ref_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA, lda, hB, ldb);

    hXorB_1 = hB; // hXorB <- B
    hXorB_2 = hB; // hXorB <- B
    cpuXorB = hB; // cpuXorB <- B

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

    rocblas_int stride_A    = TRSM_BLOCK * lda + TRSM_BLOCK;
    rocblas_int stride_invA = TRSM_BLOCK * TRSM_BLOCK;

    int blocks = K / TRSM_BLOCK;

    double max_err_1 = 0.0;
    double max_err_2 = 0.0;
    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used  = 0.0;
    double error_eps_multiplier    = ERROR_EPS_MULTIPLIER;
    double residual_eps_multiplier = RESIDUAL_EPS_MULTIPLIER;
    double eps                     = std::numeric_limits<real_t<T>>::epsilon();

    if(arg.unit_check || arg.norm_check)
    {
        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

        handle.pre_test(arg);
        if(blocks > 0)
            CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(handle,
                                                                 uplo,
                                                                 diag,
                                                                 TRSM_BLOCK,
                                                                 dA,
                                                                 lda,
                                                                 stride_A,
                                                                 dinvA,
                                                                 TRSM_BLOCK,
                                                                 stride_invA,
                                                                 blocks));

        if(K % TRSM_BLOCK != 0 || blocks == 0)
        {
            int remainder = K - TRSM_BLOCK * blocks;
            if(remainder)
                CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(handle,
                                                                     uplo,
                                                                     diag,
                                                                     remainder,
                                                                     dA + stride_A * blocks,
                                                                     lda,
                                                                     stride_A,
                                                                     dinvA + stride_invA * blocks,
                                                                     TRSM_BLOCK,
                                                                     stride_invA,
                                                                     1));
        }

        CHECK_ROCBLAS_ERROR(rocblas_trsm_ex_fn(handle,
                                               side,
                                               uplo,
                                               transA,
                                               diag,
                                               M,
                                               N,
                                               &alpha_h,
                                               dA,
                                               lda,
                                               dXorB,
                                               ldb,
                                               dinvA,
                                               TRSM_BLOCK * K,
                                               arg.compute_type));
        handle.post_test(arg);
        CHECK_HIP_ERROR(hXorB_1.transfer_from(dXorB));

        // calculate dXorB <- A^(-1) B   rocblas_device_pointer_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_2));
        CHECK_HIP_ERROR(hipMemcpy(alpha_d, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_trsm_ex_fn(handle,
                                               side,
                                               uplo,
                                               transA,
                                               diag,
                                               M,
                                               N,
                                               alpha_d,
                                               dA,
                                               lda,
                                               dXorB,
                                               ldb,
                                               dinvA,
                                               TRSM_BLOCK * K,
                                               arg.compute_type));

        CHECK_HIP_ERROR(hXorB_2.transfer_from(dXorB));

        if(arg.repeatability_check)
        {
            HOST_MEMCHECK(host_matrix<T>, hXorB_copy, (M, N, ldb));
            CHECK_HIP_ERROR(invATemp1.transfer_from(dinvA));

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
                DEVICE_MEMCHECK(device_matrix<T>, dA_copy, (K, K, lda));
                DEVICE_MEMCHECK(device_matrix<T>, dXorB_copy, (M, N, ldb));
                DEVICE_MEMCHECK(device_matrix<T>, dinvA_copy, (TRSM_BLOCK, TRSM_BLOCK, K));
                DEVICE_MEMCHECK(device_vector<T>, alpha_d_copy, (1));

                CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                CHECK_HIP_ERROR(dinvA_copy.transfer_from(invATemp1));

                CHECK_HIP_ERROR(
                    hipMemcpy(alpha_d_copy, &alpha_h, sizeof(T), hipMemcpyHostToDevice));

                CHECK_ROCBLAS_ERROR(
                    rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                for(int runs = 0; runs < arg.iters; runs++)
                {
                    CHECK_HIP_ERROR(dXorB_copy.transfer_from(hB));
                    CHECK_ROCBLAS_ERROR(rocblas_trsm_ex_fn(handle_copy,
                                                           side,
                                                           uplo,
                                                           transA,
                                                           diag,
                                                           M,
                                                           N,
                                                           alpha_d_copy,
                                                           dA_copy,
                                                           lda,
                                                           dXorB_copy,
                                                           ldb,
                                                           dinvA_copy,
                                                           TRSM_BLOCK * K,
                                                           arg.compute_type));

                    CHECK_HIP_ERROR(hXorB_copy.transfer_from(dXorB_copy));
                    unit_check_general<T>(M, N, ldb, hXorB_2, hXorB_copy);
                }
            }
            return;
        }

        //computed result is in hx_or_b, so forward error is E = hx - hx_or_b
        // calculate vector-induced-norm 1 of matrix E
        max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX, hXorB_1));
        max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hX, hXorB_2));

        //unit test
        trsm_err_res_check<T>(max_err_1, M, error_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, error_eps_multiplier, eps);

        // hx_or_b contains A * (calculated X), so res = A * (calculated x) - b = hx_or_b - hb
        ref_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA, lda, hXorB_1, ldb);
        ref_trmm<T>(side, uplo, transA, diag, M, N, 1.0 / alpha_h, hA, lda, hXorB_2, ldb);

        max_err_1 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_1, hB));
        max_err_2 = rocblas_abs(matrix_norm_1<T>(M, N, ldb, hXorB_2, hB));

        //unit test
        trsm_err_res_check<T>(max_err_1, M, residual_eps_multiplier, eps);
        trsm_err_res_check<T>(max_err_2, M, residual_eps_multiplier, eps);
    }

    if(arg.timing)
    {
        // GPU rocBLAS
        CHECK_HIP_ERROR(dXorB.transfer_from(hXorB_1));

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds

        CHECK_ROCBLAS_ERROR(
            rocblas_trsm<T>(handle, side, uplo, transA, diag, M, N, &alpha_h, dA, lda, dXorB, ldb));

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        // CPU cblas
        cpu_time_used = get_time_us_no_sync();

        ref_trsm<T>(side, uplo, transA, diag, M, N, alpha_h, hA, lda, cpuXorB, ldb);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        ArgumentModel<e_side, e_uplo, e_transA, e_diag, e_M, e_N, e_alpha, e_lda, e_ldb>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         trsm_gflop_count<T>(M, N, K),
                         ArgumentLogging::NA_value,
                         cpu_time_used,
                         max_err_1,
                         max_err_2);
    }
}
