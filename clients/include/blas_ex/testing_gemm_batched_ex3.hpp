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

#include "frequency_monitor.hpp"
#include "testing_common.hpp"
#include "testing_gemm_ex3.hpp"

/* ============================================================================================ */
template <typename TiA, typename TiB, typename To, typename Tc>
void testing_gemm_batched_ex3_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_batched_ex3_fn
            = arg.api & c_API_FORTRAN ? rocblas_gemm_batched_ex3_fortran : rocblas_gemm_batched_ex3;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const rocblas_int M = 100;
        const rocblas_int N = 100;
        const rocblas_int K = 100;

        const rocblas_int lda = 100;
        const rocblas_int ldb = 100;
        const rocblas_int ldc = 100;
        const rocblas_int ldd = 100;

        const rocblas_int batch_count = 1;

        const rocblas_datatype    a_type                 = arg.a_type;
        const rocblas_datatype    b_type                 = arg.b_type;
        const rocblas_datatype    c_type                 = arg.c_type;
        const rocblas_datatype    d_type                 = arg.d_type;
        const rocblas_computetype composite_compute_type = arg.composite_compute_type;

        DEVICE_MEMCHECK(device_vector<float>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<float>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<float>, zero_d, (1));

        const float alpha_h(1), beta_h(1), zero_h(0);

        const float* alpha = &alpha_h;
        const float* beta  = &beta_h;
        const float* zero  = &zero_h;
        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        const rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t                 solution_index = 0;
        rocblas_int             flags          = 0;

        rocblas_int A_row = transA == rocblas_operation_none ? M : std::max(K, 1);
        rocblas_int A_col = transA == rocblas_operation_none ? std::max(K, 1) : M;
        rocblas_int B_row = transB == rocblas_operation_none ? std::max(K, 1) : N;
        rocblas_int B_col = transB == rocblas_operation_none ? N : std::max(K, 1);

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // allocate memory on device
        DEVICE_MEMCHECK(device_batch_matrix<TiA>, dA, (A_row, A_col, lda, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<TiB>, dB, (B_row, B_col, ldb, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<To>, dC, (M, N, ldc, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<To>, dD, (M, N, ldd, batch_count));

        // host
        HOST_MEMCHECK(host_batch_matrix<To>, hC, (M, N, ldc, batch_count));
        rocblas_seedrand();
        rocblas_init_matrix<To>(
            hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
        dC.transfer_from(hC);

        if(rocblas_handle(handle)->getArch() < 942 || rocblas_handle(handle)->getArch() >= 1000)
        {
            // check for invalid arch
            EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              K,
                                                              alpha,
                                                              dA.ptr_on_device(),
                                                              a_type,
                                                              lda,
                                                              dB.ptr_on_device(),
                                                              b_type,
                                                              ldb,
                                                              beta,
                                                              dC.ptr_on_device(),
                                                              c_type,
                                                              ldc,
                                                              dD.ptr_on_device(),
                                                              d_type,
                                                              ldd,
                                                              batch_count,
                                                              composite_compute_type,
                                                              algo,
                                                              solution_index,
                                                              flags),
                                  rocblas_status_arch_mismatch);

            return;
        }

        // check for invalid enum
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          (rocblas_operation)rocblas_side_both,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_value);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          (rocblas_operation)rocblas_side_both,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_value);

        // check for invalid size
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          -1,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          -1,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          -1,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        // check for invalid leading dimension
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          rocblas_operation_none,
                                                          rocblas_operation_none,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          M - 1,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          rocblas_operation_none,
                                                          rocblas_operation_none,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          K - 1,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          rocblas_operation_transpose,
                                                          rocblas_operation_transpose,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          K - 1,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          rocblas_operation_transpose,
                                                          rocblas_operation_transpose,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          N - 1,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          M - 1,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          M - 1,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        // check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(nullptr,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          alpha,
                                                          dA.ptr_on_device(),
                                                          a_type,
                                                          lda,
                                                          dB.ptr_on_device(),
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dC.ptr_on_device(), // aliased C
                                                          d_type,
                                                          ldc + 1,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_invalid_size);

        // If batch_count==0, then all pointers can be nullptr without error
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          0,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          0,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_success);

        // If M==0, then all pointers can be nullptr without issue
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          0,
                                                          N,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_success);

        // If N==0, then all pointers can be nullptr without issue
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          0,
                                                          K,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          c_type,
                                                          ldc,
                                                          nullptr,
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_success);

        // If alpha==0 then A, B can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          zero,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_success);

        // the following tests still output to D

        // If K==0, then alpha, A and B can both be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          0,
                                                          nullptr,
                                                          nullptr,
                                                          a_type,
                                                          lda,
                                                          nullptr,
                                                          b_type,
                                                          ldb,
                                                          beta,
                                                          dC.ptr_on_device(),
                                                          c_type,
                                                          ldc,
                                                          dD.ptr_on_device(),
                                                          d_type,
                                                          ldd,
                                                          batch_count,
                                                          composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              rocblas_status_success);

        // alpha==0 && beta==1 must still copy C to D so no quick return

        //TODO
        // // If alpha==0 && beta==0 then A, B and C can be nullptr without issue.
        // EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle, transA, transB, M, N, K, zero,
        // nullptr, a_type, lda, nullptr, b_type, ldb, zero, nullptr, c_type, ldc,
        // dD.ptr_on_device(), d_type, ldd, composite_compute_type, algo, solution_index, flags), rocblas_status_success);
    }
}

template <typename TiA, typename TiB, typename To, typename Tc>
void testing_gemm_batched_ex3(const Arguments& arg)
{
    auto rocblas_gemm_batched_ex3_fn
        = arg.api & c_API_FORTRAN ? rocblas_gemm_batched_ex3_fortran : rocblas_gemm_batched_ex3;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool stochastic_rounding = flags & rocblas_gemm_flags_stochastic_rounding;

    if(stochastic_rounding)
    {
        std::random_device                      rd;
        std::mt19937                            gen(rd());
        std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
        uint32_t                                seedA = 0, seedB = 0;

        seedA = distribution(gen);
        seedB = distribution(gen);

        int setenv_status;

        setenv_status = setenv("SR_SEED_A", std::to_string(seedA).c_str(), false);

#ifdef GOOGLE_TEST
        ASSERT_EQ(setenv_status, 0);
#endif

        setenv_status = setenv("SR_SEED_B", std::to_string(seedB).c_str(), false);

#ifdef GOOGLE_TEST
        ASSERT_EQ(setenv_status, 0);
#endif
    }

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used      = 0.0;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle{arg};
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    int                  M = arg.M, N = arg.N, K = arg.K;
    int                  lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row       = transA == rocblas_operation_none ? M : std::max(K, 1);
    auto                 A_col       = transA == rocblas_operation_none ? std::max(K, 1) : M;
    auto                 B_row       = transB == rocblas_operation_none ? std::max(K, 1) : N;
    auto                 B_col       = transB == rocblas_operation_none ? N : std::max(K, 1);
    int                  batch_count = arg.batch_count;
    auto                 d_type      = arg.d_type;

    // Quick-return or error sizes
    // Note: K==0 is not an early exit, since we still must multiply C by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;

    // NOTE: for mixed of f8 bf8, we will consider it is_bf8 as true since bf8 has less precision than f8
    // TODO: we have not considered f8/b8 with other non-f8 types yet!!
    bool is_bf8
        = arg.composite_compute_type == rocblas_compute_type_bf8_f8_f32
          || arg.composite_compute_type == rocblas_compute_type_f8_bf8_f32
          || arg.composite_compute_type == rocblas_compute_type_bf8_bf8_f32
          || (arg.composite_compute_type == rocblas_compute_type_f32
              && (arg.a_type == rocblas_datatype_bf8_r || arg.b_type == rocblas_datatype_bf8_r));
    bool is_f8
        = arg.composite_compute_type == rocblas_compute_type_f8_f8_f32
          || (arg.composite_compute_type == rocblas_compute_type_f32
              && (arg.a_type == rocblas_datatype_f8_r && arg.b_type == rocblas_datatype_f8_r));
    bool is_8bit_float = is_f8 || is_bf8;

    if(invalid_size || !M || !N || !batch_count)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_batched_ex3_fn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &h_alpha_Tc,
                                                          nullptr,
                                                          arg.a_type,
                                                          lda,
                                                          nullptr,
                                                          arg.b_type,
                                                          ldb,
                                                          nullptr,
                                                          nullptr,
                                                          arg.c_type,
                                                          ldc,
                                                          nullptr,
                                                          arg.d_type,
                                                          ldd,
                                                          batch_count,
                                                          arg.composite_compute_type,
                                                          algo,
                                                          solution_index,
                                                          flags),
                              invalid_size ? rocblas_status_invalid_size : rocblas_status_success);
        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        DEVICE_MEMCHECK(device_batch_vector<TiA>, dA, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<TiB>, dB, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<To>, dC, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<To>, dD, (1, 1, batch_count));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex3_fn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        &h_alpha_Tc,
                                                        dA.ptr_on_device(),
                                                        arg.a_type,
                                                        lda,
                                                        dB.ptr_on_device(),
                                                        arg.b_type,
                                                        ldb,
                                                        &h_beta_Tc,
                                                        dC.ptr_on_device(),
                                                        arg.c_type,
                                                        ldc,
                                                        dD.ptr_on_device(),
                                                        arg.d_type,
                                                        ldd,
                                                        batch_count,
                                                        arg.composite_compute_type,
                                                        algo,
                                                        solution_index,
                                                        flags));
        return;
    }
#endif

    // update after invalid checks
    if(!arg.outofplace)
    {
        // c alias of d must be identical descriptors
        ldd    = ldc;
        d_type = arg.c_type;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    HOST_MEMCHECK(host_batch_matrix<TiA>, hA, (A_row, A_col, lda, batch_count));
    HOST_MEMCHECK(host_batch_matrix<TiB>, hB, (B_row, B_col, ldb, batch_count));
    HOST_MEMCHECK(host_batch_matrix<To>, hC, (M, N, ldc, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_batch_matrix<TiA>, dA, (A_row, A_col, lda, batch_count));
    DEVICE_MEMCHECK(device_batch_matrix<TiB>, dB, (B_row, B_col, ldb, batch_count));
    // if C!=D, allocate C and D normally
    // if C==D, allocate C big enough for the larger of C and D; D points to C
    DEVICE_MEMCHECK(device_batch_matrix<To>, dC, (M, N, ldc, batch_count));
    device_batch_matrix<To> dD = (arg.outofplace) ? device_batch_matrix<To>(M, N, ldd, batch_count)
                                                  : device_batch_matrix<To>(0, 1, 1, 1);
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    device_batch_matrix<To>& dDref = (arg.outofplace) ? dD : dC;

    DEVICE_MEMCHECK(device_vector<Tc>, d_alpha_Tc, (1));
    DEVICE_MEMCHECK(device_vector<Tc>, d_beta_Tc, (1));

    // Initialize data on host memory
    rocblas_init_matrix<TiA>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<TiB>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        HOST_MEMCHECK(host_batch_matrix<To>, hD_1, (M, N, ldd, batch_count));
        HOST_MEMCHECK(host_batch_matrix<To>, hD_2, (M, N, ldd, batch_count));
        HOST_MEMCHECK(host_batch_matrix<To>, hD_gold, (M, N, ldd, batch_count));

        // Initialize data on host memory
        for(int b = 0; b < batch_count; b++)
        {
            rocblas_init_nan<To>(hD_1[b], M, N, ldd);
            rocblas_init_nan<To>(hD_gold[b], M, N, ldd);
        }

        hD_2.copy_from(hD_1);

        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        handle.pre_test(arg);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex3_fn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        &h_alpha_Tc,
                                                        dA.ptr_on_device(),
                                                        arg.a_type,
                                                        lda,
                                                        dB.ptr_on_device(),
                                                        arg.b_type,
                                                        ldb,
                                                        &h_beta_Tc,
                                                        dC.ptr_on_device(),
                                                        arg.c_type,
                                                        ldc,
                                                        dDref.ptr_on_device(),
                                                        d_type,
                                                        ldd,
                                                        batch_count,
                                                        arg.composite_compute_type,
                                                        algo,
                                                        solution_index,
                                                        flags));
        handle.post_test(arg);
        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_1.transfer_from(dDref));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex3_fn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        d_alpha_Tc,
                                                        dA.ptr_on_device(),
                                                        arg.a_type,
                                                        lda,
                                                        dB.ptr_on_device(),
                                                        arg.b_type,
                                                        ldb,
                                                        d_beta_Tc,
                                                        dC.ptr_on_device(),
                                                        arg.c_type,
                                                        ldc,
                                                        dDref.ptr_on_device(),
                                                        d_type,
                                                        ldd,
                                                        batch_count,
                                                        arg.composite_compute_type,
                                                        algo,
                                                        solution_index,
                                                        flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hD_2.transfer_from(dDref));

        // copy C matrix into D matrix
        copy_matrix_with_different_leading_dimensions(hC, hD_gold);

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

#define TEST_PARM                                                                                 \
    handle, transA, transB, M, N, K, h_alpha_Tc, dA[b], hA[b], lda, dB[b], hB[b], ldb, h_beta_Tc, \
        hC[b], ldc, hD_gold[b], ldd, stochastic_rounding

        for(rocblas_int b = 0; b < batch_count; b++)
        {

            if((arg.a_type == rocblas_datatype_f8_r && arg.b_type == rocblas_datatype_f8_r
                && arg.c_type == arg.d_type
                && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                    || arg.c_type == rocblas_datatype_f32_r || arg.c_type == rocblas_datatype_f16_r)
                && arg.composite_compute_type == rocblas_compute_type_f32)
               || arg.composite_compute_type == rocblas_compute_type_f8_f8_f32)
                call_trusted_gemm_f8<TiA, TiB, To, rocblas_f8, rocblas_f8, float>(TEST_PARM);
            else if((arg.a_type == rocblas_datatype_bf8_r && arg.b_type == rocblas_datatype_bf8_r
                     && arg.c_type == arg.d_type
                     && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                         || arg.c_type == rocblas_datatype_f32_r
                         || arg.c_type == rocblas_datatype_f16_r)
                     && arg.composite_compute_type == rocblas_compute_type_f32)
                    || arg.composite_compute_type == rocblas_compute_type_bf8_bf8_f32)
                call_trusted_gemm_f8<TiA, TiB, To, rocblas_bf8, rocblas_bf8, float>(TEST_PARM);
            else if((arg.a_type == rocblas_datatype_f8_r && arg.b_type == rocblas_datatype_bf8_r
                     && arg.c_type == arg.d_type
                     && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                         || arg.c_type == rocblas_datatype_f32_r
                         || arg.c_type == rocblas_datatype_f16_r)
                     && arg.composite_compute_type == rocblas_compute_type_f32)
                    || arg.composite_compute_type == rocblas_compute_type_f8_bf8_f32)
                call_trusted_gemm_f8<TiA, TiB, To, rocblas_f8, rocblas_bf8, float>(TEST_PARM);
            else if((arg.a_type == rocblas_datatype_bf8_r && arg.b_type == rocblas_datatype_f8_r
                     && arg.c_type == arg.d_type
                     && (arg.c_type == rocblas_datatype_f8_r || arg.c_type == rocblas_datatype_bf8_r
                         || arg.c_type == rocblas_datatype_f32_r
                         || arg.c_type == rocblas_datatype_f16_r)
                     && arg.composite_compute_type == rocblas_compute_type_f32)
                    || arg.composite_compute_type == rocblas_compute_type_bf8_f8_f32)
                call_trusted_gemm_f8<TiA, TiB, To, rocblas_bf8, rocblas_f8, float>(TEST_PARM);
            else
                rocblas_cout << "ERROR Trusted combo not found " << std::endl;
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.unit_check)
        {
            // NOTE: f8 can represent integer number from 1.0 to 16.0 and bf8 can represent from 1.0 to 8.0.
            // it may only be safe to match the output exactly when K is very small
            // if(is_8bit_float && K > 240)
            //     rocblas_cout << "******* Applying unit_check with large K is not a good idea "
            //                     "(clipped to max_value or overflow)"
            //                  << std::endl;

            if((std::is_same<Tc, rocblas_half>{} && K > 10000) || (is_8bit_float && K > 16))
            {
                // Note: accumulator type for f8/bf8 is still f32. So, the tol would be 0.
                // However, downcasting the output to f8 or f16 may have rounding errors.... near_check should take care it
                const double tol = K * sum_error_tolerance<Tc>;
                //rocblas_cout << "********** Applying unit->near check, K = " << K << " tol = " << tol << std::endl;
                // rocblas_cout << "********** Applying unit->near check, K = " << K << std::endl;
                near_check_general<To, To>(M, N, ldd, hD_gold, hD_1, batch_count, tol);
                near_check_general<To, To>(M, N, ldd, hD_gold, hD_2, batch_count, tol);
            }
            else
            {
                // rocblas_cout << "********** Applying unit check, K = " << K << std::endl;
                unit_check_general<To, To>(M, N, ldd, hD_gold, hD_1, batch_count);
                unit_check_general<To, To>(M, N, ldd, hD_gold, hD_2, batch_count);
            }
        }
        else // can apply both res_check and norm_check if not unit_check
        {
            if(arg.res_check)
            {
                // NOTE: Ti and To can be non-f8 types, but the TcA or TcB still can be f8/bf8 types.
                // We need to consider eps for f8/bf8 since we have lost precision when we downcast the input to f8 (TcA, TcB)
                double eps = is_f8 ? get_epsilon<rocblas_f8>()
                                   : (is_bf8 ? get_epsilon<rocblas_bf8>() : get_epsilon<To>());
                // NOTE: Accumulation is in f32 and the eps of f32 is very small when compared with eps of f8. So, we are not
                // considering the reduction errors in tolerance here
                //double tolerance = 100 * sqrt(K);
                //double tolerance = 2 * K;
                double tolerance = 2;
                // rocblas_cout << "********** Applying res_check, K = " << K
                //              << " tol*eps = " << tolerance * eps << std::endl;
                for(rocblas_int i = 0; i < batch_count; i++)
                    res_check<To, To>(M, N, ldd, hD_gold[i], hD_1[i], tolerance * eps);

                for(rocblas_int i = 0; i < batch_count; i++)
                    res_check<To, To>(M, N, ldd, hD_gold[i], hD_2[i], tolerance * eps);
            }

            if(arg.norm_check)
            {
                // NOTE: NORM calcualtion is based on the To values, so we are considering eps for To type here
                // rocblas_cout << "********** Applying norm check, M = " << M << " N = " << N
                //              << " K = " << K << std::endl;
                auto   err1 = std::abs(norm_check_general('O', hD_gold, hD_1));
                double eps  = is_f8 ? get_epsilon<rocblas_f8>()
                                    : (is_bf8 ? get_epsilon<rocblas_bf8>() : get_epsilon<To>());
                double tolerance
                    = 50; // threshold in lapack style testing, value 50 or 100 in some libraries
                size_t minMN = M < N ? M : N;
#ifdef GOOGLE_TEST
                ASSERT_LE(err1, tolerance * eps * minMN);
#endif
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;

                auto err2 = std::abs(norm_check_general('O', hD_gold, hD_2));
#ifdef GOOGLE_TEST
                ASSERT_LE(err1, tolerance * eps * minMN);
#endif
                rocblas_error = err2 > rocblas_error ? err2 : rocblas_error;
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        for(int i = 0; i < number_cold_calls; i++)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex3_fn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha_Tc,
                                                            dA.ptr_on_device(),
                                                            arg.a_type,
                                                            lda,
                                                            dB.ptr_on_device(),
                                                            arg.b_type,
                                                            ldb,
                                                            &h_beta_Tc,
                                                            dC.ptr_on_device(),
                                                            arg.c_type,
                                                            ldc,
                                                            dDref.ptr_on_device(),
                                                            d_type,
                                                            ldd,
                                                            batch_count,
                                                            arg.composite_compute_type,
                                                            algo,
                                                            solution_index,
                                                            flags));
        }

        int         number_hot_calls = arg.iters;
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_batched_ex3_fn(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha_Tc,
                                        dA.ptr_on_device(),
                                        arg.a_type,
                                        lda,
                                        dB.ptr_on_device(),
                                        arg.b_type,
                                        ldb,
                                        &h_beta_Tc,
                                        dC.ptr_on_device(),
                                        arg.c_type,
                                        ldc,
                                        dDref.ptr_on_device(),
                                        d_type,
                                        ldd,
                                        batch_count,
                                        arg.composite_compute_type,
                                        algo,
                                        solution_index,
                                        flags);
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_beta,
                      e_ldb,
                      e_ldc,
                      e_ldd,
                      e_batch_count>{}
            .log_args<To>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          gemm_gflop_count<Tc>(M, N, K),
                          ArgumentLogging::NA_value,
                          cpu_time_used,
                          rocblas_error);
    }
}
