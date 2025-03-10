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

#include <rocblas/internal/rocblas-beta.h>

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_batched_ex_fn
            = arg.api & c_API_FORTRAN ? rocblas_gemm_batched_ex_fortran : rocblas_gemm_batched_ex;
        auto rocblas_gemm_batched_ex_fn_64 = arg.api & c_API_FORTRAN
                                                 ? rocblas_gemm_batched_ex_64_fortran
                                                 : rocblas_gemm_batched_ex_64;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const int64_t M = 100;
        const int64_t N = 100;
        const int64_t K = 101;

        const int64_t lda = 101;
        const int64_t ldb = 101;
        const int64_t ldc = 101;
        const int64_t ldd = 101;

        const int64_t batch_count = 1;

        const rocblas_datatype a_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype b_type       = rocblas_type2datatype<Ti>();
        const rocblas_datatype c_type       = rocblas_type2datatype<To>();
        const rocblas_datatype d_type       = rocblas_type2datatype<To>();
        const rocblas_datatype compute_type = rocblas_type2datatype<Tc>();

        DEVICE_MEMCHECK(device_vector<Tc>, alpha_d, (1));
        DEVICE_MEMCHECK(device_vector<Tc>, beta_d, (1));
        DEVICE_MEMCHECK(device_vector<Tc>, zero_d, (1));

        const Tc alpha_h(1), beta_h(1), zero_h(0);

        const Tc* alpha = &alpha_h;
        const Tc* beta  = &beta_h;
        const Tc* zero  = &zero_h;

        if(pointer_mode == rocblas_pointer_mode_device)
        {
            CHECK_HIP_ERROR(hipMemcpy(alpha_d, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            alpha = alpha_d;
            CHECK_HIP_ERROR(hipMemcpy(beta_d, beta, sizeof(*beta), hipMemcpyHostToDevice));
            beta = beta_d;
            CHECK_HIP_ERROR(hipMemcpy(zero_d, zero, sizeof(*zero), hipMemcpyHostToDevice));
            zero = zero_d;
        }

        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t           solution_index = 0;
        uint32_t          flags          = 0;

        int64_t Kmax  = std::max(K, int64_t(1));
        int64_t A_row = transA == rocblas_operation_none ? M : Kmax;
        int64_t A_col = transA == rocblas_operation_none ? Kmax : M;
        int64_t B_row = transB == rocblas_operation_none ? Kmax : N;
        int64_t B_col = transB == rocblas_operation_none ? N : Kmax;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_matrix<Ti>, dA, (A_row, A_col, lda, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<Ti>, dB, (B_row, B_col, ldb, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<To>, dC, (M, N, ldc, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<To>, dD, (M, N, ldd, batch_count));

        // host
        HOST_MEMCHECK(host_batch_matrix<To>, hC, (M, N, ldc, batch_count));
        rocblas_seedrand();
        rocblas_init_matrix<To>(
            hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
        dC.transfer_from(hC);

        // clang-format off
// check for invalid enum
DAPI_EXPECT(rocblas_status_invalid_value, rocblas_gemm_batched_ex_fn, (handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc, nullptr, d_type, ldd, batch_count,
compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_value, rocblas_gemm_batched_ex_fn, (handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc, nullptr, d_type, ldd, batch_count,
compute_type, algo, solution_index, flags));

// check for invalid size
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, -1, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, -1, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, -1,
nullptr, nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, -1, compute_type, algo, solution_index, flags));

// check for invalid leading dimension
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, a_type, M-1, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, K-1, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, a_type, K-1, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, N-1, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, M-1,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, M-1, batch_count, compute_type, algo, solution_index, flags));

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
DAPI_EXPECT(rocblas_status_invalid_handle, rocblas_gemm_batched_ex_fn, (nullptr, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), a_type, lda, dB.ptr_on_device(), b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, nullptr,
dA.ptr_on_device(), a_type, lda, dB.ptr_on_device(), b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, alpha,
nullptr, a_type, lda, dB.ptr_on_device(), b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), a_type, lda, nullptr, b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), a_type, lda, dB.ptr_on_device(), b_type, ldb, nullptr, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), a_type, lda, dB.ptr_on_device(), b_type, ldb, beta, nullptr, c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), a_type, lda, dB.ptr_on_device(), b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

// if D aliased to C then ldd must equal ldc
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA.ptr_on_device(), a_type, lda, dB.ptr_on_device(), b_type, ldb, beta,
dC.ptr_on_device(), c_type, ldc, dC.ptr_on_device(), // aliased C
d_type, ldc + 1, batch_count, compute_type, algo, solution_index, flags));

// If batch_count==0, then all pointers can be nullptr without error
DAPI_CHECK(rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, 0, compute_type, algo, solution_index, flags));

// If M==0, then all pointers can be nullptr without error
DAPI_CHECK(rocblas_gemm_batched_ex_fn, (handle, transA, transB, 0, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

// If N==0, then all pointers can be nullptr without error
DAPI_CHECK(rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, 0, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

// the following tests still output to D

// If K==0, then alpha, A and B can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, 0, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

// If alpha==0, then A and B can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, nullptr, b_type, ldb, beta, dC.ptr_on_device(), c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));

// alpha==0 && beta==1 must still copy C to D so no quick return

// If alpha==0 && beta==0 then A, B and C can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, nullptr, b_type, ldb, zero, nullptr, c_type, ldc,
dD.ptr_on_device(), d_type, ldd, batch_count, compute_type, algo, solution_index, flags));
        // clang-format on
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex_run(const Arguments& arg)
{
    auto rocblas_gemm_batched_ex_fn
        = arg.api & c_API_FORTRAN ? rocblas_gemm_batched_ex_fortran : rocblas_gemm_batched_ex;
    auto rocblas_gemm_batched_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemm_batched_ex_64_fortran : rocblas_gemm_batched_ex_64;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used      = 0.0;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle{arg};
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    rocblas_operation    transB = char2rocblas_operation(arg.transB);
    int64_t              M = arg.M, N = arg.N, K = arg.K;
    int64_t              lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    int64_t              Kmax        = std::max(K, int64_t(1));
    int64_t              A_row       = transA == rocblas_operation_none ? M : Kmax;
    int64_t              A_col       = transA == rocblas_operation_none ? Kmax : M;
    int64_t              B_row       = transB == rocblas_operation_none ? Kmax : N;
    int64_t              B_col       = transB == rocblas_operation_none ? N : Kmax;
    int64_t              batch_count = arg.batch_count;
    rocblas_datatype     d_type      = arg.d_type;

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, math_mode));
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    // Quick-return or error sizes
    // Note: K==0 is not an early exit, since we still must multiply C by beta
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M
                        || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        // clang-format off
        DAPI_EXPECT(invalid_size ? rocblas_status_invalid_size : rocblas_status_success, rocblas_gemm_batched_ex_fn, (
				handle, transA, transB, M, N, K, &h_alpha_Tc,
                                nullptr, arg.a_type, lda,
                                nullptr, arg.b_type, ldb, nullptr,
                                nullptr, arg.c_type, ldc,
                                nullptr, arg.d_type, ldd,
                                batch_count, arg.compute_type, algo, solution_index, flags));
        // clang-format on
        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        DEVICE_MEMCHECK(device_batch_vector<Ti>, dA, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<Ti>, dB, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<To>, dC, (1, 1, batch_count));
        DEVICE_MEMCHECK(device_batch_vector<To>, dD, (1, 1, batch_count));
        // clang-format off
        DAPI_CHECK(rocblas_gemm_batched_ex_fn, (
				handle, transA, transB, M, N, K, &h_alpha_Tc,
                                dA.ptr_on_device(), arg.a_type, lda,
                                dB.ptr_on_device(), arg.b_type, ldb, &h_beta_Tc,
                                dC.ptr_on_device(), arg.c_type, ldc,
                                dD.ptr_on_device(), arg.d_type, ldd,
                                batch_count, arg.compute_type, algo, solution_index, flags));
        // clang-format on
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
    HOST_MEMCHECK(host_batch_matrix<Ti>, hA, (A_row, A_col, lda, batch_count));
    HOST_MEMCHECK(host_batch_matrix<Ti>, hB, (B_row, B_col, ldb, batch_count));
    HOST_MEMCHECK(host_batch_matrix<To>, hC, (M, N, ldc, batch_count));

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tc>, d_alpha_Tc, (1));
    DEVICE_MEMCHECK(device_vector<Tc>, d_beta_Tc, (1));

    // Initialize data on host memory
    rocblas_init_matrix<Ti>(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix<Ti>(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix<To>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

    if(arg.unit_check || arg.norm_check)
    {
        // Allocate device memory
        DEVICE_MEMCHECK(device_batch_matrix<Ti>, dA, (A_row, A_col, lda, batch_count));
        DEVICE_MEMCHECK(device_batch_matrix<Ti>, dB, (B_row, B_col, ldb, batch_count));
        // if C!=D, allocate C and D normally
        // if C==D, allocate C big enough for the larger of C and D; D points to C
        DEVICE_MEMCHECK(device_batch_matrix<To>, dC, (M, N, ldc, batch_count));
        device_batch_matrix<To> dD = (arg.outofplace)
                                         ? device_batch_matrix<To>(M, N, ldd, batch_count)
                                         : device_batch_matrix<To>(0, 1, 1, 1);
        CHECK_DEVICE_ALLOCATION(dD.memcheck());
        device_batch_matrix<To>& dDref = (arg.outofplace) ? dD : dC;

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));

        using To_hpa = std::conditional_t<std::is_same_v<To, rocblas_bfloat16>, float, To>;
        HOST_MEMCHECK(host_batch_matrix<To>, hD_1, (M, N, ldd, batch_count));
        HOST_MEMCHECK(host_batch_matrix<To>, hD_2, (M, N, ldd, batch_count));
        HOST_MEMCHECK(host_batch_matrix<To_hpa>, hD_gold, (M, N, ldd, batch_count));

        // Initialize data on host memory
        for(int64_t b = 0; b < batch_count; b++)
        {
            rocblas_init_nan<To>(hD_1[b], M, N, ldd);
            rocblas_init_nan<To_hpa>(hD_gold[b], M, N, ldd);
        }

        hD_2.copy_from(hD_1);

        if(arg.pointer_mode_host)
        {
            // ROCBLAS rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            // clang-format off
            DAPI_CHECK(rocblas_gemm_batched_ex_fn, (
				    handle, transA, transB, M, N, K, &h_alpha_Tc,
                                    dA.ptr_on_device(), arg.a_type, lda,
                                    dB.ptr_on_device(), arg.b_type, ldb, &h_beta_Tc,
                                    dC.ptr_on_device(), arg.c_type, ldc,
                                    dDref.ptr_on_device(),  d_type, ldd,
                                    batch_count, arg.compute_type, algo, solution_index, flags));
            // clang-format on
            handle.post_test(arg);
            // copy output from device to CPU
            CHECK_HIP_ERROR(hD_1.transfer_from(dDref));
        }

        if(arg.pointer_mode_device)
        {
            // ROCBLAS rocblas_pointer_mode_device
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC));
            CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));
            // clang-format off
            DAPI_CHECK(rocblas_gemm_batched_ex_fn, (
				    handle, transA, transB, M, N, K, d_alpha_Tc,
                                    dA.ptr_on_device(), arg.a_type, lda,
                                    dB.ptr_on_device(), arg.b_type, ldb, d_beta_Tc,
                                    dC.ptr_on_device(), arg.c_type, ldc,
                                    dDref.ptr_on_device(),  d_type, ldd,
                                    batch_count, arg.compute_type, algo, solution_index, flags));
            // clang-format on

            // copy output from device to CPU
            CHECK_HIP_ERROR(hD_2.transfer_from(dDref));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_batch_matrix<To>, hD_2_copy, (M, N, ldd, batch_count));
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
                        device_batch_matrix<Ti>, dA_copy, (A_row, A_col, lda, batch_count));
                    DEVICE_MEMCHECK(
                        device_batch_matrix<Ti>, dB_copy, (B_row, B_col, ldb, batch_count));
                    // if C!=D, allocate C and D normally
                    // if C==D, allocate C big enough for the larger of C and D; D points to C
                    DEVICE_MEMCHECK(device_batch_matrix<To>, dC_copy, (M, N, ldc, batch_count));
                    device_batch_matrix<To> dD_copy
                        = (arg.outofplace) ? device_batch_matrix<To>(M, N, ldd, batch_count)
                                           : device_batch_matrix<To>(0, 1, 1, 1);
                    CHECK_DEVICE_ALLOCATION(dD_copy.memcheck());
                    device_batch_matrix<To>& dDref_copy = (arg.outofplace) ? dD_copy : dC_copy;

                    DEVICE_MEMCHECK(device_vector<Tc>, d_alpha_Tc_copy, (1));
                    DEVICE_MEMCHECK(device_vector<Tc>, d_beta_Tc_copy, (1));

                    // copy data from CPU to device
                    CHECK_HIP_ERROR(dA_copy.transfer_from(hA));
                    CHECK_HIP_ERROR(dB_copy.transfer_from(hB));
                    CHECK_HIP_ERROR(dC_copy.transfer_from(hC));

                    CHECK_HIP_ERROR(
                        hipMemcpy(d_alpha_Tc_copy, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(d_beta_Tc_copy, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));

                    CHECK_ROCBLAS_ERROR(
                        rocblas_set_pointer_mode(handle_copy, rocblas_pointer_mode_device));

                    for(int runs = 0; runs < arg.iters; runs++)
                    {
                        CHECK_HIP_ERROR(dC_copy.transfer_from(hC));

                        // clang-format off
                    DAPI_CHECK(rocblas_gemm_batched_ex_fn, (
					    handle_copy, transA, transB, M, N, K, d_alpha_Tc_copy,
                                            dA_copy.ptr_on_device(), arg.a_type, lda,
                                            dB_copy.ptr_on_device(), arg.b_type, ldb, d_beta_Tc_copy,
                                            dC_copy.ptr_on_device(), arg.c_type, ldc,
                                            dDref_copy.ptr_on_device(),  d_type, ldd,
                                            batch_count, arg.compute_type, algo, solution_index, flags));
                        // clang-format on
                        // copy output from device to CPU
                        CHECK_HIP_ERROR(hD_2_copy.transfer_from(dDref_copy));
                        unit_check_general<To>(M, N, ldd, hD_2, hD_2_copy, batch_count);
                    }
                }
                return;
            }
        }
        // copy C matrix into D matrix
        copy_matrix_with_different_leading_dimensions(hC, hD_gold);

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if(std::is_same<Ti, float>{} && math_mode == rocblas_xf32_xdl_math_op)
        {
            for(int64_t b = 0; b < batch_count; b++)
            {
                type_to_xdl_math_op_type<rocblas_xfloat32, float>(hA[b], hA.nmemb());
                type_to_xdl_math_op_type<rocblas_xfloat32, float>(hB[b], hB.nmemb());
            }
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        for(int64_t b = 0; b < batch_count; b++)
        {
            // clang-format off
            ref_gemm<Ti, To_hpa>(transA, transB, M, N, K, h_alpha_Tc,
                                 hA[b], lda,
                                 hB[b], ldb, h_beta_Tc,
                                 hD_gold[b], ldd);
            // clang-format on
        }

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if((rocblas_handle(handle)->getArchMajor() == 11) && (sizeof(Ti) == 2))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count, tol);
                }
                else if(std::is_same_v<Tc, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<Tc>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count, tol);
                }
                else
                {
                    unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, batch_count);
                }
            }

            if(arg.norm_check)
            {
                auto err1     = std::abs(norm_check_general('F', hD_gold, hD_1));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }
        if(arg.pointer_mode_device)
        {
            if(arg.unit_check)
            {
                if((rocblas_handle(handle)->getArchMajor() == 11) && (sizeof(Ti) == 2))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count, tol);
                }
                else if(std::is_same_v<Tc, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<Tc>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count, tol);
                }
                else
                {
                    unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, batch_count);
                }
            }

            if(arg.norm_check)
            {
                auto err1     = std::abs(norm_check_general('F', hD_gold, hD_2));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }
    }

    if(arg.timing)
    {
        size_t a_size = M * K * batch_count * sizeof(Ti);
        size_t b_size = K * N * batch_count * sizeof(Ti);
        size_t c_size = M * N * batch_count * sizeof(To);
        //      exclude d_size from cached_size calculation because
        //      - for arg.outofplace == false : D == C
        //      - for arg.outofplace == true  : D is write only
        size_t a_b_c_cached_size = a_size + b_size + c_size;

        size_t flush_batch_count = calculate_flush_batch_count(
            arg.flush_batch_count, arg.flush_memory_size, a_b_c_cached_size);

        // Allocate device memory
        DEVICE_MEMCHECK(
            device_batch_matrix<Ti>, dA, (A_row, A_col, lda, batch_count * flush_batch_count));
        DEVICE_MEMCHECK(
            device_batch_matrix<Ti>, dB, (B_row, B_col, ldb, batch_count * flush_batch_count));
        // if C!=D, allocate C and D normally
        // if C==D, allocate C big enough for the larger of C and D; D points to C
        DEVICE_MEMCHECK(device_batch_matrix<To>, dC, (M, N, ldc, batch_count * flush_batch_count));
        device_batch_matrix<To> dD
            = (arg.outofplace) ? device_batch_matrix<To>(M, N, ldd, batch_count * flush_batch_count)
                               : device_batch_matrix<To>(0, 1, 1, 1);
        CHECK_DEVICE_ALLOCATION(dD.memcheck());
        device_batch_matrix<To>& dDref = (arg.outofplace) ? dD : dC;

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.broadcast_one_batch_matrix_from(hA));
        CHECK_HIP_ERROR(dB.broadcast_one_batch_matrix_from(hB));
        CHECK_HIP_ERROR(dC.broadcast_one_batch_matrix_from(hC));

        int number_cold_calls = arg.cold_iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        // if solution index is given check it is valid before benchmarking
        if(algo == rocblas_gemm_algo_solution_index && solution_index != 0)
        {
            uint32_t test_flags = flags | rocblas_gemm_flags_check_solution_index;
            // clang-format off
            DAPI_CHECK(rocblas_gemm_batched_ex_fn, (
                                    handle, transA, transB, M, N, K, &h_alpha_Tc,
                                    dA.ptr_on_device(), arg.a_type, lda,
                                    dB.ptr_on_device(), arg.b_type, ldb, &h_beta_Tc,
                                    dC.ptr_on_device(), arg.c_type, ldc,
                                    dDref.ptr_on_device(),  d_type, ldd,
                                    batch_count, arg.compute_type, algo, solution_index, test_flags));
            // clang-format on
        }

        for(int i = 0; i < number_cold_calls; i++)
        {
            // clang-format off
            DAPI_DISPATCH(rocblas_gemm_batched_ex_fn, (
                                    handle, transA, transB, M, N, K, &h_alpha_Tc,
                                    dA.ptr_on_device(), arg.a_type, lda,
                                    dB.ptr_on_device(), arg.b_type, ldb, &h_beta_Tc,
                                    dC.ptr_on_device(), arg.c_type, ldc,
                                    dDref.ptr_on_device(),  d_type, ldd,
                                    batch_count, arg.compute_type, algo, solution_index, flags));
            // clang-format on
        }

        int         number_hot_calls = arg.iters;
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        FrequencyMonitor& freq_monitor = getFrequencyMonitor();
        freq_monitor.start();
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            int flush_index = (i + 1) % flush_batch_count;
            // clang-format off
            DAPI_DISPATCH(rocblas_gemm_batched_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                       (dA.ptr_on_device() + (flush_index * batch_count)), arg.a_type, lda,
                                       (dB.ptr_on_device() + (flush_index * batch_count)), arg.b_type, ldb, &h_beta_Tc,
                                       (dC.ptr_on_device() + (flush_index * batch_count)), arg.c_type, ldc,
                                       (dDref.ptr_on_device() + (flush_index * batch_count)),  d_type, ldd,
                                       batch_count, arg.compute_type, algo, solution_index, flags));
            // clang-format on
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        freq_monitor.stop();

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

template <typename Ti, typename To, typename Tc>
void testing_gemm_batched_ex(const Arguments& arg)
{
    bool                     compare_solutions = arg.solution_index == -2;
    const Arguments*         arguments         = &arg;
    Arguments                run_arg(arg);
    std::vector<rocblas_int> solutions_that_solve(1, 0);

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);

#ifdef BUILD_WITH_TENSILE // tensile or hipblaslt only for now
    if(compare_solutions && algo == rocblas_gemm_algo_solution_index)
    {
        arguments = &run_arg; // override

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        // fetch all solutions
        rocblas_int             solution_count = 0;
        const rocblas_operation transA         = char2rocblas_operation(arg.transA);
        const rocblas_operation transB         = char2rocblas_operation(arg.transB);
        if(!arg.outofplace)
        {
            run_arg.ldd    = arg.ldc;
            run_arg.d_type = arg.c_type;
        }

        Tc h_alpha_Tc = arg.get_alpha<Tc>();
        Tc h_beta_Tc  = arg.get_beta<Tc>();

#define GEMM_EX_SOL_ARGS                                                                    \
    handle, transA, transB, arg.M, arg.N, arg.K, &h_alpha_Tc, nullptr, arg.a_type, arg.lda, \
        nullptr, arg.b_type, arg.ldb, &h_beta_Tc, nullptr, arg.c_type, arg.ldc, nullptr,    \
        run_arg.d_type, run_arg.ldd, run_arg.batch_count, arg.compute_type, algo

        CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions(
            GEMM_EX_SOL_ARGS, rocblas_gemm_flags_none, nullptr, &solution_count));
        solutions_that_solve.resize(solution_count);
        if(solution_count > 0)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_batched_ex_get_solutions(GEMM_EX_SOL_ARGS,
                                                                      rocblas_gemm_flags_none,
                                                                      solutions_that_solve.data(),
                                                                      &solution_count));

            // append default
            solutions_that_solve.push_back(0);
        }
    }
#endif

    for(auto sol : solutions_that_solve)
    {
        if(compare_solutions)
        {
            run_arg.solution_index = sol;
        }

        testing_gemm_batched_ex_run<Ti, To, Tc>(*arguments);
    }

#undef GEMM_EX_SOL_ARGS
}
