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

#include "check_numerics_matrix.hpp"
#include "frequency_monitor.hpp"
#include "testing_common.hpp"

#include <rocblas/internal/rocblas-beta.h>

/* ============================================================================================ */
template <typename Ti, typename To, typename Tc>
void testing_gemm_ex_bad_arg(const Arguments& arg)
{
    for(auto pointer_mode : {rocblas_pointer_mode_host, rocblas_pointer_mode_device})
    {
        auto rocblas_gemm_ex_fn
            = arg.api & c_API_FORTRAN ? rocblas_gemm_ex_fortran : rocblas_gemm_ex;
        auto rocblas_gemm_ex_fn_64
            = arg.api & c_API_FORTRAN ? rocblas_gemm_ex_64_fortran : rocblas_gemm_ex_64;

        const rocblas_operation transA = rocblas_operation_none;
        const rocblas_operation transB = rocblas_operation_none;

        const int64_t M = 100;
        const int64_t N = 100;
        const int64_t K = 101;

        const int64_t lda = 101;
        const int64_t ldb = 101;
        const int64_t ldc = 101;
        const int64_t ldd = 101;

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

        const rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        // we still have to copy C to D for some quick returns

        int64_t Kmax  = std::max(K, int64_t(1));
        int64_t A_row = transA == rocblas_operation_none ? M : Kmax;
        int64_t A_col = transA == rocblas_operation_none ? Kmax : M;
        int64_t B_row = transB == rocblas_operation_none ? Kmax : N;
        int64_t B_col = transB == rocblas_operation_none ? N : Kmax;

        int32_t  solution_index = 0;
        uint32_t flags          = 0;

        rocblas_local_handle handle{arg};
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, pointer_mode));

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<Ti>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<Ti>, dB, (B_row, B_col, ldb));
        DEVICE_MEMCHECK(device_matrix<To>, dC, (M, N, ldc));
        DEVICE_MEMCHECK(device_matrix<To>, dD, (M, N, ldd));

        // host
        HOST_MEMCHECK(host_matrix<To>, hC, (M, N, ldc));
        rocblas_seedrand();
        rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
        dC.transfer_from(hC);

        // clang-format off

// check for invalid enum
DAPI_EXPECT(rocblas_status_invalid_value, rocblas_gemm_ex_fn, (handle, (rocblas_operation) rocblas_side_both, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_value, rocblas_gemm_ex_fn, (handle, transA, (rocblas_operation) rocblas_side_both, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

// check for invalid size
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, -1, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, M, -1, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, -1,
nullptr, nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

// check for invalid leading dimension
DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, a_type, M-1, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, rocblas_operation_none, rocblas_operation_none, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, K-1, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, a_type, K-1, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, rocblas_operation_transpose, rocblas_operation_transpose, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, N-1, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, M-1,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, M-1, compute_type, algo, solution_index, flags));

// check that nullptr gives rocblas_status_invalid_handle or rocblas_status_invalid_pointer
DAPI_EXPECT(rocblas_status_invalid_handle, rocblas_gemm_ex_fn, (nullptr, transA, transB, M, N, K, alpha,
dA, a_type, lda, dB, b_type, ldb, beta, dC, c_type, ldc,
dD,d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, nullptr,
dA, a_type, lda, dB, b_type, ldb, beta, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, alpha,
nullptr, a_type, lda, dB, b_type, ldb, beta, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, nullptr, b_type, ldb, beta, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, dB, b_type, ldb, nullptr, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, dB, b_type, ldb, beta, nullptr, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

DAPI_EXPECT(rocblas_status_invalid_pointer, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, dB, b_type, ldb, beta, dC, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));


DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, alpha,
dA, a_type, lda, dB, b_type, ldb, beta, dC, c_type, ldc, dC, // aliased C
d_type, ldc + 1, compute_type, algo, solution_index, flags));

// If M==0, then all pointers can be nullptr without issue
DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, 0, N, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

// If N==0, then all pointers can be nullptr without issue
DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, 0, K, nullptr,
nullptr, a_type, lda, nullptr, b_type, ldb, nullptr, nullptr, c_type, ldc,
nullptr, d_type, ldd, compute_type, algo, solution_index, flags));

// If alpha==0 then A, B can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, nullptr, b_type, ldb, beta, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

// the following tests still output to D

// If K==0, then alpha, A and B can both be nullptr without issue.
DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, 0,
nullptr, nullptr, a_type, lda, nullptr, b_type, ldb, beta, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

// If alpha==0, then A and B can both be nullptr without issue.
DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, nullptr, b_type, ldb, beta, dC, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

// alpha==0 && beta==1 must still copy C to D so no quick return

// If alpha==0 && beta==0 then A, B and C can be nullptr without issue.
DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, zero,
nullptr, a_type, lda, nullptr, b_type, ldb, zero, nullptr, c_type, ldc,
dD, d_type, ldd, compute_type, algo, solution_index, flags));

        // clang-format on
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_ex_run(const Arguments& arg)
{
    auto rocblas_gemm_ex_fn = arg.api & c_API_FORTRAN ? rocblas_gemm_ex_fortran : rocblas_gemm_ex;
    auto rocblas_gemm_ex_fn_64
        = arg.api & c_API_FORTRAN ? rocblas_gemm_ex_64_fortran : rocblas_gemm_ex_64;

    rocblas_gemm_algo algo = rocblas_gemm_algo(arg.algo);
    int32_t           solution_index(arg.solution_index);
    uint32_t          flags(arg.flags);

    bool alpha_isnan = arg.alpha_isnan<Tc>();
    bool beta_isnan  = arg.beta_isnan<Tc>();
    if(!std::is_same_v<
           To,
           float> && !std::is_same_v<To, double> && !std::is_same_v<To, rocblas_half> && !rocblas_is_complex<To> && (alpha_isnan || beta_isnan))
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    double rocblas_error          = 0.0;

    rocblas_local_handle handle{arg};
    rocblas_operation    transA = char2rocblas_operation(arg.transA);
    rocblas_operation    transB = char2rocblas_operation(arg.transB);
    int64_t              M = arg.M, N = arg.N, K = arg.K;
    int64_t              lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    int64_t              Kmax   = std::max(K, int64_t(1));
    int64_t              A_row  = transA == rocblas_operation_none ? M : Kmax;
    int64_t              A_col  = transA == rocblas_operation_none ? Kmax : M;
    int64_t              B_row  = transB == rocblas_operation_none ? Kmax : N;
    int64_t              B_col  = transB == rocblas_operation_none ? N : Kmax;
    rocblas_datatype     d_type = arg.d_type;

    rocblas_math_mode math_mode = rocblas_math_mode(arg.math_mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_math_mode(handle, math_mode));
    CHECK_ROCBLAS_ERROR(rocblas_get_math_mode(handle, &math_mode));

    // check for invalid sizes
    bool invalid_size = M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || ldd < M;
    if(invalid_size)
    {
        // clang-format off
        DAPI_EXPECT(rocblas_status_invalid_size, rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, nullptr,
                                                 nullptr, arg.a_type, lda,
                                                 nullptr, arg.b_type, ldb, nullptr,
                                                 nullptr, arg.c_type, ldc,
                                                 nullptr, arg.d_type, ldd,
                                                 arg.compute_type, algo, solution_index, flags));
        // clang-format on
        return;
    }

#ifdef ROCBLAS_BENCH
    if(rocblas_internal_tensile_debug_skip_launch())
    {
        DEVICE_MEMCHECK(device_vector<Ti>, dA, (1));
        DEVICE_MEMCHECK(device_vector<Ti>, dB, (1));
        DEVICE_MEMCHECK(device_vector<To>, dC, (1));
        DEVICE_MEMCHECK(device_vector<To>, dD, (1));
        // clang-format off
        DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                               dA, arg.a_type, lda,
                                               dB, arg.b_type, ldb, &h_beta_Tc,
                                               dC, arg.c_type, ldc,
                                               dD, arg.d_type, ldd,
                                               arg.compute_type, algo, solution_index, flags));
        // clang-format on
        return;
    }
#endif
    // update after invalid checks
    if(!arg.outofplace)
    {
        ldd    = ldc;
        d_type = arg.c_type;
    }

    // Allocate device memory
    DEVICE_MEMCHECK(device_vector<Tc>, d_alpha_Tc, (1));
    DEVICE_MEMCHECK(device_vector<Tc>, d_beta_Tc, (1));

    bool alt       = (rocblas_gemm_flags_fp16_alt_impl & flags);
    bool alt_round = (rocblas_gemm_flags_fp16_alt_impl_rnz & flags);

    if(arg.unit_check || arg.norm_check)
    {
        // Allocate host memory
        HOST_MEMCHECK(host_matrix<Ti>, hA, (A_row, A_col, lda));
        HOST_MEMCHECK(host_matrix<Ti>, hB, (B_row, B_col, ldb));
        HOST_MEMCHECK(host_matrix<To>, hC, (M, N, ldc));

        // Allocate device memory
        DEVICE_MEMCHECK(device_matrix<Ti>, dA, (A_row, A_col, lda));
        DEVICE_MEMCHECK(device_matrix<Ti>, dB, (B_row, B_col, ldb));

        // if C!=D, allocate C and D normally
        // if C==D, allocate C big enough for the larger of C and D; D points to C
        DEVICE_MEMCHECK(device_matrix<To>, dC, (M, N, ldc));
        device_matrix<To> dD
            = (arg.outofplace) ? device_matrix<To>(M, N, ldd) : device_matrix<To>(0, 1, 1);
        CHECK_DEVICE_ALLOCATION(dD.memcheck());
        device_matrix<To>& dDref = (arg.outofplace) ? dD : dC;
        DEVICE_MEMCHECK(device_vector<Tc>, d_alpha_Tc, (1));
        DEVICE_MEMCHECK(device_vector<Tc>, d_beta_Tc, (1));

        // Initialize data on host memory
        rocblas_init_matrix<Ti>(
            hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix<Ti, true>(
            hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
        rocblas_init_matrix<To, true>(
            hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dC.transfer_from(hC));

        using To_hpa = std::conditional_t<std::is_same_v<To, rocblas_bfloat16>, float, To>;

        HOST_MEMCHECK(host_matrix<To>, hD_1, (M, N, ldd));
        HOST_MEMCHECK(host_matrix<To>, hD_2, (M, N, ldd));
        HOST_MEMCHECK(host_matrix<To_hpa>, hD_gold, (M, N, ldd));

        rocblas_init_nan<To>(hD_1, M, N, ldd);
        rocblas_init_nan<To>(hD_2, M, N, ldd);
        rocblas_init_nan<To_hpa>(hD_gold, M, N, ldd);

        if(arg.pointer_mode_host)
        {
            // ROCBLAS rocblas_pointer_mode_host
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            handle.pre_test(arg);
            // clang-format off
            DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                               dA, arg.a_type, lda,
                                               dB, arg.b_type, ldb, &h_beta_Tc,
                                               dC, arg.c_type, ldc,
                                               dDref,  d_type, ldd,
                                               arg.compute_type, algo, solution_index, flags));
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
            DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, d_alpha_Tc,
                                               dA, arg.a_type, lda,
                                               dB, arg.b_type, ldb, d_beta_Tc,
                                               dC, arg.c_type, ldc,
                                               dDref,  d_type, ldd,
                                               arg.compute_type, algo, solution_index, flags));
            // clang-format on

            CHECK_HIP_ERROR(hD_2.transfer_from(dDref));

            if(arg.repeatability_check)
            {
                HOST_MEMCHECK(host_matrix<To>, hD_copy, (M, N, ldd));

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
                    DEVICE_MEMCHECK(device_matrix<Ti>, dB_copy, (B_row, B_col, ldb));

                    // if C!=D, allocate C and D normally
                    // if C==D, allocate C big enough for the larger of C and D; D points to C
                    DEVICE_MEMCHECK(device_matrix<To>, dC_copy, (M, N, ldc));
                    device_matrix<To> dD_copy = (arg.outofplace) ? device_matrix<To>(M, N, ldd)
                                                                 : device_matrix<To>(0, 1, 1);
                    CHECK_DEVICE_ALLOCATION(dD_copy.memcheck());
                    device_matrix<To>& dDref_copy = (arg.outofplace) ? dD_copy : dC_copy;

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
                    DAPI_CHECK(rocblas_gemm_ex_fn, (handle_copy, transA, transB, M, N, K, d_alpha_Tc_copy,
                                               dA_copy, arg.a_type, lda,
                                               dB_copy, arg.b_type, ldb, d_beta_Tc_copy,
                                               dC_copy, arg.c_type, ldc,
                                               dDref_copy,  d_type, ldd,
                                               arg.compute_type, algo, solution_index, flags));
                        // clang-format on
                        // fetch device mode GPU results
                        CHECK_HIP_ERROR(hD_copy.transfer_from(dDref_copy));

                        unit_check_general<To>(M, N, ldd, hD_2, hD_copy);
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
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hA.data(), hA.size());
            type_to_xdl_math_op_type<rocblas_xfloat32, float>(hB.data(), hB.size());
        }

        // CPU BLAS
        cpu_time_used = get_time_us_no_sync();

        ref_gemm<Ti, To_hpa, Tc>(
            transA,
            transB,
            M,
            N,
            K,
            h_alpha_Tc,
            hA,
            lda,
            hB,
            ldb,
            h_beta_Tc,
            (To_hpa*)hD_gold,
            ldd,
            alt ? (alt_round ? rocblas_bfloat16::rocblas_truncate_t::rocblas_round_near_zero
                             : rocblas_bfloat16::rocblas_truncate_t::rocblas_truncate)
                : rocblas_bfloat16::rocblas_truncate_t::rocblas_round_near_even);

        cpu_time_used = get_time_us_no_sync() - cpu_time_used;

        if(arg.pointer_mode_host)
        {
            if(arg.unit_check)
            {
                if((rocblas_handle(handle)->getArchMajor() == 11) && (sizeof(Ti) == 2))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
                }
                else if(std::is_same_v<Tc, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<Tc>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1, tol);
                }
                else
                {
                    unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_1);
                }
            }

            if(arg.norm_check)
            {
                auto err1
                    = std::abs(norm_check_general<To>('F', M, N, ldd, (To_hpa*)hD_gold, (To*)hD_1));

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
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, tol);
                }
                else if(std::is_same_v<Tc, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<Tc>;
                    near_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2, tol);
                }
                else
                {
                    unit_check_general<To, To_hpa>(M, N, ldd, hD_gold, hD_2);
                }
            }

            if(arg.norm_check)
            {
                auto err1
                    = std::abs(norm_check_general<To>('F', M, N, ldd, (To_hpa*)hD_gold, (To*)hD_2));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        // Information on flush_memory_size and flush_batch_count
        // - To time gemm_ex it is called number_hot_calls times.
        // - if the size of dA, dB, dC, dD are small enough they will be cached
        //   and reused number_hot_calls-1 times.
        // - This "hot-cache" timing will give higher performance than if the
        //   cache is flushed
        // - arg.flush_batch_count or arg.flush_memory_size can be used to avoid caching of dA, dB, dC, dD
        // - if arg.flush_memory_size is specified, then flush_batch_count is calculated
        // - only one of arg.flush_memory_size or arg.flush_batch_count can be used, not both
        // - Note that this is only used in timing code, not in testing code.
        // - The method is as outlined in
        //   "Achieving accurate and context-sensitive timing for code optimization" by Whaley and Castaldo.
        // - In the number_hot_calls timing loop it cycles through the arg.flush_batch_count copies
        //   of dA, dB, dC, dD, and if flush_memory_size is large enough they will be evicted
        //   from cache before they are reused.
        // - The individual matrices are aligned on the same byte boundaries provided by hipMalloc.

        static std::unique_ptr<device_strided_batch_matrix<Ti>> dA_alloc;
        static std::unique_ptr<device_strided_batch_matrix<Ti>> dB_alloc;
        static std::unique_ptr<device_strided_batch_matrix<To>> dC_alloc;
        static std::unique_ptr<device_strided_batch_matrix<To>> dD_alloc;

        rocblas_stride stride_a = lda * A_col;
        rocblas_stride stride_b = ldb * B_col;
        rocblas_stride stride_c = ldc * N;
        rocblas_stride stride_d = arg.outofplace ? ldd * N : 0;

        rocblas_stride aligned_stride_a = align_stride<Ti>(stride_a);
        rocblas_stride aligned_stride_b = align_stride<Ti>(stride_b);
        rocblas_stride aligned_stride_c = align_stride<To>(stride_c);
        rocblas_stride aligned_stride_d = align_stride<To>(stride_d);

        size_t a_size = M * K * sizeof(Ti);
        size_t b_size = K * N * sizeof(Ti);
        size_t c_size = M * N * sizeof(To);
        //      exclude d_size from cached_size calculation because
        //      - for arg.outofplace == false : D == C
        //      - for arg.outofplace == true  : D is write only
        size_t a_b_c_cached_size = a_size + b_size + c_size;

        size_t flush_batch_count = calculate_flush_batch_count(
            arg.flush_batch_count, arg.flush_memory_size, a_b_c_cached_size);

        // Allocate extra memory (then resize), so later runs are more likely to be able to reuse the memory.
        float extra_memory = 1.2;

        if(!dA_alloc || !dA_alloc->resize(A_row, A_col, lda, aligned_stride_a, flush_batch_count))
        {
            dA_alloc = std::make_unique<device_strided_batch_matrix<Ti>>(aligned_stride_a,
                                                                         1,
                                                                         aligned_stride_a,
                                                                         aligned_stride_a,
                                                                         flush_batch_count
                                                                             * extra_memory);
            rocblas_init_matrix<Ti>(handle,
                                    *dA_alloc,
                                    arg,
                                    rocblas_client_alpha_sets_nan,
                                    rocblas_client_general_matrix,
                                    true);
            dA_alloc->resize(A_row, A_col, lda, aligned_stride_a, flush_batch_count);
        }
        if(!dB_alloc || !dB_alloc->resize(B_row, B_col, ldb, aligned_stride_b, flush_batch_count))
        {
            dB_alloc = std::make_unique<device_strided_batch_matrix<Ti>>(aligned_stride_b,
                                                                         1,
                                                                         aligned_stride_b,
                                                                         aligned_stride_b,
                                                                         flush_batch_count
                                                                             * extra_memory);
            rocblas_init_matrix<Ti>(handle,
                                    *dB_alloc,
                                    arg,
                                    rocblas_client_alpha_sets_nan,
                                    rocblas_client_general_matrix,
                                    false,
                                    true);
            dB_alloc->resize(B_row, B_col, ldb, aligned_stride_b, flush_batch_count);
        }
        if(!dC_alloc || !dC_alloc->resize(M, N, ldc, aligned_stride_c, flush_batch_count))
        {
            dC_alloc = std::make_unique<device_strided_batch_matrix<To>>(aligned_stride_c,
                                                                         1,
                                                                         aligned_stride_c,
                                                                         aligned_stride_c,
                                                                         flush_batch_count
                                                                             * extra_memory);
            rocblas_init_matrix<To>(handle,
                                    *dC_alloc,
                                    arg,
                                    rocblas_client_beta_sets_nan,
                                    rocblas_client_general_matrix);
            dC_alloc->resize(M, N, ldc, aligned_stride_c, flush_batch_count);
        }
        if(arg.outofplace)
        {
            if(!dD_alloc || !dD_alloc->resize(M, N, ldd, aligned_stride_d, flush_batch_count))
            {
                dD_alloc = std::make_unique<device_strided_batch_matrix<To>>(
                    M, N, ldd, aligned_stride_d, flush_batch_count * extra_memory);
                dD_alloc->resize(M, N, ldd, aligned_stride_d, flush_batch_count);
            }
        }

        auto& dA = *dA_alloc;
        auto& dB = *dB_alloc;
        auto& dC = *dC_alloc;
        auto& dD = (arg.outofplace) ? *dD_alloc : dC;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        // if solution index is given check it is valid before benchmarking
        if(algo == rocblas_gemm_algo_solution_index && solution_index != 0)
        {
            uint32_t test_flags = flags | rocblas_gemm_flags_check_solution_index;
            // clang-format off
            DAPI_CHECK(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                                   dA[0], arg.a_type, lda,
                                                   dB[0], arg.b_type, ldb, &h_beta_Tc,
                                                   dC[0], arg.c_type, ldc,
                                                   dD[0],     d_type, ldd,
                                                   arg.compute_type, algo, solution_index, test_flags));
            // clang-format on
        }

        for(int i = 0; i < number_cold_calls; i++)
        {
            // clang-format off
            DAPI_DISPATCH(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                                   dA[0], arg.a_type, lda,
                                                   dB[0], arg.b_type, ldb, &h_beta_Tc,
                                                   dC[0], arg.c_type, ldc,
                                                   dD[0],     d_type, ldd,
                                                   arg.compute_type, algo, solution_index, flags));
            // clang-format on
        }

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        FrequencyMonitor& freq_monitor = getFrequencyMonitor();
        freq_monitor.start();
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            int flush_index = (i + 1) % flush_batch_count;
            // clang-format off
            if(arg.outofplace)
            {
                DAPI_DISPATCH(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                   dA[flush_index], arg.a_type, lda,
                                   dB[flush_index], arg.b_type, ldb, &h_beta_Tc,
                                   dC[flush_index], arg.c_type, ldc,
                                   dD[          0],     d_type, ldd,
                                   arg.compute_type, algo, solution_index, flags));
            }
            else
            {
                DAPI_DISPATCH(rocblas_gemm_ex_fn, (handle, transA, transB, M, N, K, &h_alpha_Tc,
                                   dA[flush_index], arg.a_type, lda,
                                   dB[flush_index], arg.b_type, ldb, &h_beta_Tc,
                                   dC[flush_index], arg.c_type, ldc,
                                   dD[flush_index],     d_type, ldd,
                                   arg.compute_type, algo, solution_index, flags));
            }
            // clang-format on
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        freq_monitor.stop();

        if(!arg.outofplace)
        {
            if constexpr(!std::is_same<To, int>::value)
            {
                int check_numerics
                    = rocblas_check_numerics_mode_only_nan_inf | rocblas_check_numerics_mode_fail;
                // Only check the last computed output matrix.
                int  flush_index = (number_hot_calls - 1 + 1) % flush_batch_count;
                auto nan_or_inf_check
                    = rocblas_internal_check_numerics_matrix_template("testing_gemm_ex",
                                                                      handle,
                                                                      rocblas_operation_none,
                                                                      rocblas_fill_full,
                                                                      rocblas_client_general_matrix,
                                                                      M,
                                                                      N,
                                                                      dC[flush_index],
                                                                      0,
                                                                      ldc,
                                                                      aligned_stride_c,
                                                                      1,
                                                                      check_numerics,
                                                                      false);
                if(nan_or_inf_check == rocblas_status_check_numerics_fail)
                {
                    rocblas_cerr << "Warning: Output matrix contains Nan or Inf!" << std::endl;
                }
            }
        }

        if(arg.cleanup)
        {
            dA_alloc.reset();
            dB_alloc.reset();
            dC_alloc.reset();
            dD_alloc.reset();
        }
        else if(!arg.outofplace)
        {
            rocblas_init_matrix<To>(
                handle, dC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
        }

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
            .log_args<Tc>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          gemm_gflop_count<Tc>(M, N, K),
                          ArgumentLogging::NA_value,
                          cpu_time_used,
                          rocblas_error,
                          ArgumentLogging::NA_value,
                          ArgumentLogging::NA_value,
                          ArgumentLogging::NA_value);
    }
#undef GEMM_EX_ARGS
#undef rocblas_gemm_exM
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_ex(const Arguments& arg)
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
        run_arg.d_type, run_arg.ldd, arg.compute_type, algo

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(
            GEMM_EX_SOL_ARGS, rocblas_gemm_flags_none, nullptr, &solution_count));
        solutions_that_solve.resize(solution_count);
        if(solution_count > 0)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(GEMM_EX_SOL_ARGS,
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

        testing_gemm_ex_run<Ti, To, Tc>(*arguments);
    }

#undef GEMM_EX_SOL_ARGS
}
