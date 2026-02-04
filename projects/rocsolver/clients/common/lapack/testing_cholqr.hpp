/* **************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

#include <iomanip>
#include <iostream>

template <bool STRIDED, typename T, typename I, typename S, typename U, typename V>
void cholqr_checkBadArgs(const rocblas_handle handle,
                         const I m,
                         const I n,
                         T dA,
                         const I lda,
                         const rocblas_stride stA,
                         U dR,
                         const I ldr,
                         const rocblas_stride stR,
                         S dSigma,
                         const rocsolver_cholqr_algo algo,
                         V dInfo,
                         const I bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, nullptr, m, n, dA, lda, stA, dR, ldr, stR,
                                           dSigma, algo, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                               dSigma, algo, dInfo, (I)-1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, (T) nullptr, lda, stA, dR, ldr,
                                           stR, dSigma, algo, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, (U) nullptr, ldr,
                                           stR, dSigma, algo, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                           (S) nullptr, algo, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                           dSigma, algo, (V) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, (I)0, n, (T) nullptr, lda, stA,
                                           (U) nullptr, ldr, stR, dSigma, algo, dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, (I)0, (T) nullptr, lda, stA,
                                           (U) nullptr, ldr, stR, dSigma, algo, dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA, lda, stA, dR, ldr, stR,
                                               (S) nullptr, algo, (V) nullptr, (I)0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_cholqr_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    I m = 1;
    I n = 1;
    I lda = 1;
    I ldr = 1;
    rocblas_stride stA = 1;
    rocblas_stride stR = 1;
    I bc = 1;
    rocsolver_cholqr_algo algo = rocsolver_cholqr_cholqr3_compute;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<T> dR(1, 1, stR, 1);
        device_strided_batch_vector<S> dSigma(1, 1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        cholqr_checkBadArgs<STRIDED>(handle, m, n, dA.data(), lda, stA, dR.data(), ldr, stR,
                                     dSigma.data(), algo, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<T> dR(1, 1, 1, 1);
        device_strided_batch_vector<S> dSigma(1, 1, 1, 1);
        device_strided_batch_vector<I> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        cholqr_checkBadArgs<STRIDED>(handle, m, n, dA.data(), lda, stA, dR.data(), ldr, stR,
                                     dSigma.data(), algo, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Ud, typename Sd, typename Th, typename Uh, typename Sh>
void cholqr_initData(const rocblas_handle handle,
                     const I m,
                     const I n,
                     Td& dA,
                     const I lda,
                     const rocblas_stride stA,
                     Ud& dR,
                     const I ldr,
                     const rocblas_stride stR,
                     Sd& dSigma,
                     const rocsolver_cholqr_algo algo,
                     const I bc,
                     Th& hA,
                     Uh& hR,
                     Sh& hSigma)
{
    using S = decltype(std::real(T{}));

    if(CPU)
    {
        rocblas_init<T>(hA, true);

        for(I b = 0; b < bc; ++b)
        {
            for(I i = 0; i < m; i++)
            {
                for(I j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }

        // Initialize sigma array based on algorithm mode
        // For cholqr3_user, we compute sigma using the same formula as cholqr3_compute:
        //   sigma = 11 * n * eps * (m + (n+1)) * ||A||_F^2
        // For other modes, sigma is either not used or computed internally
        if(algo == rocsolver_cholqr_cholqr3_user)
        {
            S eps = std::numeric_limits<S>::epsilon();
            for(I b = 0; b < bc; ++b)
            {
                // Compute ||A||_F^2 (Frobenius norm squared)
                S gnorm = snorm('F', m, n, hA[b], lda);

                // sigma = 11 * n * eps * (m + (n+1)) * gnorm_sq
                hSigma[b][0] = S(11.0) * S(n) * eps * (S(m) + S(n + 1)) * gnorm * gnorm;
            }
        }
        else
        {
            // For other algorithms, sigma is not used or computed internally
            for(I b = 0; b < bc; ++b)
            {
                hSigma[b][0] = S(0.0);
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dSigma.transfer_from(hSigma));
    }
}

template <bool STRIDED,
          typename T,
          typename I,
          typename Td,
          typename Ud,
          typename Sd,
          typename Id,
          typename Th,
          typename Uh,
          typename Sh,
          typename Ih>
void cholqr_getError(const rocblas_handle handle,
                     const I m,
                     const I n,
                     Td& dA,
                     const I lda,
                     const rocblas_stride stA,
                     Ud& dR,
                     const I ldr,
                     const rocblas_stride stR,
                     Sd& dSigma,
                     const rocsolver_cholqr_algo algo,
                     Id& dInfo,
                     const I bc,
                     Th& hA,
                     Th& hARes,
                     Uh& hR,
                     Uh& hRRes,
                     Sh& hSigma,
                     Ih& hInfo,
                     double* max_err)
{
    using S = decltype(std::real(T{}));
    I min_mn = std::min(m, n);

    // Work arrays for CPU reference (GEQRF + ORGQR)
    I lwork = std::max(I(1), std::max(m, n) * 32);
    std::vector<T> hW(lwork);
    std::vector<T> hTau(min_mn);

    // input data initialization
    cholqr_initData<true, true, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc, hA,
                                   hR, hSigma);

    // execute GPU cholqr
    CHECK_ROCBLAS_ERROR(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA, dR.data(), ldr,
                                         stR, dSigma.data(), algo, dInfo.data(), bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA)); // Q_gpu is stored in A
    CHECK_HIP_ERROR(hRRes.transfer_from(dR)); // R_gpu
    CHECK_HIP_ERROR(hInfo.transfer_from(dInfo));

    // Check info for factorization success
    // (We expect the used input matrices to always succeed. The test matrices
    // are well-conditioned with large diagonal dominance.)
    *max_err = 0;
    for(I b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], 0) << "where b = " << b;
        if(hInfo[b][0] != 0)
            *max_err += 1;
    }

    // Compute Error
    // ||Q^T Q - I||_F (orthogonality) and ||A - QR||_F (reconstruction)
    double err = 0;
    for(I b = 0; b < bc; ++b)
    {
        // Only compute numerical error if factorization succeeded
        if(hInfo[b][0] == 0)
        {
            // Compute Q_gpu^H * Q_gpu using GEMM (should be identity for orthogonal Q)
            // QtQ = Q^H * Q, result is min_mn x min_mn
            std::vector<T> QtQ(size_t(min_mn) * min_mn, T(0));
            I const ldQtQ = min_mn;

            // QtQ <- identity
            cpu_laset('A', min_mn, min_mn, T(0), T(1), QtQ.data(), ldQtQ);

            // Compute QtQ <-  Q' * Q - identity
            rocblas_operation transQ = rocblas_operation_conjugate_transpose;
            cpu_gemm(rocblas_operation_conjugate_transpose, rocblas_operation_none, min_mn, min_mn,
                     m, T(1), hARes[b], lda, hARes[b], lda, T(-1), QtQ.data(), ldQtQ);

            // -----------------------
            // Compute orth_err = norm( Q^H Q - I )
            // -----------------------
            S orth_err = snorm('F', min_mn, min_mn, QtQ.data(), ldQtQ);

            // Compute Q_gpu * R_gpu using GEMM (should equal original A)
            // QR = Q * R, result is m x n
            // Note: R may have garbage below diagonal from GPU, so extract only upper triangular

            std::vector<T> R_gpu_clean(size_t(ldr) * n, T(0));

            //  ------------------------------
            //  Copy upper triangular part
            //  R_gpu_clean = triu( hRRes[b] )
            //  ------------------------------
            cpu_lacpy('U', min_mn, min_mn, hRRes[b], ldr, R_gpu_clean.data(), ldr);

            std::vector<T> QR(size_t(lda) * n, T(0));
            I ldQR = lda;

            // -------------------------
            // compute  QR <-  Q * R - A
            // -------------------------
            cpu_lacpy('A', m, n, hA[b], lda, QR.data(), ldQR);

            cpu_gemm(rocblas_operation_none, rocblas_operation_none, m, n, min_mn, T(1), hARes[b],
                     lda, R_gpu_clean.data(), ldr, T(-1), QR.data(), ldQR);

            // Compute recon_err = norm(A - QR)
            // Note: Must iterate column by column to respect leading dimension lda

            auto recon_err = snorm('F', m, n, QR.data(), ldQR);

            err = std::max(orth_err, recon_err);
            *max_err = err > *max_err ? err : *max_err;
        }
    }
}

template <bool STRIDED,
          typename T,
          typename I,
          typename Td,
          typename Ud,
          typename Sd,
          typename Id,
          typename Th,
          typename Uh,
          typename Sh,
          typename Ih>
void cholqr_getPerfData(const rocblas_handle handle,
                        const I m,
                        const I n,
                        Td& dA,
                        const I lda,
                        const rocblas_stride stA,
                        Ud& dR,
                        const I ldr,
                        const rocblas_stride stR,
                        Sd& dSigma,
                        const rocsolver_cholqr_algo algo,
                        Id& dInfo,
                        const I bc,
                        Th& hA,
                        Uh& hR,
                        Sh& hSigma,
                        Ih& hInfo,
                        double* gpu_time_used,
                        double* cpu_time_used,
                        const rocblas_int hot_calls,
                        const int profile,
                        const bool profile_kernels,
                        const bool perf)
{
    using S = decltype(std::real(T{}));
    std::vector<T> hW(n);
    std::vector<T> hTau(std::min(m, n));

    if(!perf)
    {
        cholqr_initData<true, false, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc,
                                        hA, hR, hSigma);

        // cpu-lapack performance (using GEQRF as reference, only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(I b = 0; b < bc; ++b)
        {
            cpu_geqrf(m, n, hA[b], lda, hTau.data(), hW.data(), n);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    cholqr_initData<true, false, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc, hA,
                                    hR, hSigma);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        cholqr_initData<false, true, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc,
                                        hA, hR, hSigma);

        CHECK_ROCBLAS_ERROR(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA, dR.data(),
                                             ldr, stR, dSigma.data(), algo, dInfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        cholqr_initData<false, true, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo, bc,
                                        hA, hR, hSigma);

        timer.start(stream);
        rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA, dR.data(), ldr, stR,
                         dSigma.data(), algo, dInfo.data(), bc);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
void testing_cholqr(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I lda = argus.get<I>("lda", m);
    I ldr = argus.get<I>("ldr", n);
    rocblas_stride stA = argus.get<rocblas_stride>("strideA", lda * n);
    rocblas_stride stR = argus.get<rocblas_stride>("strideR", ldr * n);

    // Get cholqr algorithm from command line
    char algo_char = argus.get<char>("cholqr_algo", 'D');
    rocsolver_cholqr_algo algo = char2rocsolver_cholqr_algo(algo_char);

    I bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    rocblas_stride stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    rocblas_stride stRRes = (argus.unit_check || argus.norm_check) ? stR : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A = size_t(lda) * n;
    size_t size_R = size_t(ldr) * n;
    size_t size_Sigma = 1; // one sigma per batch
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_RRes = (argus.unit_check || argus.norm_check) ? size_R : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || ldr < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, (T* const*)nullptr, lda,
                                                   stA, (T*)nullptr, ldr, stR, (S*)nullptr, algo,
                                                   (I*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, (T*)nullptr, lda, stA,
                                                   (T*)nullptr, ldr, stR, (S*)nullptr, algo,
                                                   (I*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_cholqr(STRIDED, handle, m, n, (T* const*)nullptr, lda, stA,
                                               (T*)nullptr, ldr, stR, (S*)nullptr, algo,
                                               (I*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_cholqr(STRIDED, handle, m, n, (T*)nullptr, lda, stA,
                                               (T*)nullptr, ldr, stR, (S*)nullptr, algo,
                                               (I*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hARes(size_ARes, 1, bc);
        host_strided_batch_vector<T> hR(size_R, 1, stR, bc);
        host_strided_batch_vector<T> hRRes(size_RRes, 1, stR, bc);
        host_strided_batch_vector<S> hSigma(size_Sigma, 1, size_Sigma, bc);
        host_strided_batch_vector<I> hInfo(1, 1, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        device_strided_batch_vector<T> dR(size_R, 1, stR, bc);
        device_strided_batch_vector<S> dSigma(size_Sigma, 1, size_Sigma, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_R)
            CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA,
                                                   dR.data(), ldr, stR, dSigma.data(), algo,
                                                   dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                        dInfo, bc, hA, hARes, hR, hRRes, hSigma, hInfo, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                           dInfo, bc, hA, hR, hSigma, hInfo, &gpu_time_used,
                                           &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T> hR(size_R, 1, stR, bc);
        host_strided_batch_vector<T> hRRes(size_RRes, 1, stRRes, bc);
        host_strided_batch_vector<S> hSigma(size_Sigma, 1, size_Sigma, bc);
        host_strided_batch_vector<I> hInfo(1, 1, 1, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dR(size_R, 1, stR, bc);
        device_strided_batch_vector<S> dSigma(size_Sigma, 1, size_Sigma, bc);
        device_strided_batch_vector<I> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_R)
            CHECK_HIP_ERROR(dR.memcheck());
        CHECK_HIP_ERROR(dSigma.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check quick return
        if(m == 0 || n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_cholqr(STRIDED, handle, m, n, dA.data(), lda, stA,
                                                   dR.data(), ldr, stR, dSigma.data(), algo,
                                                   dInfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                rocsolver_bench_inform(inform_quick_return);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
            cholqr_getError<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                        dInfo, bc, hA, hARes, hR, hRRes, hSigma, hInfo, &max_error);

        // collect performance data
        if(argus.timing && hot_calls > 0)
            cholqr_getPerfData<STRIDED, T>(handle, m, n, dA, lda, stA, dR, ldr, stR, dSigma, algo,
                                           dInfo, bc, hA, hR, hSigma, hInfo, &gpu_time_used,
                                           &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);
    }

    // Use tolerance based on algorithm and problem size
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 10 * m * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            if(BATCHED)
            {
                rocsolver_bench_output("m", "n", "lda", "ldr", "strideR", "algo", "batch_c");
                rocsolver_bench_output(m, n, lda, ldr, stR, algo_char, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "ldr", "strideR", "algo",
                                       "batch_c");
                rocsolver_bench_output(m, n, lda, stA, ldr, stR, algo_char, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda", "ldr", "algo");
                rocsolver_bench_output(m, n, lda, ldr, algo_char);
            }
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_CHOLQR(...) extern template void testing_cholqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CHOLQR,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_SCALAR_TYPE,
            FOREACH_INT_TYPE,
            APPLY_STAMP)
